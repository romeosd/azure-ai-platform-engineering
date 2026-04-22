"""
Azure AI Search — hybrid RAG retrieval client.

Provides:
- Vector search (pure semantic similarity)
- Keyword search (BM25 full-text)
- Hybrid search (vector + keyword with RRF fusion)
- Semantic re-ranking (L2 re-ranker)
- Faceted search and metadata filtering
- Document indexing and batch upsert
- Index lifecycle management
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """A single result from Azure AI Search."""

    content: str
    score: float
    reranker_score: float = 0.0
    id: str = ""
    source: str = ""
    title: str = ""
    captions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexStats:
    """Statistics about a search index."""

    document_count: int
    storage_size_bytes: int
    index_name: str


class AzureAISearchClient:
    """
    Production Azure AI Search client supporting hybrid retrieval.

    Implements the full retrieval pipeline used in enterprise RAG:
    vector search for semantic similarity, BM25 for keyword matching,
    RRF fusion for hybrid ranking, and L2 semantic re-ranking.

    Example:
        search = AzureAISearchClient()

        # Hybrid search with semantic re-ranking (best for RAG)
        results = search.hybrid_search(
            query="What are the data retention requirements?",
            embedding=openai_client.embed("data retention requirements").embedding,
            top_k=10,
        )

        for result in results:
            print(f"[{result.reranker_score:.3f}] {result.title}: {result.content[:100]}")

        # Index new documents
        search.upsert_documents([
            {"id": "doc1", "content": "...", "title": "Policy", "source": "s3://bucket/policy.pdf"}
        ], embedding_field="content")
    """

    def __init__(
        self,
        index_name: str | None = None,
    ) -> None:
        cfg = get_config()
        raw = load_config()
        search_cfg = cfg.ai_search
        raw_search = raw.get("ai_search", {})

        self._endpoint = search_cfg.endpoint
        self._api_key = search_cfg.api_key
        self._api_version = search_cfg.api_version
        self._index_name = index_name or search_cfg.indexes.get("documents", "documents")
        self._retrieval = search_cfg.retrieval
        self._vector = search_cfg.vector

        credential = AzureKeyCredential(self._api_key)

        self._search_client = SearchClient(
            endpoint=self._endpoint,
            index_name=self._index_name,
            credential=credential,
        )
        self._index_client = SearchIndexClient(
            endpoint=self._endpoint,
            credential=credential,
        )

        logger.info(
            "AzureAISearchClient initialised",
            endpoint=self._endpoint,
            index=self._index_name,
        )

    def vector_search(
        self,
        embedding: list[float],
        top_k: int | None = None,
        filter_expr: str | None = None,
        select_fields: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        Pure vector (semantic) search using HNSW ANN index.

        Args:
            embedding: Query vector from Azure OpenAI text-embedding-3-large.
            top_k: Number of results to return.
            filter_expr: OData filter expression (e.g. "category eq 'policy'").
            select_fields: Fields to return (default: all).

        Returns:
            List of SearchResult sorted by cosine similarity.
        """
        k = top_k or self._retrieval.get("top_k", 10)

        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=k,
            fields="content_vector",
        )

        results = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=select_fields,
            top=k,
        )

        return self._parse_results(results)

    def keyword_search(
        self,
        query: str,
        top_k: int | None = None,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """
        BM25 full-text keyword search.

        Args:
            query: The search query string.
            top_k: Number of results.
            filter_expr: OData filter expression.

        Returns:
            List of SearchResult sorted by BM25 score.
        """
        k = top_k or self._retrieval.get("top_k", 10)

        results = self._search_client.search(
            search_text=query,
            query_type=QueryType.FULL,
            filter=filter_expr,
            top=k,
        )

        return self._parse_results(results)

    def hybrid_search(
        self,
        query: str,
        embedding: list[float],
        top_k: int | None = None,
        filter_expr: str | None = None,
        semantic_config: str | None = None,
        use_semantic_reranker: bool = True,
    ) -> list[SearchResult]:
        """
        Hybrid search: vector + BM25 fused with Reciprocal Rank Fusion (RRF),
        optionally re-ranked by Azure's L2 semantic re-ranker.

        This is the recommended retrieval mode for enterprise RAG. The L2
        re-ranker uses a cross-encoder model to rescore results based on
        deep contextual relevance to the query.

        Args:
            query: The search query string (used for BM25 and semantic re-ranking).
            embedding: Query vector for vector search leg.
            top_k: Number of final results.
            filter_expr: OData filter expression.
            semantic_config: Semantic configuration name (overrides config).
            use_semantic_reranker: Enable L2 semantic re-ranking (requires Semantic tier).

        Returns:
            List of SearchResult sorted by semantic re-ranker score (if enabled)
            or RRF fusion score.
        """
        k = top_k or self._retrieval.get("top_k", 10)
        sem_config = semantic_config or self._retrieval.get("semantic_config", "default")

        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=k * 2,  # Over-fetch for re-ranking
            fields="content_vector",
        )

        kwargs: dict[str, Any] = {
            "search_text": query,
            "vector_queries": [vector_query],
            "filter": filter_expr,
            "top": k,
        }

        if use_semantic_reranker:
            kwargs["query_type"] = QueryType.SEMANTIC
            kwargs["semantic_configuration_name"] = sem_config
            kwargs["query_caption"] = QueryCaptionType.EXTRACTIVE
            kwargs["query_answer"] = QueryAnswerType.EXTRACTIVE

        results = self._search_client.search(**kwargs)
        return self._parse_results(results)

    def upsert_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert documents into the search index.

        Documents must include an 'id' field and a 'content_vector' field
        (pre-computed embedding). Use embed_and_upsert for automatic embedding.

        Args:
            documents: List of document dicts.
            batch_size: Number of documents per upload batch.

        Returns:
            Total number of documents indexed.
        """
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = self._search_client.upload_documents(documents=batch)
            succeeded = sum(1 for r in result if r.succeeded)
            total += succeeded
            logger.info(
                "Documents indexed",
                batch=i // batch_size + 1,
                succeeded=succeeded,
                failed=len(batch) - succeeded,
            )

        return total

    def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID from the index."""
        docs = [{"id": doc_id} for doc_id in ids]
        result = self._search_client.delete_documents(documents=docs)
        deleted = sum(1 for r in result if r.succeeded)
        logger.info("Documents deleted", count=deleted)
        return deleted

    def get_index_stats(self) -> IndexStats:
        """Return statistics for the current index."""
        stats = self._index_client.get_index_statistics(self._index_name)
        return IndexStats(
            document_count=stats["documentCount"],
            storage_size_bytes=stats["storageSize"],
            index_name=self._index_name,
        )

    def create_index(
        self,
        index_name: str | None = None,
        vector_dimensions: int | None = None,
    ) -> SearchIndex:
        """
        Create a new search index with vector, keyword, and semantic search support.

        Args:
            index_name: Name for the new index.
            vector_dimensions: Embedding dimensions (default: 3072 for text-embedding-3-large).

        Returns:
            The created SearchIndex object.
        """
        name = index_name or self._index_name
        dims = vector_dimensions or self._vector.get("dimensions", 3072)

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
            SearchableField(name="title", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="date_created", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dims,
                vector_search_profile_name="vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw-config", parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"})],
            profiles=[VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")],
        )

        semantic_search = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name="default",
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="content")],
                        keywords_fields=[SemanticField(field_name="category")],
                    ),
                )
            ]
        )

        index = SearchIndex(
            name=name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        created = self._index_client.create_or_update_index(index)
        logger.info("Search index created", index_name=name, dimensions=dims)
        return created

    def delete_index(self, index_name: str | None = None) -> None:
        """Delete a search index permanently."""
        name = index_name or self._index_name
        self._index_client.delete_index(name)
        logger.info("Search index deleted", index_name=name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_results(self, results: Any) -> list[SearchResult]:
        """Convert raw Azure Search results to SearchResult objects."""
        parsed: list[SearchResult] = []

        for r in results:
            score = r.get("@search.score", 0.0) or 0.0
            reranker_score = r.get("@search.rerankerScore", 0.0) or 0.0

            captions: list[str] = []
            if r.get("@search.captions"):
                captions = [c.text for c in r["@search.captions"] if c.text]

            answers: list[str] = []
            if hasattr(results, "get_answers") and results.get_answers():
                answers = [a.text for a in results.get_answers() if a.text]

            parsed.append(SearchResult(
                content=r.get("content", ""),
                score=score,
                reranker_score=reranker_score,
                id=r.get("id", ""),
                source=r.get("source", ""),
                title=r.get("title", ""),
                captions=captions,
                answers=answers,
                metadata={k: v for k, v in r.items() if not k.startswith("@") and k not in ("id", "content", "source", "title", "content_vector")},
            ))

        # Sort by reranker score if available, else by search score
        parsed.sort(key=lambda x: x.reranker_score if x.reranker_score > 0 else x.score, reverse=True)

        logger.info(
            "Search complete",
            results=len(parsed),
            top_score=f"{parsed[0].reranker_score or parsed[0].score:.3f}" if parsed else "0",
        )

        return parsed
