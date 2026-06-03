"""
Unit tests for AzureAISearchClient.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.rag.ai_search_client import AzureAISearchClient, SearchResult


class TestAzureAISearchClient:

    def _mock_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.ai_search.endpoint = "https://test.search.windows.net"
        cfg.ai_search.api_key = "test-key"
        cfg.ai_search.api_version = "2024-07-01"
        cfg.ai_search.indexes = {"documents": "test-index"}
        cfg.ai_search.retrieval = {"top_k": 10, "semantic_config": "default", "query_type": "semantic"}
        cfg.ai_search.vector = {"profile_name": "vector-profile", "dimensions": 3072, "similarity": "cosine"}
        return cfg

    @patch("src.rag.ai_search_client.get_config")
    @patch("src.rag.ai_search_client.load_config")
    @patch("src.rag.ai_search_client.SearchClient")
    @patch("src.rag.ai_search_client.SearchIndexClient")
    def test_hybrid_search_returns_sorted_results(
        self, mock_idx_cls: MagicMock, mock_sc_cls: MagicMock,
        mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        mock_results = [
            {
                "id": "doc1",
                "content": "High relevance content",
                "title": "Policy",
                "source": "policy.pdf",
                "@search.score": 0.85,
                "@search.rerankerScore": 0.95,
                "@search.captions": [],
            },
            {
                "id": "doc2",
                "content": "Medium relevance content",
                "title": "Guide",
                "source": "guide.pdf",
                "@search.score": 0.7,
                "@search.rerankerScore": 0.72,
                "@search.captions": [],
            },
        ]

        mock_search_client = MagicMock()
        mock_search_client.search.return_value = iter(mock_results)
        mock_sc_cls.return_value = mock_search_client
        mock_idx_cls.return_value = MagicMock()

        client = AzureAISearchClient()
        results = client.hybrid_search(
            query="data retention policy",
            embedding=[0.1] * 3072,
            top_k=5,
        )

        assert len(results) == 2
        assert results[0].reranker_score == 0.95
        assert results[1].reranker_score == 0.72
        assert results[0].content == "High relevance content"

    @patch("src.rag.ai_search_client.get_config")
    @patch("src.rag.ai_search_client.load_config")
    @patch("src.rag.ai_search_client.SearchClient")
    @patch("src.rag.ai_search_client.SearchIndexClient")
    def test_upsert_documents_returns_count(
        self, mock_idx_cls: MagicMock, mock_sc_cls: MagicMock,
        mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        mock_upload_result = [MagicMock(succeeded=True) for _ in range(3)]

        mock_search_client = MagicMock()
        mock_search_client.upload_documents.return_value = mock_upload_result
        mock_sc_cls.return_value = mock_search_client
        mock_idx_cls.return_value = MagicMock()

        client = AzureAISearchClient()
        docs = [
            {"id": f"doc{i}", "content": f"Content {i}", "content_vector": [0.1] * 3072}
            for i in range(3)
        ]
        count = client.upsert_documents(docs)
        assert count == 3

    def test_search_result_sorting(self) -> None:
        """Results should sort by reranker_score when available."""
        results = [
            SearchResult(content="A", score=0.9, reranker_score=0.0),
            SearchResult(content="B", score=0.5, reranker_score=0.8),
            SearchResult(content="C", score=0.7, reranker_score=0.95),
        ]
        results.sort(
            key=lambda x: x.reranker_score if x.reranker_score > 0 else x.score,
            reverse=True,
        )
        assert results[0].content == "C"
        assert results[1].content == "B"
        assert results[2].content == "A"
