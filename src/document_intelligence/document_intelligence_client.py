"""
Azure AI Document Intelligence (formerly Form Recognizer).

Provides:
- Layout analysis (text, tables, selection marks, structure)
- Prebuilt models: invoice, receipt, ID document, contract
- Custom model analysis
- Batch document processing
- Structured field extraction with confidence scores
- Table extraction with cell-level metadata
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    DocumentAnalysisFeature,
    DocumentContentFormat,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedTable:
    """A table extracted from a document."""

    page: int
    rows: int
    columns: int
    cells: list[dict[str, Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the table as Markdown."""
        grid: dict[tuple[int, int], str] = {}
        for cell in self.cells:
            r, c = cell.get("row_index", 0), cell.get("column_index", 0)
            grid[(r, c)] = cell.get("content", "")

        rows_md = []
        for r in range(self.rows):
            row = [grid.get((r, c), "") for c in range(self.columns)]
            rows_md.append("| " + " | ".join(row) + " |")
            if r == 0:
                rows_md.append("| " + " | ".join(["---"] * self.columns) + " |")

        return "\n".join(rows_md)


@dataclass
class ExtractedField:
    """A key-value field extracted from a prebuilt model."""

    name: str
    value: Any
    confidence: float = 0.0
    value_type: str = ""

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.8


@dataclass
class DocumentResult:
    """Aggregated result from Document Intelligence analysis."""

    model_used: str
    page_count: int = 0
    raw_text: str = ""
    tables: list[ExtractedTable] = field(default_factory=list)
    fields: list[ExtractedField] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    selection_marks: list[dict[str, Any]] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    raw_response: Any = None

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def field_count(self) -> int:
        return len(self.fields)

    def get_field(self, name: str, default: Any = None) -> Any:
        """Retrieve a field value by name."""
        for f in self.fields:
            if f.name.lower() == name.lower():
                return f.value
        return default

    def confident_fields(self, threshold: float = 0.8) -> list[ExtractedField]:
        """Return only fields with confidence above threshold."""
        return [f for f in self.fields if f.confidence >= threshold]

    def tables_as_markdown(self) -> str:
        parts = []
        for i, table in enumerate(self.tables):
            parts.append(f"### Table {i+1} (Page {table.page})\n{table.to_markdown()}")
        return "\n\n".join(parts)


class AzureDocumentIntelligenceClient:
    """
    Production Azure AI Document Intelligence client.

    Handles layout analysis, prebuilt model extraction (invoices,
    receipts, IDs, contracts), and custom model analysis.

    Example:
        doc_client = AzureDocumentIntelligenceClient()

        # Analyse document layout
        result = doc_client.analyze_layout(Path("contract.pdf"))
        print(result.raw_text)
        print(result.tables_as_markdown())

        # Extract invoice fields
        invoice = doc_client.analyze_invoice(Path("invoice.pdf"))
        print(f"Vendor: {invoice.get_field('VendorName')}")
        print(f"Total: {invoice.get_field('InvoiceTotal')}")
        print(f"Due: {invoice.get_field('DueDate')}")

        # Custom model
        result = doc_client.analyze_custom(
            Path("insurance_form.pdf"),
            model_id="my-custom-model-id",
        )
    """

    def __init__(self) -> None:
        raw = load_config()
        di_cfg = raw.get("document_intelligence", {})

        self._endpoint = di_cfg.get("endpoint", "")
        self._api_key = di_cfg.get("api_key", "")
        self._api_version = di_cfg.get("api_version", "2024-07-31-preview")
        self._models = di_cfg.get("models", {})

        self._client = DocumentIntelligenceClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

        logger.info("AzureDocumentIntelligenceClient initialised", endpoint=self._endpoint)

    def analyze_layout(
        self,
        file_path: Path,
        output_format: str = "markdown",
        features: list[str] | None = None,
    ) -> DocumentResult:
        """
        Analyse document layout: text, tables, paragraphs, selection marks.

        Args:
            file_path: Path to the document (PDF, JPEG, PNG, TIFF, BMP, HEIF).
            output_format: "markdown" (rich structure) or "text" (plain).
            features: Additional features: "ocrHighResolution", "languages", "barcodes".

        Returns:
            DocumentResult with full layout extraction.
        """
        return self._analyze(
            file_path=file_path,
            model_id=self._models.get("prebuilt_layout", "prebuilt-layout"),
            output_format=output_format,
            features=features,
        )

    def analyze_invoice(self, file_path: Path) -> DocumentResult:
        """
        Extract structured fields from an invoice.

        Extracts: VendorName, VendorAddress, CustomerName, InvoiceId,
        InvoiceDate, DueDate, InvoiceTotal, SubTotal, TotalTax, Items.
        """
        return self._analyze(
            file_path=file_path,
            model_id=self._models.get("prebuilt_invoice", "prebuilt-invoice"),
        )

    def analyze_receipt(self, file_path: Path) -> DocumentResult:
        """
        Extract structured fields from a receipt.

        Extracts: MerchantName, TransactionDate, TransactionTime,
        Total, Subtotal, Tax, Items.
        """
        return self._analyze(
            file_path=file_path,
            model_id=self._models.get("prebuilt_receipt", "prebuilt-receipt"),
        )

    def analyze_id_document(self, file_path: Path) -> DocumentResult:
        """
        Extract identity information from passports and driving licences.

        Extracts: FirstName, LastName, DocumentNumber, DateOfBirth,
        DateOfExpiration, Country.
        """
        return self._analyze(
            file_path=file_path,
            model_id=self._models.get("prebuilt_id", "prebuilt-idDocument"),
        )

    def analyze_contract(self, file_path: Path) -> DocumentResult:
        """
        Extract structured fields from legal contracts.

        Extracts: Parties, Dates, PaymentTerms, Jurisdiction, Clauses.
        """
        return self._analyze(
            file_path=file_path,
            model_id=self._models.get("prebuilt_contract", "prebuilt-contract"),
        )

    def analyze_custom(
        self,
        file_path: Path,
        model_id: str | None = None,
    ) -> DocumentResult:
        """
        Analyse a document using a custom trained model.

        Args:
            file_path: Path to the document.
            model_id: Custom model ID (overrides config custom_model_id).
        """
        mid = model_id or self._models.get("custom_model_id", "")
        if not mid:
            raise ValueError("No custom model ID provided in arguments or config.")

        return self._analyze(file_path=file_path, model_id=mid)

    def analyze_batch(
        self,
        file_paths: list[Path],
        model_id: str = "prebuilt-layout",
    ) -> list[DocumentResult]:
        """Process a batch of documents sequentially."""
        results = []
        for i, path in enumerate(file_paths):
            logger.info("Processing batch document", index=i, total=len(file_paths), path=str(path))
            try:
                result = self._analyze(path, model_id)
                results.append(result)
            except Exception as exc:
                logger.error("Batch document failed", path=str(path), error=str(exc))
                results.append(DocumentResult(model_used=model_id))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _analyze(
        self,
        file_path: Path,
        model_id: str,
        output_format: str | None = None,
        features: list[str] | None = None,
    ) -> DocumentResult:
        """Core analysis method — calls Document Intelligence and parses result."""
        file_bytes = file_path.read_bytes()

        kwargs: dict[str, Any] = {
            "model_id": model_id,
            "analyze_request": AnalyzeDocumentRequest(bytes_source=file_bytes),
        }

        if output_format == "markdown":
            kwargs["output_content_format"] = DocumentContentFormat.MARKDOWN

        if features:
            kwargs["features"] = [DocumentAnalysisFeature(f) for f in features]

        try:
            poller = self._client.begin_analyze_document(**kwargs)
            response = poller.result()
        except HttpResponseError as exc:
            logger.error("Document Intelligence analysis failed", error=str(exc), model=model_id)
            raise

        return self._parse_response(response, model_id)

    def _parse_response(self, response: Any, model_id: str) -> DocumentResult:
        """Convert raw Document Intelligence response to DocumentResult."""
        raw_text = response.content or ""

        # Page count
        page_count = len(response.pages) if response.pages else 0

        # Tables
        tables: list[ExtractedTable] = []
        for table in (response.tables or []):
            cells = [
                {
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "content": cell.content,
                    "is_header": getattr(cell, "kind", "") == "columnHeader",
                }
                for cell in (table.cells or [])
            ]
            tables.append(ExtractedTable(
                page=table.bounding_regions[0].page_number if table.bounding_regions else 0,
                rows=table.row_count,
                columns=table.column_count,
                cells=cells,
            ))

        # Structured fields (prebuilt models)
        fields: list[ExtractedField] = []
        if response.documents:
            for doc in response.documents:
                for name, field_val in (doc.fields or {}).items():
                    if field_val is None:
                        continue
                    fields.append(ExtractedField(
                        name=name,
                        value=getattr(field_val, "value", field_val.content),
                        confidence=field_val.confidence or 0.0,
                        value_type=field_val.type or "",
                    ))

        # Paragraphs
        paragraphs = [p.content for p in (response.paragraphs or []) if p.content]

        # Selection marks
        marks = []
        for page in (response.pages or []):
            for mark in (page.selection_marks or []):
                marks.append({
                    "state": mark.state,
                    "confidence": mark.confidence,
                    "page": page.page_number,
                })

        # Languages
        languages = [lang.locale for lang in (response.languages or []) if lang.locale]

        result = DocumentResult(
            model_used=model_id,
            page_count=page_count,
            raw_text=raw_text,
            tables=tables,
            fields=fields,
            paragraphs=paragraphs,
            selection_marks=marks,
            languages=languages,
            raw_response=response,
        )

        logger.info(
            "Document Intelligence analysis complete",
            model=model_id,
            pages=page_count,
            tables=len(tables),
            fields=len(fields),
        )

        return result
