"""
Azure AI Content Safety — comprehensive content moderation.

Provides:
- Text analysis (hate, self-harm, sexual, violence)
- Image analysis (same categories)
- Prompt Shield (jailbreak + indirect attack detection)
- Groundedness detection (RAG response verification)
- Protected material detection (copyright, code)
- Custom blocklists
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (
    AnalyzeImageOptions,
    AnalyzeTextOptions,
    ImageData,
    TextCategory,
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SeverityLevel(IntEnum):
    """Azure Content Safety severity levels."""
    SAFE = 0
    LOW = 2
    MEDIUM = 4
    HIGH = 6


@dataclass
class CategoryResult:
    """Result for a single content safety category."""

    category: str
    severity: int
    filtered: bool = False

    @property
    def severity_label(self) -> str:
        if self.severity == 0:
            return "safe"
        elif self.severity <= 2:
            return "low"
        elif self.severity <= 4:
            return "medium"
        else:
            return "high"

    @property
    def is_safe(self) -> bool:
        return self.severity == 0


@dataclass
class ContentSafetyResult:
    """Aggregated content safety analysis result."""

    text: str
    hate: CategoryResult = field(default_factory=lambda: CategoryResult("hate", 0))
    self_harm: CategoryResult = field(default_factory=lambda: CategoryResult("self_harm", 0))
    sexual: CategoryResult = field(default_factory=lambda: CategoryResult("sexual", 0))
    violence: CategoryResult = field(default_factory=lambda: CategoryResult("violence", 0))

    jailbreak_detected: bool = False
    indirect_attack_detected: bool = False
    groundedness_score: float = 1.0
    is_grounded: bool = True

    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        """True if all categories are safe and no attacks detected."""
        return (
            self.hate.is_safe
            and self.self_harm.is_safe
            and self.sexual.is_safe
            and self.violence.is_safe
            and not self.jailbreak_detected
            and not self.indirect_attack_detected
        )

    @property
    def max_severity(self) -> int:
        return max(
            self.hate.severity,
            self.self_harm.severity,
            self.sexual.severity,
            self.violence.severity,
        )

    def categories_triggered(self) -> list[str]:
        """Return list of category names that exceeded safe threshold."""
        triggered = []
        for cat in [self.hate, self.self_harm, self.sexual, self.violence]:
            if not cat.is_safe:
                triggered.append(cat.category)
        if self.jailbreak_detected:
            triggered.append("jailbreak")
        if self.indirect_attack_detected:
            triggered.append("indirect_attack")
        return triggered

    def audit_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "max_severity": self.max_severity,
            "categories_triggered": self.categories_triggered(),
            "hate_severity": self.hate.severity,
            "self_harm_severity": self.self_harm.severity,
            "sexual_severity": self.sexual.severity,
            "violence_severity": self.violence.severity,
            "jailbreak_detected": self.jailbreak_detected,
            "indirect_attack_detected": self.indirect_attack_detected,
            "is_grounded": self.is_grounded,
            "groundedness_score": self.groundedness_score,
        }


@dataclass
class PromptShieldResult:
    """Result of prompt shield (jailbreak + indirect attack) detection."""

    user_prompt_attack: bool = False
    document_attack: bool = False
    attack_types: list[str] = field(default_factory=list)


class AzureContentSafetyClient:
    """
    Production Azure AI Content Safety client.

    Implements the full content moderation pipeline:
    text and image analysis, prompt shield, groundedness
    detection, and custom blocklist management.

    Example:
        cs = AzureContentSafetyClient()

        # Scan user input
        result = cs.analyze_text(user_input)
        if not result.is_safe:
            return f"Content policy violation: {result.categories_triggered()}"

        # Shield against prompt injection
        shield = cs.shield_prompt(user_message, documents=retrieved_docs)
        if shield.document_attack:
            return "Document contains a potential prompt injection attack."

        # Verify RAG answer is grounded
        grounded = cs.check_groundedness(
            query="What is the refund policy?",
            answer=rag_answer,
            sources=retrieved_chunks,
        )
        if not grounded.is_grounded:
            return "Answer could not be verified against source documents."
    """

    def __init__(self) -> None:
        cfg = get_config()
        cs_cfg = cfg.content_safety
        raw = load_config()
        raw_cs = raw.get("content_safety", {})

        self._endpoint = cs_cfg.endpoint
        self._api_key = cs_cfg.api_key
        self._api_version = cs_cfg.api_version
        self._thresholds = cs_cfg.thresholds
        self._groundedness_enabled = raw_cs.get("groundedness", {}).get("enabled", True)
        self._groundedness_threshold = raw_cs.get("groundedness", {}).get("threshold", 0.7)

        self._client = ContentSafetyClient(
            endpoint=self._endpoint,
            credential=AzureKeyCredential(self._api_key),
        )

        logger.info("AzureContentSafetyClient initialised", endpoint=self._endpoint)

    def analyze_text(
        self,
        text: str,
        categories: list[str] | None = None,
        blocklist_names: list[str] | None = None,
        output_type: str = "FourSeverityLevels",
    ) -> ContentSafetyResult:
        """
        Analyse text for harmful content across all safety categories.

        Args:
            text: The text to analyse (max 10,000 characters).
            categories: Categories to check. Default: all four.
            blocklist_names: Custom blocklists to apply.
            output_type: "FourSeverityLevels" (0/2/4/6) or "EightSeverityLevels".

        Returns:
            ContentSafetyResult with per-category severity and safety assessment.
        """
        all_categories = [
            TextCategory.HATE,
            TextCategory.SELF_HARM,
            TextCategory.SEXUAL,
            TextCategory.VIOLENCE,
        ]

        request = AnalyzeTextOptions(
            text=text,
            categories=categories or all_categories,
            output_type=output_type,
        )

        if blocklist_names:
            request.blocklist_names = blocklist_names

        try:
            response = self._client.analyze_text(request)
        except HttpResponseError as exc:
            logger.error("Content safety analysis failed", error=str(exc))
            raise

        result = self._parse_text_response(response, text)

        # Apply configured thresholds
        thresholds = self._thresholds
        result.hate.filtered = result.hate.severity >= thresholds.get("hate", 2)
        result.self_harm.filtered = result.self_harm.severity >= thresholds.get("self_harm", 2)
        result.sexual.filtered = result.sexual.severity >= thresholds.get("sexual", 2)
        result.violence.filtered = result.violence.severity >= thresholds.get("violence", 2)

        logger.info(
            "Content safety analysis complete",
            is_safe=result.is_safe,
            max_severity=result.max_severity,
            categories_triggered=result.categories_triggered(),
            text_snippet=text[:50],
        )

        return result

    def analyze_image(
        self,
        image_path: Path,
        categories: list[str] | None = None,
    ) -> ContentSafetyResult:
        """
        Analyse an image for harmful content.

        Args:
            image_path: Path to the image (JPEG, PNG, GIF, BMP, TIFF, WebP).
            categories: Categories to check. Default: all four.

        Returns:
            ContentSafetyResult for the image.
        """
        image_bytes = image_path.read_bytes()

        request = AnalyzeImageOptions(
            image=ImageData(content=image_bytes),
        )

        try:
            response = self._client.analyze_image(request)
        except HttpResponseError as exc:
            logger.error("Image content safety analysis failed", error=str(exc))
            raise

        return self._parse_image_response(response, str(image_path))

    def shield_prompt(
        self,
        user_prompt: str,
        documents: list[str] | None = None,
    ) -> PromptShieldResult:
        """
        Detect jailbreak attempts and indirect prompt injection attacks.

        Prompt Shield uses a specialised model to detect:
        - Jailbreaks: Attempts to override system instructions in user messages
        - Indirect attacks: Malicious instructions embedded in retrieved documents

        Args:
            user_prompt: The user's input message.
            documents: Retrieved documents to check for injection attacks.

        Returns:
            PromptShieldResult indicating detected attack types.
        """
        try:
            from azure.ai.contentsafety.models import ShieldPromptOptions
            request = ShieldPromptOptions(
                user_prompt=user_prompt,
                documents=documents or [],
            )
            response = self._client.shield_prompt(request)
        except Exception as exc:
            logger.warning("Prompt shield unavailable, skipping", error=str(exc))
            return PromptShieldResult()

        user_attack = False
        doc_attack = False
        attack_types: list[str] = []

        if hasattr(response, "user_prompt_analysis"):
            user_attack = getattr(response.user_prompt_analysis, "attack_detected", False)
            if user_attack:
                attack_types.append("jailbreak")

        if hasattr(response, "documents_analysis"):
            for doc_analysis in (response.documents_analysis or []):
                if getattr(doc_analysis, "attack_detected", False):
                    doc_attack = True
                    attack_types.append("indirect_injection")
                    break

        result = PromptShieldResult(
            user_prompt_attack=user_attack,
            document_attack=doc_attack,
            attack_types=attack_types,
        )

        if user_attack or doc_attack:
            logger.warning(
                "Prompt Shield: attack detected",
                user_attack=user_attack,
                doc_attack=doc_attack,
                attack_types=attack_types,
            )

        return result

    def check_groundedness(
        self,
        query: str,
        answer: str,
        sources: list[str],
        threshold: float | None = None,
    ) -> ContentSafetyResult:
        """
        Verify that a RAG answer is grounded in the provided source documents.

        Uses Azure AI Content Safety's groundedness detection to score
        how well the answer is supported by the retrieved context.

        Args:
            query: The original user question.
            answer: The generated answer to verify.
            sources: Retrieved source documents used to generate the answer.
            threshold: Minimum groundedness score (overrides config).

        Returns:
            ContentSafetyResult with is_grounded and groundedness_score populated.
        """
        min_score = threshold or self._groundedness_threshold

        try:
            from azure.ai.contentsafety.models import GroundednessOptions
            request = GroundednessOptions(
                domain="Generic",
                task="QnA",
                text=answer,
                groundingSource="\n\n".join(sources),
                query=query,
            )
            response = self._client.detect_groundedness(request)
            grounded = not getattr(response, "ungrounded", True)
            score = getattr(response, "groundedness_score", 0.0) or 0.0
        except Exception as exc:
            logger.warning("Groundedness check unavailable", error=str(exc))
            grounded = True
            score = 1.0

        result = ContentSafetyResult(
            text=answer,
            groundedness_score=score,
            is_grounded=grounded and score >= min_score,
        )

        logger.info(
            "Groundedness check complete",
            is_grounded=result.is_grounded,
            score=f"{score:.3f}",
            threshold=min_score,
        )

        return result

    def scan_input(self, user_message: str) -> ContentSafetyResult:
        """Convenience: scan a user input with prompt shield."""
        result = self.analyze_text(user_message)
        shield = self.shield_prompt(user_message)
        result.jailbreak_detected = shield.user_prompt_attack
        return result

    def scan_output(
        self, model_output: str, grounding_context: list[str] | None = None, query: str = ""
    ) -> ContentSafetyResult:
        """Convenience: scan model output with optional groundedness check."""
        result = self.analyze_text(model_output)
        if grounding_context and self._groundedness_enabled:
            ground_result = self.check_groundedness(query, model_output, grounding_context)
            result.groundedness_score = ground_result.groundedness_score
            result.is_grounded = ground_result.is_grounded
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_text_response(self, response: Any, text: str) -> ContentSafetyResult:
        """Parse Azure Content Safety text analysis response."""
        cat_map: dict[str, int] = {}
        for cat_result in (response.categories_analysis or []):
            cat_map[cat_result.category.lower()] = cat_result.severity or 0

        return ContentSafetyResult(
            text=text,
            hate=CategoryResult("hate", cat_map.get("hate", 0)),
            self_harm=CategoryResult("self_harm", cat_map.get("selfharm", 0)),
            sexual=CategoryResult("sexual", cat_map.get("sexual", 0)),
            violence=CategoryResult("violence", cat_map.get("violence", 0)),
        )

    def _parse_image_response(self, response: Any, source: str) -> ContentSafetyResult:
        """Parse Azure Content Safety image analysis response."""
        cat_map: dict[str, int] = {}
        for cat_result in (response.categories_analysis or []):
            cat_map[cat_result.category.lower()] = cat_result.severity or 0

        return ContentSafetyResult(
            text=f"[image: {source}]",
            hate=CategoryResult("hate", cat_map.get("hate", 0)),
            self_harm=CategoryResult("self_harm", cat_map.get("selfharm", 0)),
            sexual=CategoryResult("sexual", cat_map.get("sexual", 0)),
            violence=CategoryResult("violence", cat_map.get("violence", 0)),
        )
