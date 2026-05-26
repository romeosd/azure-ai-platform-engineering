"""
Azure AI Evaluation — Prompt Flow and LLM-as-judge evaluation.

Provides:
- RAG evaluation: groundedness, relevance, coherence, fluency, similarity
- Prompt Flow run submission and monitoring
- Batch evaluation over test datasets
- Azure AI Studio evaluation integration
- Application Insights metric publishing
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import AzureOpenAI

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetric:
    """A single evaluation metric."""

    name: str
    score: float
    reasoning: str = ""


@dataclass
class RAGEvalResult:
    """Full RAG evaluation result with all quality metrics."""

    question: str
    answer: str
    context: str
    ground_truth: str = ""

    groundedness: float = 0.0       # Is the answer supported by context?
    relevance: float = 0.0          # Is the answer relevant to the question?
    coherence: float = 0.0          # Is the answer well-structured and logical?
    fluency: float = 0.0            # Is the answer grammatically correct and natural?
    similarity: float = 0.0         # How similar is the answer to ground truth?
    f1_score: float = 0.0           # Token-level F1 vs ground truth

    reasoning: dict[str, str] = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        scores = [s for s in [self.groundedness, self.relevance, self.coherence, self.fluency] if s > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def passed(self, threshold: float = 3.0) -> bool:
        return self.overall_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "fluency": self.fluency,
            "similarity": self.similarity,
            "f1_score": self.f1_score,
            "overall_score": self.overall_score,
            "passed": self.overall_score >= 3.0,
        }


@dataclass
class BatchEvalSummary:
    """Summary statistics for a batch evaluation run."""

    total: int
    passed: int
    avg_groundedness: float = 0.0
    avg_relevance: float = 0.0
    avg_coherence: float = 0.0
    avg_fluency: float = 0.0
    avg_overall: float = 0.0
    results: list[RAGEvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


class PromptFlowEvaluator:
    """
    Azure AI evaluation engine using LLM-as-judge and Prompt Flow metrics.

    Implements the standard Azure AI evaluation metrics:
    groundedness (1-5), relevance (1-5), coherence (1-5), fluency (1-5),
    and similarity (0-1) with optional token-level F1 against ground truth.

    Example:
        evaluator = PromptFlowEvaluator()

        result = evaluator.evaluate_rag_response(
            question="What is the refund window?",
            answer=rag_response,
            context="\n".join(chunk.content for chunk in retrieved_chunks),
            ground_truth="Customers can request a refund within 30 days.",
        )

        print(f"Groundedness: {result.groundedness}/5")
        print(f"Overall: {result.overall_score:.1f}/5")
    """

    # Scoring prompts aligned with Azure AI Studio metric definitions
    _GROUNDEDNESS_PROMPT = """You are an expert evaluator assessing whether an AI-generated answer is grounded in the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

Score the GROUNDEDNESS of the answer on a scale of 1 to 5:
1 = The answer contradicts or fabricates information not in the context
2 = The answer has mostly unsupported claims
3 = The answer is partially supported by the context
4 = The answer is mostly supported with minor unsupported details
5 = Every claim in the answer is directly supported by the context

Respond with valid JSON only:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}"""

    _RELEVANCE_PROMPT = """You are an expert evaluator assessing whether an AI-generated answer is relevant to the question asked.

QUESTION: {question}
ANSWER: {answer}

Score the RELEVANCE on a scale of 1 to 5:
1 = The answer does not address the question at all
2 = The answer is tangentially related but misses the core question
3 = The answer partially addresses the question
4 = The answer mostly addresses the question with minor gaps
5 = The answer directly and completely addresses the question

Respond with valid JSON only:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}"""

    _COHERENCE_PROMPT = """You are an expert evaluator assessing the coherence and logical structure of an AI-generated answer.

QUESTION: {question}
ANSWER: {answer}

Score the COHERENCE on a scale of 1 to 5:
1 = The answer is incoherent, contradictory, or impossible to follow
2 = The answer has significant logical gaps or structure issues
3 = The answer is somewhat coherent with some structural issues
4 = The answer is mostly coherent with minor issues
5 = The answer is perfectly logical, well-structured, and easy to follow

Respond with valid JSON only:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}"""

    _FLUENCY_PROMPT = """You are an expert evaluator assessing the fluency and language quality of an AI-generated answer.

ANSWER: {answer}

Score the FLUENCY on a scale of 1 to 5:
1 = The answer is barely readable with severe grammar or spelling issues
2 = The answer has significant language quality issues
3 = The answer is readable but with noticeable language issues
4 = The answer is well-written with minor imperfections
5 = The answer is fluent, natural, and professionally written

Respond with valid JSON only:
{{"score": <int 1-5>, "reasoning": "<brief explanation>"}}"""

    def __init__(self) -> None:
        cfg = get_config()

        self._client = AzureOpenAI(
            azure_endpoint=cfg.azure_openai.endpoint,
            api_key=cfg.azure_openai.api_key,
            api_version=cfg.azure_openai.api_version,
        )
        self._eval_deployment = cfg.get_deployment("gpt4o")

        logger.info("PromptFlowEvaluator initialised", eval_model=self._eval_deployment)

    def evaluate_rag_response(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str = "",
    ) -> RAGEvalResult:
        """
        Evaluate a RAG response across all quality metrics.

        Args:
            question: The user's original question.
            answer: The RAG-generated answer to evaluate.
            context: The retrieved context used to generate the answer.
            ground_truth: Optional reference answer for similarity/F1 scoring.

        Returns:
            RAGEvalResult with per-metric scores and reasoning.
        """
        result = RAGEvalResult(
            question=question,
            answer=answer,
            context=context,
            ground_truth=ground_truth,
        )

        # Evaluate each metric
        result.groundedness, result.reasoning["groundedness"] = self._score_metric(
            self._GROUNDEDNESS_PROMPT.format(context=context, question=question, answer=answer)
        )

        result.relevance, result.reasoning["relevance"] = self._score_metric(
            self._RELEVANCE_PROMPT.format(question=question, answer=answer)
        )

        result.coherence, result.reasoning["coherence"] = self._score_metric(
            self._COHERENCE_PROMPT.format(question=question, answer=answer)
        )

        result.fluency, result.reasoning["fluency"] = self._score_metric(
            self._FLUENCY_PROMPT.format(answer=answer)
        )

        if ground_truth:
            result.similarity = self._compute_similarity(answer, ground_truth)
            result.f1_score = self._compute_f1(answer, ground_truth)

        logger.info(
            "RAG evaluation complete",
            groundedness=f"{result.groundedness:.1f}",
            relevance=f"{result.relevance:.1f}",
            coherence=f"{result.coherence:.1f}",
            fluency=f"{result.fluency:.1f}",
            overall=f"{result.overall_score:.1f}",
        )

        return result

    def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
        publish_to_appinsights: bool = False,
    ) -> BatchEvalSummary:
        """
        Evaluate a batch of RAG responses.

        Args:
            test_cases: List of dicts with keys: question, answer, context, ground_truth (optional).
            publish_to_appinsights: Publish aggregate metrics to Application Insights.

        Returns:
            BatchEvalSummary with aggregate statistics and all individual results.
        """
        results: list[RAGEvalResult] = []

        for i, case in enumerate(test_cases):
            logger.info("Evaluating test case", index=i, total=len(test_cases))
            result = self.evaluate_rag_response(**case)
            results.append(result)

        n = len(results)
        passed = sum(1 for r in results if r.passed)

        summary = BatchEvalSummary(
            total=n,
            passed=passed,
            avg_groundedness=sum(r.groundedness for r in results) / n if n else 0,
            avg_relevance=sum(r.relevance for r in results) / n if n else 0,
            avg_coherence=sum(r.coherence for r in results) / n if n else 0,
            avg_fluency=sum(r.fluency for r in results) / n if n else 0,
            avg_overall=sum(r.overall_score for r in results) / n if n else 0,
            results=results,
        )

        logger.info(
            "Batch evaluation complete",
            total=n,
            passed=passed,
            pass_rate=f"{summary.pass_rate:.1%}",
            avg_overall=f"{summary.avg_overall:.2f}",
        )

        if publish_to_appinsights:
            self._publish_to_appinsights(summary)

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_metric(self, prompt: str) -> tuple[float, str]:
        """Call the LLM judge and parse the score."""
        try:
            response = self._client.chat.completions.create(
                model=self._eval_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content or "{}")
            score = float(raw.get("score", 0))
            reasoning = raw.get("reasoning", "")
            return score, reasoning
        except Exception as exc:
            logger.error("Metric scoring failed", error=str(exc))
            return 0.0, f"Evaluation error: {exc}"

    def _compute_similarity(self, answer: str, ground_truth: str) -> float:
        """Compute semantic similarity using embedding cosine distance."""
        from openai import AzureOpenAI
        cfg = get_config()
        embed_deployment = cfg.get_deployment("text_embed_3_small")

        try:
            resp = self._client.embeddings.create(
                model=embed_deployment,
                input=[answer, ground_truth],
            )
            import numpy as np
            a = np.array(resp.data[0].embedding)
            b = np.array(resp.data[1].embedding)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            return max(0.0, min(1.0, cosine))
        except Exception:
            return 0.0

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score between prediction and ground truth."""
        pred_tokens = set(prediction.lower().split())
        gt_tokens = set(ground_truth.lower().split())

        if not pred_tokens or not gt_tokens:
            return 0.0

        intersection = pred_tokens & gt_tokens
        if not intersection:
            return 0.0

        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _publish_to_appinsights(self, summary: BatchEvalSummary) -> None:
        """Publish evaluation metrics to Azure Application Insights."""
        raw = load_config()
        conn_str = raw.get("observability", {}).get("application_insights", {}).get("connection_string", "")

        if not conn_str:
            logger.warning("No App Insights connection string configured, skipping metric publish")
            return

        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from opentelemetry import metrics

            configure_azure_monitor(connection_string=conn_str)
            meter = metrics.get_meter("PromptFlowEvaluator")

            for name, value in [
                ("rag.groundedness", summary.avg_groundedness),
                ("rag.relevance", summary.avg_relevance),
                ("rag.coherence", summary.avg_coherence),
                ("rag.fluency", summary.avg_fluency),
                ("rag.overall", summary.avg_overall),
                ("rag.pass_rate", summary.pass_rate),
            ]:
                gauge = meter.create_gauge(name)
                gauge.set(value)

            logger.info("Evaluation metrics published to Application Insights")
        except Exception as exc:
            logger.warning("Failed to publish metrics to App Insights", error=str(exc))
