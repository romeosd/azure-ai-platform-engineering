"""
Azure OpenAI Service client.

Provides:
- Chat completions (GPT-4o, GPT-4 Turbo, o1)
- Streaming chat completions
- Embeddings (text-embedding-3-large, ada-002)
- Image generation (DALL-E 3)
- Audio transcription (Whisper)
- Multi-modal vision (image + text)
- Token counting and cost estimation
- Automatic retry with exponential backoff
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

from openai import AzureOpenAI, APIError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Cost per 1K tokens (USD) — Azure australiaeast pricing
_TOKEN_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o":               {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini":          {"input": 0.000150, "output": 0.000600},
    "gpt-4-turbo":          {"input": 0.010, "output": 0.030},
    "o1-preview":           {"input": 0.015, "output": 0.060},
    "o1-mini":              {"input": 0.003, "output": 0.012},
    "text-embedding-3-large": {"input": 0.000130, "output": 0.0},
    "text-embedding-3-small": {"input": 0.000020, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.000100, "output": 0.0},
}


@dataclass
class ChatResult:
    """Structured result from an Azure OpenAI chat completion."""

    content: str
    model: str
    deployment: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class EmbeddingResult:
    """Structured result from an Azure OpenAI embedding call."""

    embedding: list[float]
    model: str
    input_tokens: int = 0

    @property
    def dimensions(self) -> int:
        return len(self.embedding)


class AzureOpenAIClient:
    """
    Production-grade Azure OpenAI client.

    Wraps the openai SDK with typed results, cost tracking,
    automatic retry, and streaming support.

    Example:
        client = AzureOpenAIClient(deployment_key="gpt4o")

        # Chat completion
        result = client.chat("Explain the CAP theorem in three sentences.")
        print(result.content)
        print(f"Cost: ${result.estimated_cost_usd:.4f}")

        # Streaming
        for chunk in client.chat_stream("Write a haiku about Azure"):
            print(chunk, end="", flush=True)

        # Embeddings
        emb = client.embed("Azure AI Search supports hybrid retrieval")
        print(f"Dimensions: {emb.dimensions}")

        # Vision
        result = client.chat_with_image(
            "Describe what you see in this architecture diagram.",
            image_path=Path("diagram.png"),
        )
    """

    def __init__(
        self,
        deployment_key: str = "gpt4o",
        api_version: str | None = None,
    ) -> None:
        cfg = get_config()
        raw = load_config()

        self.deployment = cfg.get_deployment(deployment_key)
        self._endpoint = cfg.azure_openai.endpoint
        self._api_key = cfg.azure_openai.api_key
        self._api_version = api_version or cfg.azure_openai.api_version
        self._inference = cfg.azure_openai.inference

        self._client = AzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )

        logger.info(
            "AzureOpenAIClient initialised",
            deployment=self.deployment,
            api_version=self._api_version,
        )

    @retry(
        retry=retry_if_exception_type((APIError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    def chat(
        self,
        user_message: str,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] = "auto",
        response_format: dict[str, str] | None = None,
    ) -> ChatResult:
        """
        Send a chat completion request to Azure OpenAI.

        Args:
            user_message: The user turn message.
            system: Optional system prompt.
            messages: Full message history (overrides user_message if provided).
            max_tokens: Override config max_tokens.
            temperature: Override config temperature.
            tools: List of tool definitions for function calling.
            tool_choice: "auto", "none", or specific tool dict.
            response_format: e.g. {"type": "json_object"} for JSON mode.

        Returns:
            ChatResult with content, token usage, and cost estimate.
        """
        msgs = messages or []
        if not msgs:
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user_message})

        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": msgs,
            "max_tokens": max_tokens or self._inference.get("max_tokens", 4096),
            "temperature": temperature if temperature is not None else self._inference.get("temperature", 0.1),
            "top_p": self._inference.get("top_p", 0.95),
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        if response_format:
            kwargs["response_format"] = response_format

        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (APIError, RateLimitError) as exc:
            logger.error("Azure OpenAI chat failed", error=str(exc), deployment=self.deployment)
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in choice.message.tool_calls
            ]

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        costs = _TOKEN_COSTS.get(self.deployment, {"input": 0.0, "output": 0.0})
        cost = (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])

        result = ChatResult(
            content=content,
            model=response.model,
            deployment=self.deployment,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=choice.finish_reason or "",
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            tool_calls=tool_calls,
            raw_response=response,
        )

        logger.info(
            "Azure OpenAI chat complete",
            deployment=self.deployment,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=f"{latency_ms:.0f}",
            cost_usd=f"{cost:.6f}",
        )

        return result

    def chat_stream(
        self,
        user_message: str,
        system: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion tokens from Azure OpenAI.

        Yields text chunks as they are received.
        """
        msgs = messages or []
        if not msgs:
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": user_message})

        stream = self._client.chat.completions.create(
            model=self.deployment,
            messages=msgs,
            max_tokens=max_tokens or self._inference.get("max_tokens", 4096),
            temperature=temperature if temperature is not None else self._inference.get("temperature", 0.1),
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def chat_with_image(
        self,
        prompt: str,
        image_path: Path,
        system: str | None = None,
        detail: str = "high",
    ) -> ChatResult:
        """
        Send a multi-modal (vision) request with an image and text.

        Args:
            prompt: The text prompt about the image.
            image_path: Path to the image file (JPEG, PNG, GIF, WebP).
            system: Optional system prompt.
            detail: "low" | "high" | "auto" — image resolution for vision.

        Returns:
            ChatResult with the model's description/analysis.
        """
        img_bytes = image_path.read_bytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        suffix = image_path.suffix.lower().lstrip(".")
        media_type_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
        media_type = media_type_map.get(suffix, "image/jpeg")

        content = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_b64}", "detail": detail}},
            {"type": "text", "text": prompt},
        ]

        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})

        return self.chat(user_message="", messages=msgs)

    def embed(
        self,
        text: str,
        deployment_key: str = "text_embed_3_large",
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        """
        Generate a vector embedding for text.

        Args:
            text: The text to embed.
            deployment_key: Config key for the embedding deployment.
            dimensions: Optional reduced dimensions (text-embedding-3 models only).

        Returns:
            EmbeddingResult with the embedding vector.
        """
        cfg = get_config()
        embed_deployment = cfg.get_deployment(deployment_key)

        kwargs: dict[str, Any] = {
            "input": text,
            "model": embed_deployment,
        }
        if dimensions and "3" in embed_deployment:
            kwargs["dimensions"] = dimensions

        try:
            response = self._client.embeddings.create(**kwargs)
        except APIError as exc:
            logger.error("Embedding failed", error=str(exc))
            raise

        embedding = response.data[0].embedding
        input_tokens = response.usage.prompt_tokens if response.usage else 0

        return EmbeddingResult(
            embedding=embedding,
            model=embed_deployment,
            input_tokens=input_tokens,
        )

    def embed_batch(
        self,
        texts: list[str],
        deployment_key: str = "text_embed_3_large",
    ) -> list[EmbeddingResult]:
        """Embed a list of texts, respecting API batch limits."""
        results = []
        for i, text in enumerate(texts):
            logger.debug("Embedding batch item", index=i, total=len(texts))
            results.append(self.embed(text, deployment_key=deployment_key))
        return results

    def transcribe(self, audio_path: Path, language: str = "en") -> str:
        """
        Transcribe audio to text using Azure OpenAI Whisper.

        Args:
            audio_path: Path to the audio file (mp3, mp4, wav, webm, etc.).
            language: ISO 639-1 language code.

        Returns:
            Transcribed text string.
        """
        cfg = get_config()
        whisper_deployment = cfg.get_deployment("whisper")

        with open(audio_path, "rb") as f:
            response = self._client.audio.transcriptions.create(
                model=whisper_deployment,
                file=f,
                language=language,
            )

        logger.info("Audio transcription complete", path=str(audio_path), language=language)
        return response.text

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "hd",
        style: str = "natural",
        n: int = 1,
    ) -> list[str]:
        """
        Generate images using DALL-E 3.

        Args:
            prompt: Description of the image to generate.
            size: "1024x1024" | "1792x1024" | "1024x1792"
            quality: "standard" | "hd"
            style: "natural" | "vivid"
            n: Number of images (DALL-E 3 supports n=1 only).

        Returns:
            List of image URLs.
        """
        cfg = get_config()
        dalle_deployment = cfg.get_deployment("dall_e_3")

        response = self._client.images.generate(
            model=dalle_deployment,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=n,
        )

        urls = [img.url for img in response.data if img.url]
        logger.info("DALL-E 3 image generation complete", count=len(urls))
        return urls

    def json_mode(
        self,
        prompt: str,
        system: str | None = None,
        schema_hint: str = "",
    ) -> dict[str, Any]:
        """
        Request a JSON-structured response from the model.

        Args:
            prompt: The user prompt (should describe the expected JSON structure).
            system: System prompt (automatically includes JSON instruction).
            schema_hint: Optional schema description to include in system prompt.

        Returns:
            Parsed JSON dict.
        """
        import json

        sys_prompt = system or "You are a helpful assistant."
        if schema_hint:
            sys_prompt += f"\n\nRespond with valid JSON matching this schema:\n{schema_hint}"
        else:
            sys_prompt += "\n\nAlways respond with valid JSON."

        result = self.chat(
            user_message=prompt,
            system=sys_prompt,
            response_format={"type": "json_object"},
        )

        try:
            return json.loads(result.content)
        except json.JSONDecodeError as exc:
            logger.error("JSON mode response parse failed", error=str(exc))
            raise ValueError(f"Model returned invalid JSON: {result.content[:200]}") from exc
