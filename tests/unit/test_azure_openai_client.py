"""
Unit tests for AzureOpenAIClient.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.models.azure_openai_client import AzureOpenAIClient, ChatResult, EmbeddingResult


class TestAzureOpenAIClient:

    def _mock_config(self) -> MagicMock:
        cfg = MagicMock()
        cfg.azure_openai.endpoint = "https://test.openai.azure.com/"
        cfg.azure_openai.api_key = "test-key"
        cfg.azure_openai.api_version = "2024-10-21"
        cfg.azure_openai.inference = {"max_tokens": 4096, "temperature": 0.1, "top_p": 0.95}
        cfg.get_deployment.side_effect = lambda key: {
            "gpt4o": "gpt-4o",
            "text_embed_3_large": "text-embedding-3-large",
            "whisper": "whisper",
            "dall_e_3": "dall-e-3",
        }.get(key, "gpt-4o")
        return cfg

    @patch("src.models.azure_openai_client.get_config")
    @patch("src.models.azure_openai_client.load_config")
    @patch("src.models.azure_openai_client.AzureOpenAI")
    def test_chat_returns_chat_result(
        self, mock_aoai_cls: MagicMock, mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        mock_choice = MagicMock()
        mock_choice.message.content = "The answer is 42."
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 8

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_aoai_cls.return_value = mock_client

        client = AzureOpenAIClient()
        result = client.chat("What is the meaning of life?")

        assert isinstance(result, ChatResult)
        assert result.content == "The answer is 42."
        assert result.input_tokens == 20
        assert result.output_tokens == 8
        assert result.total_tokens == 28
        assert result.finish_reason == "stop"
        assert result.estimated_cost_usd > 0

    @patch("src.models.azure_openai_client.get_config")
    @patch("src.models.azure_openai_client.load_config")
    @patch("src.models.azure_openai_client.AzureOpenAI")
    def test_chat_with_system_prompt_includes_system(
        self, mock_aoai_cls: MagicMock, mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        mock_choice = MagicMock()
        mock_choice.message.content = "Response"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_aoai_cls.return_value = mock_client

        client = AzureOpenAIClient()
        client.chat("Hello", system="You are a helpful assistant.")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    @patch("src.models.azure_openai_client.get_config")
    @patch("src.models.azure_openai_client.load_config")
    @patch("src.models.azure_openai_client.AzureOpenAI")
    def test_embed_returns_embedding_result(
        self, mock_aoai_cls: MagicMock, mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        fake_embedding = [0.01] * 3072
        mock_emb_data = MagicMock()
        mock_emb_data.embedding = fake_embedding
        mock_emb_usage = MagicMock()
        mock_emb_usage.prompt_tokens = 6
        mock_emb_response = MagicMock()
        mock_emb_response.data = [mock_emb_data]
        mock_emb_response.usage = mock_emb_usage

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_emb_response
        mock_aoai_cls.return_value = mock_client

        client = AzureOpenAIClient()
        result = client.embed("Azure AI Search is great for RAG")

        assert isinstance(result, EmbeddingResult)
        assert result.dimensions == 3072
        assert result.input_tokens == 6

    @patch("src.models.azure_openai_client.get_config")
    @patch("src.models.azure_openai_client.load_config")
    @patch("src.models.azure_openai_client.AzureOpenAI")
    def test_chat_stream_yields_chunks(
        self, mock_aoai_cls: MagicMock, mock_load: MagicMock, mock_get: MagicMock
    ) -> None:
        mock_get.return_value = self._mock_config()
        mock_load.return_value = {}

        chunks = []
        for text in ["Hello", " Azure", "!"]:
            c = MagicMock()
            c.choices = [MagicMock()]
            c.choices[0].delta.content = text
            chunks.append(c)

        empty = MagicMock()
        empty.choices = [MagicMock()]
        empty.choices[0].delta.content = None
        chunks.append(empty)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)
        mock_aoai_cls.return_value = mock_client

        client = AzureOpenAIClient()
        result = "".join(client.chat_stream("Say hello to Azure"))
        assert result == "Hello Azure!"

    def test_chat_result_total_tokens(self) -> None:
        result = ChatResult(
            content="test", model="gpt-4o", deployment="gpt-4o",
            input_tokens=100, output_tokens=50
        )
        assert result.total_tokens == 150

    def test_embedding_result_dimensions(self) -> None:
        result = EmbeddingResult(embedding=[0.1] * 3072, model="text-embedding-3-large")
        assert result.dimensions == 3072
