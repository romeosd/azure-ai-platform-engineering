"""
Configuration loader for Azure AI Platform Engineering.
Loads and validates azure_config.yaml with environment variable substitution.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _substitute_env_vars(value: str) -> str:
    pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default)
    return pattern.sub(replacer, value)


def _process_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _process_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_process_dict(item) for item in data]
    elif isinstance(data, str):
        return _substitute_env_vars(data)
    return data


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load and return the Azure platform configuration with env var substitution."""
    if config_path is None:
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "config" / "azure_config.yaml")

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Copy config/azure_config.yaml and populate your Azure resource endpoints."
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    return _process_dict(raw)


class AzureOpenAIConfig(BaseModel):
    endpoint: str = Field(default="")
    api_key: str = Field(default="")
    api_version: str = Field(default="2024-10-21")
    deployments: dict[str, str] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)


class AISearchConfig(BaseModel):
    endpoint: str = Field(default="")
    api_key: str = Field(default="")
    api_version: str = Field(default="2024-07-01")
    indexes: dict[str, str] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    vector: dict[str, Any] = Field(default_factory=dict)


class ContentSafetyConfig(BaseModel):
    endpoint: str = Field(default="")
    api_key: str = Field(default="")
    api_version: str = Field(default="2024-09-01")
    thresholds: dict[str, int] = Field(default_factory=dict)


class AzurePlatformConfig(BaseModel):
    azure_openai: AzureOpenAIConfig = Field(default_factory=AzureOpenAIConfig)
    ai_search: AISearchConfig = Field(default_factory=AISearchConfig)
    content_safety: ContentSafetyConfig = Field(default_factory=ContentSafetyConfig)

    @classmethod
    def from_yaml(cls, config_path: str | None = None) -> "AzurePlatformConfig":
        raw = load_config(config_path)
        return cls(
            azure_openai=AzureOpenAIConfig(**raw.get("azure_openai", {})),
            ai_search=AISearchConfig(**raw.get("ai_search", {})),
            content_safety=ContentSafetyConfig(**raw.get("content_safety", {})),
        )

    def get_deployment(self, key: str) -> str:
        deployment = self.azure_openai.deployments.get(key)
        if not deployment:
            available = list(self.azure_openai.deployments.keys())
            raise KeyError(f"Deployment '{key}' not found. Available: {available}")
        return deployment


_config: AzurePlatformConfig | None = None

def get_config() -> AzurePlatformConfig:
    global _config
    if _config is None:
        _config = AzurePlatformConfig.from_yaml()
    return _config
