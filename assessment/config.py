"""Connection-scoped credentials and reproducible, non-secret run settings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_URL = "https://api.openai.com/v1"


class Pricing(BaseModel):
    input_per_million: float = Field(default=0.20, ge=0)
    cached_input_per_million: float = Field(default=0.02, ge=0)
    output_per_million: float = Field(default=1.25, ge=0)
    source: str = "https://developers.openai.com/api/docs/models/gpt-5.4-nano"
    estimated: bool = True


class Connection(BaseModel):
    profile: str = "official_test"
    base_url: str = OFFICIAL_URL
    model: str = "gpt-5.4-nano"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""), exclude=True)
    timeout_s: float = Field(default=60, gt=0, le=300)
    max_output_tokens: int = Field(default=2048, ge=32, le=16384)
    pricing: Pricing | None = None

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.rstrip("/")
        parts = urlsplit(value)
        if parts.username or parts.password or parts.query or parts.fragment or not parts.hostname:
            raise ValueError("Base URL must contain only scheme, host and API path")
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        ):
            raise ValueError("Use HTTPS, except for a local development endpoint")
        return value

    @model_validator(mode="after")
    def official_prices(self) -> "Connection":
        if self.pricing is None and self.base_url == OFFICIAL_URL and self.model == "gpt-5.4-nano":
            self.pricing = Pricing()
        return self

    def public_snapshot(self) -> dict:
        return self.model_dump(mode="json")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)
    openai_base_url: str = OFFICIAL_URL
    openai_api_key: SecretStr = SecretStr("")
    openai_api_key_file: Path | None = None
    model_name: str = "gpt-5.4-nano"
    profile: str = "official_test"
    runtime_dir: Path = ROOT / "data" / "runtime"
    model_cache_dir: Path = ROOT / "data" / "model_cache"
    chroma_host: str | None = None
    chroma_port: int = 8000
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_backend: Literal["local", "openai"] = "local"
    embedding_dimensions: int = Field(default=384, ge=64, le=1536)
    embedding_api_key: SecretStr = SecretStr("")
    embedding_threads: int = 2
    chunk_tokens: int = Field(default=320, ge=64, le=420)
    chunk_overlap: int = Field(default=40, ge=0, le=100)
    retrieval_mode: Literal["dense", "lexical", "hybrid", "rerank"] = "hybrid"
    sandbox_image: str = "assessment-sandbox:local"
    sandbox_timeout_s: int = Field(default=20, ge=1, le=120)
    sandbox_runner_url: str | None = None
    app_access_token: SecretStr = SecretStr("")
    tools_base_url: str = "http://127.0.0.1:8001"

    def connection(self) -> Connection:
        key = self.openai_api_key
        path = self.openai_api_key_file
        # The user's existing key file belongs exclusively to the official endpoint.
        if not key.get_secret_value() and path is None and self.openai_base_url.rstrip("/") == OFFICIAL_URL:
            path = ROOT / "OPENAI API KEY.txt"
        if not key.get_secret_value() and path and path.is_file():
            key = SecretStr(path.read_text(encoding="utf-8-sig").strip())
        return Connection(profile=self.profile, base_url=self.openai_base_url, model=self.model_name, api_key=key)

    def index_config(self) -> dict:
        return {"embedding_model": self.embedding_model, "embedding_backend": self.embedding_backend,
                "embedding_dimensions": self.embedding_dimensions, "chunk_tokens": self.chunk_tokens,
                "chunk_overlap": self.chunk_overlap, "parser_version": "1", "chunker_version": "1"}

    def index_id(self) -> str:
        return hashlib.sha256(json.dumps(self.index_config(), sort_keys=True).encode()).hexdigest()[:16]
