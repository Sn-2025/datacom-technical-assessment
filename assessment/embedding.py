"""CPU ONNX embeddings, a matching untruncated tokenizer and versioned identity."""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import Settings
from .documents import Chunk, Document, digest


class Embedder:
    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        self.model_name = settings.embedding_model
        pinned = settings.model_cache_dir / ("models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/"
                                             "52398278842ec682c6f32300af41344b1c0b0bb2")
        options = {"specific_model_path": str(pinned)} if (self.model_name == "BAAI/bge-small-en-v1.5"
                    and (pinned / "model_optimized.onnx").exists()) else {}
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(settings.model_cache_dir),
                                   threads=settings.embedding_threads, **options)
        self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.lock = threading.RLock()
        weights = Path(self.model.model._model_dir) / self.model.model.model_description.model_file
        self.identity = digest(self.tokenizer.to_str() + self.model_name + digest(weights.read_bytes()))

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def pieces(self, text: str, limit: int, overlap: int) -> list[str]:
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded.ids) <= limit:
            return [text]
        result = []
        for start in range(0, len(encoded.ids), limit-overlap):
            stop = min(start+limit, len(encoded.ids))
            result.append(text[encoded.offsets[start][0]:encoded.offsets[stop-1][1]])
            if stop == len(encoded.ids):
                break
        return result

    def embed(self, texts: list[str]) -> np.ndarray:
        if any(self.count(text) > 500 for text in texts):
            raise ValueError("Embedding input would exceed the validated token limit")
        with self.lock:
            return np.asarray(list(self.model.passage_embed(texts, batch_size=32)), dtype=np.float32)

    def query(self, text: str) -> list[float]:
        if self.count(text) > 450:
            raise ValueError("Query is too long; use at most 450 embedding tokens")
        with self.lock:
            return next(iter(self.model.query_embed(text))).tolist()


class OpenAIEmbedder(Embedder):
    """Official API embeddings with an explicit, bounded in-memory query cache."""
    def __init__(self, settings: Settings):
        from openai import OpenAI
        from tokenizers import Tokenizer

        from .config import OFFICIAL_URL
        from .telemetry import Telemetry

        tokenizer_path = settings.model_cache_dir / (
            "models--qdrant--bge-small-en-v1.5-onnx-q/snapshots/"
            "52398278842ec682c6f32300af41344b1c0b0bb2/tokenizer.json")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.model_name = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        key = settings.embedding_api_key.get_secret_value()
        if not key and settings.openai_base_url.rstrip("/") == OFFICIAL_URL:
            key = settings.connection().api_key.get_secret_value()
        if not key:
            raise ValueError("Official embeddings require EMBEDDING_API_KEY or an official backend credential")
        self.client = OpenAI(api_key=key, base_url=OFFICIAL_URL, max_retries=0, timeout=60)
        self.telemetry = Telemetry(settings.runtime_dir / "telemetry.sqlite")
        self.identity = digest(self.model_name + str(self.dimensions) + self.tokenizer.to_str())
        self.lock = threading.RLock()
        self.query_cache = OrderedDict()
        self.last_query_cache_hit = False

    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import APIConnectionError, InternalServerError, RateLimitError

        for attempt in range(7):
            try:
                return self._embed_once(texts)
            except (RateLimitError, APIConnectionError, InternalServerError):
                if attempt == 6:
                    raise
                time.sleep(min(2 ** attempt, 10))

    def _embed_once(self, texts: list[str]) -> np.ndarray:
        started = time.perf_counter()
        response = None
        try:
            response = self.client.embeddings.create(model=self.model_name, input=texts, dimensions=self.dimensions)
            return np.asarray([item.embedding for item in sorted(response.data, key=lambda item: item.index)],
                              dtype=np.float32)
        finally:
            usage = response.usage.total_tokens if response else None
            self.telemetry.record(uuid.uuid4().hex, "embedding_request", model=self.model_name,
                dimensions=self.dimensions, input_tokens=usage, cost_usd=usage*0.02/1_000_000
                if usage is not None and self.model_name == "text-embedding-3-small" else None,
                cost_estimated=True, latency_ms=(time.perf_counter()-started)*1000,
                status="success" if response else "error")

    def query(self, text: str) -> list[float]:
        with self.lock:
            self.last_query_cache_hit = text in self.query_cache
            if text not in self.query_cache:
                self.query_cache[text] = self.embed([text])[0].tolist()
                if len(self.query_cache) > 256:
                    self.query_cache.popitem(last=False)
            self.query_cache.move_to_end(text)
            return self.query_cache[text]


def chunk_document(document: Document, embedder, settings: Settings) -> list[Chunk]:
    result, parts, locators = [], [], []
    config_id = settings.index_id()

    def emit():
        if not parts:
            return
        text = "\n\n".join(parts)
        ordinal = len(result)
        result.append(Chunk(id=digest(f"{document.source_id}:{document.content_hash}:{config_id}:{ordinal}:{text}"),
            source_id=document.source_id, source_uri=document.source_uri, version=document.version,
            title=document.title, text=text, locators=list(locators), ordinal=ordinal, license=document.license))

    for element in document.elements:
        for piece in embedder.pieces(element.text, settings.chunk_tokens, settings.chunk_overlap):
            candidate = "\n\n".join([*parts, piece])
            if parts and embedder.count(candidate) > settings.chunk_tokens:
                emit()
                parts, locators = [], []
            parts.append(piece)
            if element.locator not in locators:
                locators.append(element.locator)
    emit()
    return result
