"""CPU ONNX embeddings, a matching untruncated tokenizer and versioned identity."""
from __future__ import annotations

import threading

import numpy as np

from .config import Settings
from .documents import Chunk, Document, digest


class Embedder:
    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        self.model_name = settings.embedding_model
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(settings.model_cache_dir),
                                   threads=settings.embedding_threads)
        self.tokenizer = Tokenizer.from_str(self.model.model.tokenizer.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.lock = threading.RLock()
        self.identity = digest(self.tokenizer.to_str() + self.model_name)

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
