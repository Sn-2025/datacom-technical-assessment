"""Shared non-secret services; provider clients remain connection/session scoped."""
from __future__ import annotations

import threading

from .config import Settings
from .telemetry import Telemetry


class Runtime:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.telemetry = Telemetry(self.settings.runtime_dir / "telemetry.sqlite")
        self._index = None
        self._lock = threading.RLock()

    @property
    def index(self):
        with self._lock:
            if self._index is None:
                from .embedding import Embedder, OpenAIEmbedder
                from .retrieval import KnowledgeIndex

                embedder = OpenAIEmbedder if self.settings.embedding_backend == "openai" else Embedder
                self._index = KnowledgeIndex(self.settings, embedder(self.settings))
            return self._index
