import os
import re

import numpy as np
import pytest

from assessment.config import Settings


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, variable in [("live", "RUN_LIVE_TESTS"), ("docker", "RUN_DOCKER_TESTS"), ("model", "RUN_MODEL_TESTS")]:
            if marker in item.keywords and os.environ.get(variable) != "1":
                item.add_marker(pytest.mark.skip(reason=f"Set {variable}=1 to run this external integration check"))


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, runtime_dir=tmp_path / "runtime", model_cache_dir=tmp_path / "models",
                    openai_api_key="test-credential", chunk_tokens=64, chunk_overlap=8)


class FakeEmbedder:
    """Deterministic integration fixture, never used for reported retrieval benchmarks."""
    identity = "test-bag-of-words-v1"
    def count(self, text):
        return len(text.split())

    def pieces(self, text, limit, overlap):
        words = text.split()
        return [" ".join(words[i:i+limit]) for i in range(0, len(words), limit-overlap)]

    def embed(self, texts):
        return np.asarray([self.query(text) for text in texts], dtype=np.float32)

    def query(self, text):
        result = np.zeros(32, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            result[sum(word.encode()) % 32] += 1
        norm = np.linalg.norm(result)
        return (result / norm if norm else result).tolist()


@pytest.fixture
def index(settings):
    from assessment.retrieval import KnowledgeIndex

    return KnowledgeIndex(settings, FakeEmbedder())
