import json
from pathlib import Path

import pytest

from assessment.code_assistant import QUICKSORT_TESTS, CodeArtifact, DockerRunner, repair_code
from assessment.telemetry import Telemetry


class Model:
    def __init__(self):
        self.calls = 0
        self.messages = []
    def structured(self, messages, output_type, **kwargs):
        self.calls += 1
        self.messages = list(messages)
        return CodeArtifact(code="def quicksort(values):\n    return sorted(values)\n", summary="Return a sorted copy.")


def test_retries_include_first_attempt_and_feed_errors_back(tmp_path):
    class Runner:
        calls = 0
        def run(self, code, tests):
            self.calls += 1
            return {"success": False, "exit_code": 1, "tests": 4, "output": "AssertionError: duplicates missing"}
    model, runner = Model(), Runner()
    events = list(repair_code("Write quicksort", model, runner, Telemetry(tmp_path / "events.sqlite"), tmp_path / "runs"))
    assert model.calls == runner.calls == 3
    assert events[-1]["status"] == "failed"
    assert "duplicates missing" in json.dumps(model.messages)
    assert any(m["role"] == "assistant" and "def quicksort" in m["content"] for m in model.messages)
    assert (Path(events[-1]["artifact_dir"]) / "test_acceptance.py").read_text() == QUICKSORT_TESTS


def test_success_stops_immediately(tmp_path):
    class Runner:
        def run(self, code, tests):
            return {"success": True, "exit_code": 0, "tests": 4, "output": "4 passed"}
    model = Model()
    events = list(repair_code("Write quicksort", model, Runner(), Telemetry(tmp_path / "events.sqlite"), tmp_path / "runs"))
    assert model.calls == 1
    assert events[-1]["status"] == "success"


def test_zero_tests_is_never_success(tmp_path):
    class Runner:
        def run(self, code, tests):
            return {"success": True, "exit_code": 0, "tests": 0, "output": "no tests"}
    events = list(repair_code("Write quicksort", Model(), Runner(), Telemetry(tmp_path / "events.sqlite"), tmp_path / "runs"))
    assert events[-1]["status"] == "failed"


@pytest.mark.docker
def test_real_container_execution_passes_and_times_out():
    runner = DockerRunner("assessment-sandbox:local", timeout_s=3)
    good = runner.run("def quicksort(values):\n    return sorted(values)\n", QUICKSORT_TESTS)
    assert good["success"] and good["tests"] == 4
    bad = runner.run("def quicksort(values):\n    while True:\n        pass\n", QUICKSORT_TESTS)
    assert not bad["success"]


@pytest.mark.docker
def test_container_cannot_edit_acceptance_tests():
    code = "from pathlib import Path\nPath('/workspace/test_acceptance.py').write_text('')\ndef quicksort(x): return sorted(x)\n"
    result = DockerRunner("assessment-sandbox:local").run(code, QUICKSORT_TESTS)
    assert not result["success"]
    assert result["test_files_unchanged"]
