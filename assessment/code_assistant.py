"""Generate, execute fixed tests in Docker, and repair at most twice."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from .telemetry import redact


class CodeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    summary: str


class TestContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tests: str
    contract: str


QUICKSORT_TESTS = '''from solution import quicksort

def test_empty_and_single():
    assert quicksort([]) == []
    assert quicksort([7]) == [7]

def test_duplicates_and_negatives():
    assert quicksort([3, -1, 3, 0, -5, 2]) == [-5, -1, 0, 2, 3, 3]

def test_orderings():
    assert quicksort(list(range(60))) == list(range(60))
    assert quicksort(list(range(59, -1, -1))) == list(range(60))

def test_preserves_input():
    data = [4, 2, 1]
    assert quicksort(data) == [1, 2, 4]
    assert data == [4, 2, 1]
'''


class SandboxUnavailable(RuntimeError):
    pass


class RemoteRunner:
    def __init__(self, url: str, timeout_s: int):
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def run(self, code: str, tests: str) -> dict:
        try:
            with httpx.Client(timeout=self.timeout_s+40, follow_redirects=False) as client:
                response = client.post(self.url+"/run", json={"code": code, "tests": tests})
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise SandboxUnavailable("The isolated runner service is unavailable") from exc


def make_runner(settings):
    if settings.sandbox_runner_url:
        return RemoteRunner(settings.sandbox_runner_url, settings.sandbox_timeout_s)
    return DockerRunner(settings.sandbox_image, settings.sandbox_timeout_s)


class DockerRunner:
    def __init__(self, image: str, timeout_s: int = 20):
        self.image, self.timeout_s = image, timeout_s

    def run(self, code: str, tests: str) -> dict:
        docker = shutil.which("docker")
        if not docker:
            raise SandboxUnavailable("Docker is required for generated-code execution")
        name = "assessment-job-"+uuid.uuid4().hex
        command = [docker, "run", "--rm", "--name", name, "--interactive", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--security-opt", "no-new-privileges", "--pids-limit", "64", "--memory", "256m", "--cpus", "1",
            "--tmpfs", "/workspace:rw,noexec,nosuid,size=32m,mode=755",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m,mode=1777", self.image]
        try:
            result = subprocess.run(command, input=json.dumps({"code": code, "tests": tests,
                "timeout_s": self.timeout_s}), text=True, capture_output=True, timeout=self.timeout_s+20, check=False)
        except subprocess.TimeoutExpired:
            subprocess.run([docker, "rm", "--force", name], capture_output=True, timeout=10, check=False)
            return {"success": False, "exit_code": -1, "tests": 0, "output": "Execution timed out", "timeout": True}
        if result.returncode == 125:
            raise SandboxUnavailable("Docker engine or sandbox image is unavailable")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {"success": False, "exit_code": result.returncode, "tests": 0,
                    "output": redact((result.stdout+result.stderr)[-12000:]), "timeout": False}


def repair_code(task: str, llm, runner, telemetry, artifact_root: Path, tests: str | None = None):
    if not task.strip() or len(task) > 10000:
        raise ValueError("Provide a coding task of 1-10000 characters")
    run_id = uuid.uuid4().hex
    root = artifact_root / run_id
    root.mkdir(parents=True, exist_ok=False)
    test_source = "user_provided" if tests else "built_in"
    if tests is None:
        if "quicksort" in task.lower() and "rust" not in task.lower():
            tests = QUICKSORT_TESTS
        else:
            test_source = "model_generated_frozen_contract"
            contract = llm.structured([{"role": "system", "content": (
                "Define a Python acceptance contract for the task. Tests must import from solution and use pytest "
                "assertions. Cover edge cases, not implementation details. Only Python standard library and pytest "
                "are installed. Do not write implementation code. Require a single solution.py module.")},
                {"role": "user", "content": task}], TestContract, run_id=run_id)
            tests = contract.tests
            (root / "contract.txt").write_text(contract.contract, encoding="utf-8")
    if not tests.strip() or len(tests) > 100000:
        raise ValueError("Provide non-empty acceptance tests of at most 100000 characters")
    (root / "test_acceptance.py").write_bytes(tests.encode("utf-8"))
    test_hash = hashlib.sha256(tests.encode()).hexdigest()
    messages = [{"role": "system", "content": (
        "Implement the user's task as a single Python solution.py. The acceptance tests are fixed and cannot be "
        "edited. Use only the Python standard library. Return code and a short change summary. Do not access files, "
        "network, processes, test internals, or environment variables. Focus on a correct general solution.")},
        {"role": "user", "content": json.dumps({"task": task, "fixed_acceptance_tests": tests})}]
    for attempt in range(1, 4):
        yield telemetry.record(run_id, "code_progress", attempt=attempt, stage="generating", test_source=test_source)
        try:
            artifact = llm.structured(messages, CodeArtifact, run_id=run_id)
            messages.append({"role": "assistant", "content": artifact.model_dump_json()})
            attempt_dir = root / f"attempt-{attempt}"
            attempt_dir.mkdir()
            (attempt_dir / "solution.py").write_text(artifact.code, encoding="utf-8")
            yield telemetry.record(run_id, "code_progress", attempt=attempt, stage="running_tests")
            result = runner.run(artifact.code, tests)
        except SandboxUnavailable:
            yield telemetry.record(run_id, "code_result", status="sandbox_unavailable", attempt=attempt,
                artifact_dir=str(root), message="Start Docker and build the sandbox image; generated code was not run on the host.")
            return
        except Exception as exc:
            result = {"success": False, "tests": 0, "exit_code": -1, "output": type(exc).__name__}
        result["output"] = redact(result.get("output", ""))[-12000:]
        (root / f"attempt-{attempt}-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if hashlib.sha256((root / "test_acceptance.py").read_bytes()).hexdigest() != test_hash:
            raise RuntimeError("Acceptance tests changed unexpectedly")
        yield telemetry.record(run_id, "test_result", attempt=attempt, **result)
        if result.get("success") and result.get("tests", 0) > 0 and result.get("exit_code") == 0:
            yield telemetry.record(run_id, "code_result", status="success", attempt=attempt,
                artifact_dir=str(root), test_source=test_source, test_hash=test_hash)
            return
        messages.append({"role": "user", "content": "Tests failed. Repair the implementation only.\n"+json.dumps(result)})
    yield telemetry.record(run_id, "code_result", status="failed", attempt=3,
        artifact_dir=str(root), test_source=test_source, test_hash=test_hash)
