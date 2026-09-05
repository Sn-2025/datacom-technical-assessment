"""Trusted container supervisor. Generated code never sees the host or API credentials."""
import ast
import hashlib
import json
import os
import resource
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    data = json.loads(sys.stdin.read(250000))
    code, tests = data["code"], data["tests"]
    tree = ast.parse(tests)
    expected = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
                   for n in ast.walk(tree))
    if not expected or not any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
        raise ValueError("Acceptance tests must contain test functions and assertions")
    work = Path("/workspace")
    target = work / "test_acceptance.py"
    target.write_text(tests, encoding="utf-8")
    (work / "solution.py").write_text(code, encoding="utf-8")
    for path in (target, work / "solution.py"):
        path.chmod(0o444)
    test_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    timeout = min(int(data.get("timeout_s", 20)), 120)

    def restrict():
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout+1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2*1024*1024, 2*1024*1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    try:
        result = subprocess.run([sys.executable, "-I", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            "--junitxml=/tmp/results.xml", str(target)], cwd=work, preexec_fn=restrict,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}, capture_output=True, text=True, timeout=timeout)
        report = Path("/tmp/results.xml")
        suites = list(ET.parse(report).getroot().iter("testsuite")) if report.exists() else []
        count = sum(int(s.attrib.get("tests", 0)) for s in suites)
        failures = sum(int(s.attrib.get("failures", 0))+int(s.attrib.get("errors", 0))+int(s.attrib.get("skipped", 0)) for s in suites)
        unchanged = hashlib.sha256(target.read_bytes()).hexdigest() == test_hash
        print(json.dumps({"success": result.returncode == 0 and count >= expected and failures == 0 and unchanged,
            "exit_code": result.returncode, "tests": count, "test_files_unchanged": unchanged,
            "output": (result.stdout+result.stderr)[-12000:], "timeout": False}))
    except subprocess.TimeoutExpired:
        print(json.dumps({"success": False, "exit_code": -1, "tests": 0, "output": "Execution timed out", "timeout": True}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"success": False, "exit_code": -1, "tests": 0, "output": type(exc).__name__, "timeout": False}))
