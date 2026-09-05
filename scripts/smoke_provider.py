"""Small, real provider probes; deliberately separate from offline pytest."""
import json
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from assessment.config import Settings
from assessment.llm import LLM
from assessment.telemetry import Telemetry


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    number: int


def main():
    settings = Settings()
    connection = settings.connection()
    connection.max_output_tokens = 256
    telemetry = Telemetry(settings.runtime_dir / "telemetry.sqlite")
    llm = LLM(connection, telemetry)
    report = {"config": connection.public_snapshot(), "checks": {}}
    events = list(llm.stream([{"role": "user", "content": "Reply with exactly: Hello."}]))
    stats = events[-1]
    report["checks"]["streaming"] = {"passed": any(e["type"] == "delta" for e in events) and stats["status"] == "success",
        "usage_present": stats["prompt_tokens"] is not None, "telemetry": stats}
    try:
        tool = {"type": "function", "function": {"name": "verify_connection", "description": "Verify the test connection.",
            "strict": True, "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
        reply = llm.complete([{"role": "user", "content": "Call verify_connection now. Do not answer in text."}],
                             run_id=uuid.uuid4().hex, tools=[tool])
        report["checks"]["tool_calling"] = {"passed": any(c["function"]["name"] == "verify_connection" for c in reply.get("tool_calls", []))}
    except Exception as exc:
        report["checks"]["tool_calling"] = {"passed": False, "error_type": type(exc).__name__}
    try:
        result = llm.structured([{"role": "user", "content": "Return ok=true and number=7."}], Probe, run_id=uuid.uuid4().hex)
        report["checks"]["structured_output"] = {"passed": result.ok and result.number == 7}
    except Exception as exc:
        report["checks"]["structured_output"] = {"passed": False, "error_type": type(exc).__name__}
    output = Path("artifacts/verification")
    output.mkdir(parents=True, exist_ok=True)
    (output / "provider.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
