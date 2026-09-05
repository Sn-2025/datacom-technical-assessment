"""Exercise the actual provider and both external travel APIs."""
import json
from datetime import date, timedelta
from pathlib import Path

from assessment.llm import LLM
from assessment.planner import TravelTools, TripRequest, plan_trip
from assessment.runtime import Runtime

if __name__ == "__main__":
    runtime = Runtime()
    request = TripRequest(start_date=date.today() + timedelta(days=1), mode="live")
    events = []
    for event in plan_trip(request, LLM(runtime.settings.connection(), runtime.telemetry),
                           TravelTools(runtime.settings.tools_base_url), runtime.telemetry):
        events.append(event)
        print(event["kind"], event.get("status", event.get("stage", "")), flush=True)
    Path("artifacts/verification/planner-live.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    assert events[-1]["status"] == "success", events[-1]["status"]
