from datetime import date, timedelta

from assessment.planner import Day, DraftItinerary, TripRequest, Visit, plan_trip, validate_itinerary
from assessment.telemetry import Telemetry
from assessment.travel_tools import attraction_data, weather_data


def fixtures():
    request = TripRequest(start_date=date(2026, 9, 6), mode="mock")
    data = {"get_weather": weather_data(request.start_date, 2, "mock"), "get_attractions": attraction_data("mock")}
    draft = DraftItinerary(days=[
        Day(date="2026-09-06", visits=[Visit(attraction_id="gallery", start="09:30", end="11:00")]),
        Day(date="2026-09-07", visits=[Visit(attraction_id="waterfront", start="10:00", end="11:30")])],
        decision_summary="Use an indoor visit on the rainy first day.")
    return request, data, draft


def test_budget_includes_all_mandatory_allowances():
    request, data, draft = fixtures()
    errors, costs, total = validate_itinerary(draft, request, data)
    assert not errors
    assert total == 27600
    assert {c["category"] for c in costs} >= {"food_per_adult_day", "transport_per_adult_day", "accommodation_per_adult_night"}
    request.budget_cents = 20000
    errors, _, _ = validate_itinerary(draft, request, data)
    assert any("exceeds budget" in error for error in errors)


def test_rejects_wrong_dates_rain_and_overlapping_visits():
    request, data, draft = fixtures()
    draft.days[0].visits = [Visit(attraction_id="waterfront", start="10:00", end="11:00"),
                           Visit(attraction_id="domain", start="10:30", end="11:30")]
    draft.days[1].date = (request.start_date+timedelta(days=3)).isoformat()
    errors, _, _ = validate_itinerary(draft, request, data)
    assert any("rain" in e for e in errors)
    assert any("minutes between" in e for e in errors)
    assert any("each requested date" in e for e in errors)


def test_agent_calls_both_tools_and_repairs_constraint_failure(tmp_path):
    request, data, good = fixtures()
    bad = good.model_copy(deep=True)
    bad.days[0].visits[0].attraction_id = "waterfront"
    class Model:
        completes = 0
        drafts = 0
        def complete(self, messages, **kwargs):
            self.completes += 1
            if self.completes == 1:
                return {"role": "assistant", "tool_calls": [{"id": name, "type": "function",
                    "function": {"name": name, "arguments": "{}"}} for name in data]}
            return {"role": "assistant", "content": "Ready to create the plan"}
        def structured(self, messages, output_type, **kwargs):
            self.drafts += 1
            return bad if self.drafts == 1 else good
    class Tools:
        calls = []
        def execute(self, name, args, request):
            self.calls.append(name)
            return data[name]
    model, tools = Model(), Tools()
    events = list(plan_trip(request, model, tools, Telemetry(tmp_path / "events.sqlite")))
    assert set(tools.calls) == {"get_weather", "get_attractions"}
    assert model.drafts == 2
    assert events[-1]["status"] == "success"
    assert any(e["kind"] == "constraint_check" and e["errors"] for e in events)


def test_missing_date_is_explicit(tmp_path):
    events = list(plan_trip(TripRequest(), None, None, Telemetry(tmp_path / "events.sqlite")))
    assert events[-1]["status"] == "needs_information"
