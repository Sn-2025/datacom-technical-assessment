from types import SimpleNamespace as NS

import pytest

from assessment.config import Connection, Settings, Pricing
from assessment.llm import LLM, ChatSession
from assessment.telemetry import Telemetry, cost_usd, redact


def test_connection_secrets_and_endpoint_boundaries():
    connection = Connection(api_key="sk-private-example")
    assert "sk-private-example" not in str(connection.public_snapshot())
    assert "api_key" not in connection.public_snapshot()
    assert Connection(model="another-model").pricing is None
    for url in ["http://example.com/v1", "https://user:password@example.com", "https://example.com?key=secret"]:
        with pytest.raises(ValueError):
            Connection(base_url=url)


def test_cost_cached_tokens_and_unknown_usage():
    prices = Pricing(input_per_million=2, cached_input_per_million=0.5, output_per_million=6)
    assert cost_usd(100, 20, 40, prices) == pytest.approx(0.00026)
    assert cost_usd(None, 20, 0, prices) is None
    assert cost_usd(100, 20, 0, None) is None


def test_stream_handles_empty_usage_chunk_and_metrics(tmp_path):
    def create(**kwargs):
        assert kwargs["stream_options"] == {"include_usage": True}
        yield NS(usage=None, model="gpt-5.4-nano", id="request-1", choices=[NS(delta=NS(content="Hello"))])
        yield NS(usage=NS(prompt_tokens=9, completion_tokens=2, prompt_tokens_details=None),
                 model="gpt-5.4-nano", id="request-1", choices=[])
    telemetry = Telemetry(tmp_path / "events.sqlite")
    llm = LLM(Connection(api_key="sk-private-example"), telemetry,
              client=NS(chat=NS(completions=NS(create=create))))
    events = list(llm.stream([{"role": "user", "content": "Hello"}]))
    assert events[0]["text"] == "Hello"
    assert events[-1]["prompt_tokens"] == 9
    assert events[-1]["completion_tokens"] == 2
    assert events[-1]["status"] == "success"
    assert "sk-private-example" not in str(telemetry.recent())


def test_interrupted_stream_keeps_usage_unknown(tmp_path):
    def create(**kwargs):
        yield NS(usage=None, model="gpt-5.4-nano", id="id", choices=[NS(delta=NS(content="Partial"))])
        raise TimeoutError("sk-do-not-log")
    telemetry = Telemetry(tmp_path / "events.sqlite")
    llm = LLM(Connection(api_key="test"), telemetry, client=NS(chat=NS(completions=NS(create=create))))
    events = list(llm.stream([]))
    assert events[-1]["status"] == "error"
    assert events[-1]["cost_usd"] is None
    assert events[-1]["prompt_tokens"] is None


def test_history_is_ten_messages_not_ten_turns():
    class Replies:
        def stream(self, messages):
            assert len(messages) <= 11  # Fixed system instruction plus at most ten conversation messages.
            yield {"type": "delta", "text": "answer"}
            yield {"type": "stats", "status": "success"}
    session = ChatSession()
    for i in range(9):
        list(session.turn(str(i), Replies()))
    assert len(session.history) == 10
    assert session.history[0]["content"] == "4"
    assert session.history[-1]["role"] == "assistant"


def test_redaction():
    assert "sk-secret" not in redact("failed key sk-secret and Bearer sensitive")
    assert "sensitive" not in redact("Bearer sensitive")


def test_nested_bearer_redaction_preserves_valid_log_json(tmp_path):
    telemetry = Telemetry(tmp_path / "events.sqlite")
    telemetry.record("run", "test", nested={"authorization": "Bearer sensitive"})
    assert telemetry.recent()[0]["nested"]["authorization"] == "Bearer [REDACTED]"


def test_dotenv_overrides_process_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\nMODEL_NAME=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    monkeypatch.setenv("MODEL_NAME", "environment-model")

    settings = Settings(_env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == "from-dotenv"
    assert settings.model_name == "dotenv-model"
