"""Assessment-compatible streaming CLI: python chat.py."""
from assessment.config import Settings
from assessment.llm import LLM, ChatSession, format_stats
from assessment.telemetry import Telemetry


def main():
    settings = Settings()
    llm = LLM(settings.connection(), Telemetry(settings.runtime_dir / "telemetry.sqlite"))
    session = ChatSession()
    print("Technical assistant. Enter /quit to exit; /clear to clear the last 10 messages.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt == "/quit":
            break
        if prompt == "/clear":
            session.history.clear()
            continue
        if not prompt:
            continue
        print("Assistant: ", end="", flush=True)
        for event in session.turn(prompt, llm):
            if event["type"] == "delta":
                print(event["text"], end="", flush=True)
            elif event["type"] == "error":
                print(f"\n[error] {event['error_type']}: {event['message']}")
            elif event["type"] == "stats":
                print("\n" + format_stats(event))


if __name__ == "__main__":
    main()
