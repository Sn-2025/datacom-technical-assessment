"""Streamlit workbench. Provider calls execute in the server-side session."""
from __future__ import annotations

import json
import statistics
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import SecretStr

from assessment.code_assistant import QUICKSORT_TESTS, make_runner, repair_code
from assessment.config import ROOT, Connection
from assessment.llm import LLM, ChatSession, format_stats
from assessment.loaders import load_document
from assessment.planner import TravelTools, TripRequest, plan_trip
from assessment.qa import answer_question
from assessment.runtime import Runtime

st.set_page_config(page_title="AI Engineering Workbench", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2.5rem;max-width:1280px}
[data-testid="stMetric"]{background:#112329;border:1px solid #24434a;padding:1rem;border-radius:12px}
[data-testid="stSidebar"]{border-right:1px solid #24434a}
.eyebrow{color:#66d9c2;font-size:.75rem;letter-spacing:.16em;text-transform:uppercase}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_runtime():
    return Runtime()


runtime = get_runtime()
state = st.session_state
state.setdefault("session_keys", {})
state.setdefault("base_url", runtime.settings.openai_base_url)
state.setdefault("model_name", runtime.settings.model_name)
state.setdefault("chat", ChatSession())


def get_connection():
    default = runtime.settings.connection()
    base = state["base_url"].rstrip("/")
    key = state["session_keys"].get(base)
    if key is None and base == default.base_url:
        key = default.api_key.get_secret_value()
    return Connection(base_url=base, model=state["model_name"], api_key=SecretStr(key or ""),
                      profile="session" if key and base in state["session_keys"] else default.profile)


with st.sidebar:
    st.markdown("### ◈ Workbench")
    st.caption("AI engineering · Technical assessment")
    page = st.radio("Workspace", ["Overview", "Chat", "Knowledge base", "Trip planner", "Code repair", "Connections"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("ACTIVE MODEL")
    st.code(state["model_name"], language=None)
    st.caption("English technical documents · NZD travel budgets")


def title(label, heading, caption):
    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
    st.title(heading)
    st.caption(caption)


if page == "Connections":
    title("Configuration", "Model connections", "Choose a provider and model. Credentials stay in your server-side session.")
    if state.pop("reset_key_input", False):
        state["key_input"] = ""
    with st.form("connection_form"):
        base = st.text_input("API base URL", value=state["base_url"])
        model = st.text_input("Model ID", value=state["model_name"])
        key = st.text_input("API key · optional session override", type="password", key="key_input",
                            help="Leave blank to keep this connection's existing credential. It is never copied to a different endpoint.")
        submitted = st.form_submit_button("Apply connection", type="primary")
    if submitted:
        try:
            validated = Connection(base_url=base, model=model, api_key=SecretStr(key))
            state["base_url"], state["model_name"] = validated.base_url, validated.model
            if key:
                state["session_keys"][validated.base_url] = key
            state["reset_key_input"] = True
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    conn = get_connection()
    st.info("Credential configured" if conn.api_key.get_secret_value() else "No credential configured for this endpoint")
    left, right = st.columns(2)
    if left.button("Test connection and list models"):
        try:
            llm = LLM(conn, runtime.telemetry)
            models = sorted(model.id for model in llm.client.models.list().data)
            st.success("Authenticated successfully. Model listing does not itself verify generation capabilities.")
            st.dataframe({"Available model IDs": models}, use_container_width=True)
        except Exception as exc:
            st.error(f"Connection check failed: {type(exc).__name__}")
    if right.button("Clear session credential"):
        state["session_keys"].pop(conn.base_url, None)
        state["reset_key_input"] = True
        st.rerun()
    st.caption("CLI and deployment credentials can also come from environment variables or a mounted secret file. "
               "A cleared session override falls back to that connection's server configuration.")

elif page == "Overview":
    title("Observability", "Every run, inspectable.", "Actual request telemetry, retrieval evaluations and workflow outcomes.")
    events = runtime.telemetry.recent(2000)
    requests = [e for e in events if e["kind"] == "llm_request"]
    known_cost = [e["cost_usd"] for e in requests if e.get("cost_usd") is not None]
    latencies = [e["latency_ms"] for e in requests]
    cols = st.columns(4)
    cols[0].metric("Model requests", len(requests))
    cols[1].metric("Generation cost", f"${sum(known_cost):.5f}" if known_cost else "—")
    cols[2].metric("Median latency", f"{statistics.median(latencies):.0f} ms" if latencies else "—")
    cols[3].metric("Unknown usage", sum(e.get("prompt_tokens") is None for e in requests))
    embedding_events = [event for event in events if event["kind"] == "embedding_request"]
    if embedding_events:
        embedding_cost = sum(event.get("cost_usd") or 0 for event in embedding_events)
        st.caption(f"Embedding requests in recent log: {len(embedding_events)} · known estimated cost ${embedding_cost:.5f}")
    if requests:
        chart = pd.DataFrame(requests)
        chart["timestamp"] = pd.to_datetime(chart["timestamp"])
        a, b = st.columns(2)
        a.line_chart(chart.set_index("timestamp")[["latency_ms"]])
        b.line_chart(chart.set_index("timestamp")[["cost_usd"]])
    else:
        st.info("Run a chat, question or workflow to populate real telemetry.")
    evaluation = ROOT / "artifacts/evaluation/retrieval.json"
    if evaluation.exists():
        st.subheader("Retrieval quality and latency")
        report = json.loads(evaluation.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame(report["summary"]), hide_index=True, width="stretch")
        st.caption(report["protocol"])
        with st.expander("Full retrieval evaluation"):
            st.json(report)
    outcomes = [e for e in events if e["kind"] in {"code_result", "planning_result"}]
    if outcomes:
        st.subheader("Workflow outcomes")
        st.bar_chart(pd.DataFrame(outcomes).groupby(["kind", "status"]).size().unstack(fill_value=0))
    with st.expander("Request and workflow log"):
        st.dataframe(pd.DataFrame(events), width="stretch")

elif page == "Chat":
    title("Conversational core", "Streaming assistant", "Recent context: 10 messages. Every request records usage, cost and latency.")
    if st.button("Clear conversation"):
        state["chat"] = ChatSession()
        st.rerun()
    for message in state["chat"].history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a technical question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            display, text = st.empty(), ""
            try:
                for event in state["chat"].turn(prompt, LLM(get_connection(), runtime.telemetry)):
                    if event["type"] == "delta":
                        text += event["text"]
                        display.markdown(text+" ▌")
                    elif event["type"] == "stats":
                        display.markdown(text)
                        st.caption(format_stats(event))
                    elif event["type"] == "error":
                        st.error(event["message"])
            except ValueError as exc:
                st.error(str(exc))

elif page == "Knowledge base":
    title("Retrieval-grounded QA", "Answers with evidence", "Inspect sources, compare retrieval modes and add technical documents.")
    with st.expander("Add documents"):
        uploads = st.file_uploader("TXT, Markdown, HTML, PDF or DOCX", type=["txt", "md", "html", "pdf", "docx"],
                                   accept_multiple_files=True)
        ocr = st.checkbox("Enable OCR for scanned PDF pages")
        if st.button("Ingest documents", disabled=not uploads):
            for upload in uploads:
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / Path(upload.name).name
                    path.write_bytes(upload.getvalue())
                    try:
                        with st.spinner(f"Parsing and indexing {upload.name}"):
                            doc = load_document(path, source_uri="upload:"+upload.name, ocr=ocr)
                            result = runtime.index.ingest(doc)
                        st.success(f"{upload.name}: {result['status']}")
                    except Exception as exc:
                        st.error(f"{upload.name}: {type(exc).__name__}: {exc}")
    question = st.text_input("Question", placeholder="How does SQL Server READ COMMITTED isolation prevent dirty reads?")
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "lexical", "rerank"])
    if st.button("Ask the knowledge base", type="primary", disabled=not question):
        try:
            with st.spinner("Retrieving evidence and composing a grounded answer"):
                state["qa_result"] = answer_question(question, runtime.index, LLM(get_connection(), runtime.telemetry),
                                                     runtime.telemetry, mode)
        except Exception as exc:
            st.error(f"Question failed: {type(exc).__name__}: {exc}")
    if result := state.get("qa_result"):
        st.markdown(result["answer"])
        st.caption(f"Retrieval: {result['retrieval']['latency_ms']:.1f} ms · {result['retrieval']['mode']}")
        for source in result["sources"]:
            with st.expander(f"[{source['citation']}] {source['title']}"):
                st.write(source["text"])
                st.json({"version": source["version"], "locators": source["locators"], "license": source["license"]})
                if source["source_uri"].startswith("https://"):
                    st.link_button("Open original source", source["source_uri"])

elif page == "Trip planner":
    title("Tool-calling agent", "Two days in Auckland", "Plans include meals, local transport and overnight accommodation allowances.")
    with st.form("trip"):
        prompt = st.text_input("Preferences", "Plan a relaxed Auckland trip with art, culture and waterfront views.")
        a, b, c = st.columns(3)
        start = a.date_input("Start date", date.today()+timedelta(days=1))
        adults = b.number_input("Adults", min_value=1, max_value=8, value=1)
        budget = c.number_input("Total budget · NZD", min_value=1, value=500)
        mode = st.selectbox("Tool data", ["live", "mock"], help="Mock mode provides fixed, labeled weather and attraction fixtures.")
        run = st.form_submit_button("Build and validate itinerary", type="primary")
    if run:
        try:
            request = TripRequest(prompt=prompt, start_date=start, adults=adults, budget_cents=int(budget*100), mode=mode)
            with st.status("Planning", expanded=True) as progress:
                for event in plan_trip(request, LLM(get_connection(), runtime.telemetry),
                                       TravelTools(runtime.settings.tools_base_url), runtime.telemetry):
                    if event["kind"] == "result":
                        state["itinerary"] = event
                    else:
                        st.write(event)
                progress.update(label="Planning finished", state="complete")
        except Exception as exc:
            st.error(f"Planner failed: {type(exc).__name__}: {exc}")
    if itinerary := state.get("itinerary"):
        st.subheader(itinerary["status"].replace("_", " ").title())
        if itinerary.get("total_cost_cents") is not None:
            st.metric("Estimated trip total", f"NZ${itinerary['total_cost_cents']/100:.2f}")
        st.json(itinerary)
        st.download_button("Download itinerary JSON", json.dumps(itinerary, indent=2), "itinerary.json", "application/json")

elif page == "Code repair":
    title("Generate · Test · Repair", "Code with a feedback loop", "Three attempts maximum. Fixed tests run inside a restricted Docker container.")
    task = st.text_area("Coding task", "Write a Python quicksort function returning a sorted copy without mutating the input.")
    tests = st.text_area("Acceptance tests · frozen for this run", QUICKSORT_TESTS, height=230)
    if st.button("Generate and test", type="primary"):
        try:
            with st.status("Code repair run", expanded=True) as progress:
                runner = make_runner(runtime.settings)
                for event in repair_code(task, LLM(get_connection(), runtime.telemetry), runner,
                                         runtime.telemetry, ROOT / "artifacts/runs", tests or None):
                    st.write(event)
                    if event["kind"] == "code_result":
                        state["code_result"] = event
                progress.update(label="Run finished", state="complete")
        except Exception as exc:
            st.error(f"Code run failed: {type(exc).__name__}: {exc}")
    if result := state.get("code_result"):
        st.json(result)
        root = Path(result["artifact_dir"])
        candidates = sorted(root.glob("attempt-*/solution.py"))
        if candidates:
            code = candidates[-1].read_text(encoding="utf-8")
            st.code(code, language="python")
            st.download_button("Download generated code", code, "solution.py", "text/x-python")
