"""HTTP entry points. Secrets are accepted in headers and never returned in payloads."""
from __future__ import annotations

import hmac
import json
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from .code_assistant import make_runner, repair_code
from .config import Connection
from .llm import LLM
from .loaders import SUPPORTED, load_document
from .planner import TravelTools, TripRequest, plan_trip
from .qa import answer_question
from .runtime import Runtime


def create_app(runtime: Runtime | None = None):
    runtime = runtime or Runtime()
    app = FastAPI(title="Technical Assessment Workbench", version="0.1.0")
    app.state.runtime = runtime

    def authorize(authorization: str = Header(default="")):
        expected = runtime.settings.app_access_token.get_secret_value()
        if expected and not hmac.compare_digest(authorization, "Bearer "+expected):
            raise HTTPException(401, "Application access token required")

    def connection(x_provider_key: str = Header(default=""), x_provider_url: str = Header(default=""),
                   x_model_name: str = Header(default=""), _=Depends(authorize)):
        default = runtime.settings.connection()
        if x_provider_url and x_provider_url.rstrip("/") != default.base_url:
            if not x_provider_key:
                raise HTTPException(422, "A changed endpoint requires its own credential")
            default = Connection(base_url=x_provider_url, model=x_model_name or default.model,
                                 api_key=SecretStr(x_provider_key), profile="session")
        elif x_provider_key or x_model_name:
            default = Connection(base_url=default.base_url, model=x_model_name or default.model,
                                 api_key=SecretStr(x_provider_key) if x_provider_key else default.api_key,
                                 profile="session")
        return default

    @app.get("/health")
    def health():
        return {"status": "ok", "index_loaded": runtime._index is not None}

    @app.get("/metrics", dependencies=[Depends(authorize)])
    def metrics():
        return runtime.telemetry.recent()

    @app.get("/knowledge/stats", dependencies=[Depends(authorize)])
    def stats():
        return runtime.index.stats()

    @app.get("/sources/{source_id}", dependencies=[Depends(authorize)])
    def source(source_id: str):
        document = runtime.index.document(source_id)
        if not document:
            raise HTTPException(404, "Source not found")
        return document

    @app.post("/knowledge/ingest", dependencies=[Depends(authorize)])
    def ingest(file: UploadFile, ocr: bool = False):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED:
            raise HTTPException(422, "Unsupported file format")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("input"+suffix)
            size = 0
            with path.open("wb") as output:
                while data := file.file.read(1024*1024):
                    size += len(data)
                    if size > 256*1024*1024:
                        raise HTTPException(413, "File exceeds 256 MiB")
                    output.write(data)
            try:
                document = load_document(path, source_uri="upload:"+Path(file.filename or "input").name, ocr=ocr)
                return runtime.index.ingest(document)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc

    @app.delete("/sources/{source_id}", dependencies=[Depends(authorize)])
    def delete_source(source_id: str):
        runtime.index.delete(source_id)
        return {"status": "deleted"}

    @app.post("/qa")
    def qa(request: Question, conn=Depends(connection)):
        return answer_question(request.question, runtime.index, LLM(conn, runtime.telemetry),
                               runtime.telemetry, request.mode)

    @app.post("/chat")
    def chat(request: Messages, conn=Depends(connection)):
        messages = [{"role": "system", "content": "You are a helpful technical assistant."},
                    *[m.model_dump() for m in request.messages[-10:]]]
        events = LLM(conn, runtime.telemetry).stream(messages)
        return StreamingResponse((json.dumps(event)+"\n" for event in events), media_type="application/x-ndjson")

    @app.post("/plan")
    def plan(request: TripRequest, conn=Depends(connection)):
        events = plan_trip(request, LLM(conn, runtime.telemetry), TravelTools(runtime.settings.tools_base_url), runtime.telemetry)
        return StreamingResponse((json.dumps(event)+"\n" for event in events), media_type="application/x-ndjson")

    @app.post("/code")
    def code(request: CodeRequest, conn=Depends(connection)):
        runner = make_runner(runtime.settings)
        events = repair_code(request.task, LLM(conn, runtime.telemetry), runner, runtime.telemetry,
                             Path("artifacts/runs"), request.tests)
        return StreamingResponse((json.dumps(event)+"\n" for event in events), media_type="application/x-ndjson")

    return app


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: str | None = None


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=20000)


class Messages(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=10)


class CodeRequest(BaseModel):
    task: str = Field(min_length=1, max_length=10000)
    tests: str | None = Field(default=None, max_length=100000)
