"""Internal supervisor service; no model credentials are mounted into this service."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .code_assistant import DockerRunner, SandboxUnavailable
from .config import Settings

app = FastAPI(title="Isolated test runner")


class Job(BaseModel):
    code: str = Field(max_length=100000)
    tests: str = Field(max_length=100000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def run(job: Job):
    settings = Settings()
    try:
        return DockerRunner(settings.sandbox_image, settings.sandbox_timeout_s).run(job.code, job.tests)
    except SandboxUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
