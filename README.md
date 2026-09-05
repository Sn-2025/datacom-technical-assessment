# Technical Assessment — AI Engineering Workbench

A Python implementation of streaming chat, cited technical-document QA, a tool-calling Auckland travel planner, and a bounded generate/test/repair assistant. Streamlit exposes the workflows and their actual request telemetry; FastAPI exposes JSON and NDJSON APIs.

## Start locally

Python 3.12 is recommended. From this directory:

```powershell
uv sync --frozen --extra dev
Copy-Item .env.example .env
# Edit .env with your provider settings, then open separate terminals:
uv run uvicorn assessment.travel_tools:app --host 127.0.0.1 --port 8001
uv run uvicorn assessment.api:create_app --factory --host 127.0.0.1 --port 8000
uv run streamlit run app.py
```

Open http://127.0.0.1:8501 and API documentation at http://127.0.0.1:8000/docs. On the prepared workstation, dependencies and the corpus already exist. The local `.env` selects official `text-embedding-3-small` embeddings; `.env.example` defaults to optional offline BGE embeddings. Do not overwrite the prepared `.env` unless intentionally resetting settings.

`OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` configures the backend credential. For local development only, the existing `OPENAI API KEY.txt` is read automatically **only for the official OpenAI URL**. The Connections page accepts a masked session override and an editable base URL/model ID. A key stays bound to its endpoint; applying another URL cannot reuse it. Clear removes the session override; a separately configured backend credential remains available. Never commit keys or the confidential assessment PDF.

Generation profiles are environment configurations: `PROFILE=official_test` with `OPENAI_BASE_URL=https://api.openai.com/v1`; `PROFILE=assessment` with the separately supplied assessment gateway and its own key. `MODEL_NAME=gpt-5.4-nano` is the supplied default. The report identifies which provider was actually tested. Pricing for other providers/models is unknown unless supplied in `Connection.pricing`; unknown is never shown as zero.

## Chat

```powershell
uv run python chat.py
```

Type `Hello`, `/clear`, or `/quit`. The client streams text, keeps the last **10 user/assistant messages** plus a fixed system instruction, and reports prompt/completion tokens, cached input, estimated USD, first-token latency and complete round-trip latency. Interrupted requests retain unknown usage when the provider did not report it. Every request gets a configuration snapshot and a run ID.

## Knowledge base and evaluation

The prepared corpus contains over 50 MiB of globally deduplicated English Microsoft SQL documentation, pinned to a Git commit. Sources retain URL, version, license, headings and line/page/element locations. Corpus bytes are measured **after parsing and before chunk overlap**, not archive size. See `docs/sources.md` and `artifacts/verification/corpus-manifest.json`.

For a fresh clone, these commands reproduce the corpus and index. Fetching is explicit; application startup never downloads more corpus:

```powershell
uv run python scripts/fetch_sql_corpus.py
uv run python scripts/prepare_corpus.py
uv run python scripts/fetch_embedding.py
uv run assessment ingest data/corpus/documents.jsonl
uv run assessment stats
uv run assessment search "How does READ COMMITTED prevent dirty reads?" --mode hybrid
uv run python scripts/evaluate_retrieval.py
uv run python scripts/evaluate_qa.py
uv run python -m scripts.audit_qa_scores --judge-model gpt-5.4
```

`fetch_embedding.py` installs a checksum-verified BGE model and tokenizer. Local embeddings use the weights; official embeddings reuse the tokenizer for conservative, structure-aware chunking. Set `EMBEDDING_BACKEND=openai`, `EMBEDDING_MODEL=text-embedding-3-small`, and `EMBEDDING_DIMENSIONS=384` for the tested API path. A separate `EMBEDDING_API_KEY` is required when generation uses another provider. Changing embedding/chunk configuration creates a different index. Model assets are fingerprinted to prevent silently mixing incompatible vectors.

Loaders support TXT, Markdown, HTML, PDF and DOCX. Scanned PDFs return an explicit OCR requirement; enable OCR and install Tesseract locally, or use the supplied image. Re-import is resumable; changed sources replace old chunks. Single-document ingestion is available through the UI, `assessment ingest file.pdf`, and `POST /knowledge/ingest`.

Retrieval compares dense HNSW, SQLite FTS5/BM25, and reciprocal-rank fusion. Optional cross-encoder reranking downloads its own model on first use and is **not included in the default benchmark**. Answers contain atomic claims with citation IDs and expandable evidence; missing or invalid citations produce abstention. Citation ID validation cannot prove semantic entailment: the separate QA evaluation checks that with an explicitly labeled model judge.

`evals/questions.jsonl` contains 50 evidence-backed questions (10 development, 40 held-out) and five unanswerable controls. Questions were AI-authored and their gold evidence verified against the source; they are reviewable, not represented as independently human-graded. Metrics distinguish source Hit@5 from stricter evidence-span Hit@5. First-query latency clears the query-vector cache; repeat-query latency uses the bounded 256-entry cache. Both include the whole retrieval pipeline; neither includes answer generation. The benchmark saves raw per-question measurements as well as medians/p95s. Do not compare repeat-cache numbers with another system's uncached numbers.

`qa.jsonl` preserves the generated answers and original nano judgments. `qa-audit.json` re-scores those unchanged answers with GPT-5.4 after finding rubric errors in the original judge. The audited answer correctness is 34/40; 36/37 affirmative answers have fully supported citations. All five negative controls abstain. This is a post-hoc automated audit, not an independent human score or a fresh held-out run. Judge-model pricing is left unknown unless explicitly configured.

## Travel agent

The form confirms dates, adult count and an NZD budget; free text supplies preferences. `POST /plan` accepts the equivalent typed request and streams action events plus a final JSON result. Example body:

```json
{"prompt":"Plan a relaxed two-day Auckland trip under NZ$500", "start_date":"2026-09-06", "days":2,"adults":1,"budget_cents":50000,"mode":"mock"}
```

Use dates within the weather provider's forecast range for `live`. The agent calls two HTTP tools: Open-Meteo weather and MediaWiki geographic attraction evidence. `mock` returns clearly labeled fixtures through the same tool service. A curated catalog supplies IDs and **planning estimates**, not live booking prices or guaranteed opening hours. Budget includes food, local transport, activities and one accommodation night between each pair of days. Deterministic checks enforce dates, budget, travel gaps, time windows and indoor choices on high-rain days. Bounded validation failures are reported honestly rather than declared mathematically infeasible.

## Code generation and repair

The Code repair page and `POST /code` accept a Python task and optional pytest acceptance tests. The initial generation counts as attempt one; at most **three total attempts** write `solution.py`, run fixed tests and feed actual errors back. Progress, code and results are retained under `artifacts/runs/`. A built-in quicksort contract covers duplicates, negatives, empty input and nonmutation. Other tasks can use user tests or a separately generated contract frozen before implementation.

Generated code requires the isolated Docker runner. This workstation has no Docker engine: local code execution reports `sandbox_unavailable`, never executes generated code on the host, and does not fabricate passing tests. The runner was tested on Linux CI. It has no network, provider keys or host workspace mount; resource limits, read-only root filesystem and protected root-owned tests restrict execution. The internal runner service controls Docker and must not be publicly exposed.

## Containers and handoff

Docker files are delivered locally. No cloud deployment is performed. With Docker available:

```powershell
docker build -f docker/Sandbox.Dockerfile -t assessment-sandbox:local .
docker compose build
docker compose up -d chroma tools runner api ui
```

Supply a backend key through `.env` or a mounted secret; the host's automatic key file is intentionally excluded from images. The Compose topology is for a Docker host: Chroma and SQLite require persistent storage, and the code runner requires a Docker engine. For your later Cloud Run deployment, preserve those persistence and runner boundaries rather than assuming local volumes or a Docker socket are available. The included GitHub workflow runs tests only; it does not deploy.

## Validation and layout

```powershell
uv run ruff check .
uv run pytest -q
# Explicit real provider checks (billable):
uv run python scripts/smoke_provider.py
uv run python scripts/smoke_planner.py
```

Offline tests use deterministic model fixtures and a real local Chroma/SQLite index. Those fixtures never generate reported RAG benchmark metrics. Linux CI additionally runs actual restricted-container tests when `RUN_DOCKER_TESTS=1`. `artifacts/verification/` records provider and workflow evidence; `artifacts/evaluation/` records RAG results. See `report.md` for measured results and limitations.

Main modules: `config.py` connection scope; `llm.py` provider adapter; `telemetry.py` SQLite events; `loaders.py` document parsing; `embedding.py` vector models; `retrieval.py` indexing/search; `qa.py` citation handling; `planner.py` agent/validators; `code_assistant.py` repair loop; `api.py` HTTP boundary. `app.py` is the Streamlit workbench, and `docker/sandbox_runner.py` is the trusted test supervisor.
