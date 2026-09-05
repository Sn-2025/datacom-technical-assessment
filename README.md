# Technical Assessment — AI Engineering Workbench

A Python implementation of streaming chat, cited technical-document QA, a tool-calling Auckland travel planner, and a bounded generate/test/repair assistant. Streamlit exposes the workflows and their actual request telemetry; FastAPI exposes JSON and NDJSON APIs.

## Start locally

Python 3.12 is recommended. From this directory:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env with your provider settings before starting services.
git lfs install
git lfs pull
uv sync --frozen --extra dev
uv run assessment ingest data/corpus/documents.jsonl
# Then open separate terminals:
uv run uvicorn assessment.travel_tools:app --host 127.0.0.1 --port 8001
uv run uvicorn assessment.api:create_app --factory --host 127.0.0.1 --port 8000
uv run streamlit run app.py
```

Open http://127.0.0.1:8501 and API documentation at http://127.0.0.1:8000/docs. The repository now includes the canonical prepared corpus and the fixed QA dataset, but the corpus file is stored through Git LFS and `data/runtime/` is still intentionally excluded. After cloning, copy `.env.example` to `.env`, run `git lfs install && git lfs pull`, set the supplied `OPENAI_API_KEY` plus a separate official `EMBEDDING_API_KEY`, and run `uv run assessment ingest data/corpus/documents.jsonl` once to build the local index. Do not overwrite a working `.env` unless intentionally resetting settings.

### Vector database: installation and startup

The vector database is **ChromaDB** (`chromadb`, pinned to 1.5.9 in the lockfile). `uv sync --frozen --extra dev` installs it into the project's `.venv` along with the other Python dependencies. There is no separate database installer for the default local setup.

- **Current local setup:** `CHROMA_HOST` is unset. The application uses `chromadb.PersistentClient`, an embedded database opened inside the Python process when the knowledge base is first accessed. No Chroma server, Docker container or separate database startup command is needed. Closing the application does not erase the persisted index.
- **Storage:** vectors and the HNSW index live under `data/runtime/<index_id>/chroma/`. Document metadata, text and the BM25 index live alongside them in `knowledge.sqlite`. The prepared index ID is `ac33852da0d355e0`, containing 57,720 chunks. Keep the whole index directory together when backing it up.
- **Fresh clone:** the canonical prepared corpus in `data/corpus/` is committed to the repository, but the large `documents.jsonl` payload is fetched through Git LFS and `data/runtime/` is intentionally not committed. Run `git lfs install && git lfs pull`, then build the local index once with `uv run assessment ingest data/corpus/documents.jsonl`. Rebuild it whenever you intentionally change chunking, embedding, or vector-store configuration.
- **Docker Compose setup:** Compose instead starts a separate `chroma` service using `chromadb/chroma:1.5.9`; the API and UI connect using `CHROMA_HOST=chroma`, `CHROMA_PORT=8000`. `docker compose up` pulls the database image automatically and its data persists in the `chroma-data` volume. This is a separate vector store from the local embedded directory. When switching backends, use a fresh matching metadata/index directory and re-ingest; copying only `knowledge.sqlite` does not populate a new Chroma service.

SQLite is supplied by Python's standard library; no separate SQLite server installation is required. OpenAI embeddings generate vectors, while Chroma stores and searches them—using the OpenAI API does not mean the vector database is hosted by OpenAI.

`OPENAI_API_KEY` or `OPENAI_API_KEY_FILE` configures the **generation** credential for `OPENAI_BASE_URL`. The default base URL is the assessment unifier; `MODEL_NAME=gpt-5.4-nano`. The existing `OPENAI API KEY.txt` is read automatically **only for the official OpenAI URL**, never for the assessment gateway. Official embeddings use `EMBEDDING_API_KEY` against `https://api.openai.com/v1` and do not reuse the assessment key. The Connections page accepts a masked session override and an editable base URL/model ID. A key stays bound to its endpoint; applying another URL cannot reuse it. Never commit keys or the confidential assessment PDF.

Optional local debugging can set `PROFILE=official_test` and `OPENAI_BASE_URL=https://api.openai.com/v1`. Submitted evidence is from the assessment gateway. Pricing for that gateway is unknown unless supplied in `Connection.pricing`; unknown is never shown as zero.

## Chat

```powershell
uv run python chat.py
```

Type `Hello`, `/clear`, or `/quit`. The client streams text, keeps the last **10 user/assistant messages** plus a fixed system instruction, and reports prompt/completion tokens, cached input, estimated USD, first-token latency and complete round-trip latency. Interrupted requests retain unknown usage when the provider did not report it. Every request gets a configuration snapshot and a run ID.

## Knowledge base and evaluation

### Reproduce the submitted results from a fresh clone

The Git repository contains code, the dependency lockfile, the fixed 55-question dataset, source licenses, the canonical prepared corpus, and measured result artifacts. It still excludes API keys, local model caches, exploratory research downloads, runtime indexes, and ephemeral telemetry. That keeps the benchmark inputs fixed while leaving index construction to the local environment.

After cloning, copy `.env.example` to `.env`, fill in the assessment `OPENAI_API_KEY` and a separate official `EMBEDDING_API_KEY`, then run:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
git lfs install
git lfs pull
uv sync --frozen --extra dev
uv run assessment ingest data/corpus/documents.jsonl

# Inspect the plan; performs no downloads or API requests:
uv run python scripts/reproduce.py --dry-run

# Reuse the committed corpus, build a local reproduction index, and evaluate retrieval:
uv run python scripts/reproduce.py

# Also regenerate answers and run the model-judge audit (additional billable API requests):
uv run python scripts/reproduce.py --with-qa
```

This entry point applies the report's exact embedding/chunk settings from `configs/reproduction.json`, regardless of the offline BGE default in `.env.example`. It uses an isolated `data/reproduction/runtime/` index and a **new** `artifacts/reproduction/<run-id>/` result directory on every invocation, so committed results cannot be mistaken for newly executed tests. When a fresh reproduction run needs to rebuild raw downloads instead of reusing the committed prepared corpus, add `--skip-fetch` only if you already have those raw downloads locally. It never regenerates the fixed question set.

Before embedding, the complete prepared corpus must match its committed canonical SHA-256, 8,183 documents and 53,486,002 text bytes. A mismatch stops the run. `uv run python scripts/verify_corpus.py` performs this check separately. Compare new `retrieval.json` with `artifacts/evaluation/retrieval.json`; model judgments, network latency and approximate-neighbor ordering need not be bit-for-bit identical across machines or provider updates.

`uv run pytest -q` is a different verification layer: it uses small deterministic fixtures and does not require this corpus or a provider key. GitHub CI also runs the isolated Docker integration tests. Full-corpus RAG evaluation is the explicit reproduction command above, not a claim that CI re-embeds 50 MiB on every push.

The prepared corpus contains over 50 MiB of globally deduplicated English Microsoft SQL documentation, pinned to a Git commit. Sources retain URL, version, license, headings and line/page/element locations. Corpus bytes are measured **after parsing and before chunk overlap**, not archive size. See `docs/sources.md` and `artifacts/verification/corpus-manifest.json`.

If you intentionally want to rebuild the corpus and index from source instead of using the committed copy, run:

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

The Knowledge base page reads a live, read-only inventory from the current index: document/chunk counts, deduplicated text size, source and format distribution, and an empty/indexed state. Browse documents by title or source, page through results, preview parsed text, or open the original source. Inventory browsing does not initialize embeddings or call a model. The page explicitly distinguishes retrieval-grounded independent questions from Chat's conversation history.

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
if (!(Test-Path .env)) { Copy-Item .env.example .env }
git lfs install
git lfs pull
docker build -f docker/Sandbox.Dockerfile -t assessment-sandbox:local .
docker compose build
docker compose up -d chroma tools runner api ui
```

Supply a backend key through `.env` or a mounted secret; the host's automatic key file is intentionally excluded from images. Pull the Git LFS corpus payload on the host before building or indexing, otherwise `data/corpus/documents.jsonl` remains a small pointer file. The Compose topology is for a Docker host: Chroma and SQLite require persistent storage, and the code runner requires a Docker engine. For your later Cloud Run deployment, preserve those persistence and runner boundaries rather than assuming local volumes or a Docker socket are available. The included GitHub workflow runs tests only; it does not deploy.

After the stack is up, use the `api` container for ingestion, search, evaluation scripts, and ad-hoc CLI commands because it has the application environment plus the mounted `data/` and `artifacts/` directories:

```bash
# Confirm the corpus payload is real, not an LFS pointer:
ls -lh data/corpus
head -5 data/corpus/documents.jsonl

# Build the runtime index inside the running stack:
docker compose exec api python -m assessment.cli ingest data/corpus/documents.jsonl

# Inspect the resulting index:
docker compose exec api python -m assessment.cli stats
docker compose exec api python -m assessment.cli search "How does READ COMMITTED prevent dirty reads?" --mode hybrid
```

For the reproduction and evaluation scripts, stay in the `api` container as well:

```bash
# Retrieval-only reproduction:
docker compose exec api python scripts/reproduce.py --dry-run
docker compose exec api python scripts/reproduce.py

# Full QA rerun and audit; billable provider calls:
docker compose exec api python scripts/reproduce.py --with-qa
docker compose exec api python scripts/evaluate_qa.py
docker compose exec api python -m scripts.audit_qa_scores --judge-model gpt-5.4
```

The production image is intentionally built with `uv sync --no-dev`, so `pytest` and `ruff` are not installed in the long-running `api` container by default. For validation commands that need dev dependencies, run an ephemeral one-off container from the same Compose service:

```bash
# Install dev extras only in the throwaway container, then run checks:
docker compose run --rm api sh -lc "uv sync --frozen --extra dev && ruff check ."
docker compose run --rm api sh -lc "uv sync --frozen --extra dev && pytest -q"

# Explicit billable provider smoke checks:
docker compose run --rm api sh -lc "uv sync --frozen --extra dev && python scripts/smoke_provider.py"
docker compose run --rm api sh -lc "uv sync --frozen --extra dev && python scripts/smoke_planner.py"
```

If you rebuild after changing code or environment wiring, restart the stack and then rerun ingestion when the runtime directory is empty:

```bash
docker compose down
docker compose build --no-cache
docker compose up -d chroma tools runner api ui
docker compose exec api python -m assessment.cli ingest data/corpus/documents.jsonl
```

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
