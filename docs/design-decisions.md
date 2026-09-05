# Implementation decisions

Status: agreed design and research preparation; the application is not implemented yet.

## Product scope

- Build four independently runnable capabilities sharing the LLM adapter, telemetry, configuration and evaluation infrastructure: streaming chat, retrieval-grounded QA, itinerary planning and a code repair loop.
- Use Python, FastAPI, Pydantic, a vector store, SQLite and Streamlit. Include Docker Compose and the evaluation dashboard in the deliverable.
- Use English technical documentation, English evaluation questions and English deliverables. The knowledge base focuses on Python and data engineering.
- Support TXT, Markdown, HTML, PDF and DOCX through separate source connectors and format parsers, with a common document-element model and source locators. Route scanned PDFs through OCR and report parsing failures.
- Keep source versions, hashes, parser/configuration versions and element locators. Repeated ingestion must be idempotent; updates and deletions must retire stale chunks. A failed update must preserve the last usable version.

## Corpus and evaluation

- Discover corpus sources online from official documentation and official project repositories, pin one version or revision per source, retain attribution and relevant license files, and record download checksums.
- Count more than 50 MB of cleaned, deduplicated text before chunk overlap. Archive sizes, images, translations, repeated releases and alternate formats of the same content do not establish corpus size.
- Research-stage text measurements are provisional until the production ingestion pipeline and global deduplication validate them.
- Validate every supported format using suitable real samples. Corpus volume and format coverage are separate acceptance checks.
- Compare lexical, dense, hybrid and reranked retrieval on the same corpus and evaluation split. Enable additional stages when measured quality gains justify their latency.
- Prepare 50-100 human-reviewed questions, separating development and final test data. Define Hit@5 and evidence Recall@5 explicitly; evaluate unanswerable questions separately. Check citation support and coverage as well as locator validity.
- Retrieval timing includes query embedding, retrieval requests, fusion, reranking and evidence assembly, excluding answer generation. Warm models and indexes, disable whole-query-result caching, and report median/P95 with hardware and concurrency settings.

## Travel and code scope

- Demonstrate a two-day Auckland itinerary for one adult already in Auckland. The demonstration budget includes one night of accommodation, food, local transport and activities. These are explicit demonstration inputs, not silent assumptions for every request.
- Provide real tool paths and clearly labeled, replayable mock fixtures. Record provenance and distinguish quotes, estimates and fixture prices. Validate budget and scheduling constraints in application code.
- Return structured states for success, missing information and infeasible constraints. Log actions, observations and concise decision summaries.
- Implement the code assistant in Python with a complete Python/pytest execution path. The first generation counts toward the maximum of three attempts. Fixed acceptance tests must remain protected from model edits.
- Execute generated code in a restricted environment without provider credentials. Persist attempt artifacts and actual test outcomes.

## Provider and model configuration

- Configure provider, base URL, model identifier, request timeout, output limits and pricing metadata. Keep the API adapter replaceable and validate provider capabilities rather than assuming every OpenAI-compatible endpoint behaves identically.
- Define an `assessment` profile preserving the supplied endpoint/model requirements and an `official_test` profile for development. Do not substitute one provider's evidence for a final assessment-gateway run.
- The user confirmed that `OPENAI API KEY.txt` is for the official OpenAI API. Bind that credential only to `https://api.openai.com/v1`.
- Default generation model: the assessment-specified `gpt-5.4-nano`. Allow optional per-module model overrides for experiments. Record the requested and returned model identifiers on each run; keep official evaluation runs on the required profile/model.
- Keep embedding configuration separate from generation configuration. The embedding model/revision, dimension, normalization and chunking configuration belong to the index identity. An incompatible change requires a rebuilt/versioned index, not mixed vectors.
- Snapshot the non-secret effective configuration at run start so that a UI setting change cannot alter an in-flight run.

## Settings page and credential handling

- Provide a connection selector, base URL, model selector with manual identifier entry, masked API-key input, connection/capability test action and session-credential clear action.
- Read deployment credentials on the backend from environment variables or a mounted secret file. Show only a configured/not-configured indicator; never return an existing secret to the browser.
- A user-entered key is submitted to the application backend and held only in that server-side session by default. Do not persist it to browser storage, application databases, logs, exported configuration or repository files.
- Use a session credential only for its bound connection profile. Switching provider/base URL must not silently reuse another provider's key.
- Instantiate clients with the current session's resolved credentials; avoid process-global mutable keys or cross-session credential caches.
- Persist non-secret settings separately. Optional persistent secret storage would use a local OS credential store or deployment secret manager, with an explicit save action.
- Model HTTP requests originate on the backend. Local development may use loopback transport; shared/remote deployment requires authenticated access and HTTPS.
- CLI and CI remain fully usable through environment configuration without a browser session.

## Read-only connection check

On 2026-09-05 (workspace local date), the supplied local credential successfully authenticated an official OpenAI `GET /v1/models` request (HTTP 200). The returned model list contained `gpt-5.4-nano`. No model-generation request was sent. This confirms authentication and model-list visibility only; generation access, usage reporting, streaming, tool calling and structured-output behavior still need their implementation-stage probes.

The local credential file and common secret locations are ignored by Git. No secret value is included in this document.

## Official references

- [OpenAI authentication and server-side key handling](https://developers.openai.com/api/reference/overview)
- [OpenAI model listing](https://developers.openai.com/api/reference/resources/models/methods/list)
- [GPT-5.4 nano capabilities](https://developers.openai.com/api/docs/models/gpt-5.4-nano)
