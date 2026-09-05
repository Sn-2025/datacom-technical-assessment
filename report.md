# Technical Assessment Report

Implemented streaming chat, an evaluated technical-document RAG service, a two-tool Auckland planner, a bounded Python code-repair workflow, and a Streamlit workbench. FastAPI exposes the workflows. Docker/Compose files are supplied locally; cloud deployment is left to the user.

## Retrieval

The corpus contains **8,183 documents, 53,486,002 parsed UTF-8 bytes, and 57,720 chunks**. It is a pinned Microsoft SQL documentation snapshot under CC-BY-4.0, with MIT code samples. Global exact-element deduplication precedes measurement and chunk overlap. Five loaders preserve source locations; scanned PDFs require explicit OCR.

The measured index uses official `text-embedding-3-small`, 384 dimensions, Chroma HNSW and SQLite BM25 with reciprocal-rank fusion. Local BGE remains configurable. Bulk import checkpoints embeddings and resumes interrupted work.

The fixed dataset has 50 evidence-backed questions: 10 development and 40 held-out, plus five separate unanswerable controls. Questions are AI-authored with verified original evidence, not independently human-graded.

On the 40 held-out questions, hybrid **source Hit@5 is 97.5% (39/40)**; strict evidence-span Hit@5 is **87.5% (35/40)**. With database/model connections warmed but each query-vector cache cleared, complete retrieval median is **234 ms**, p95 **295 ms**, including the embedding API call and excluding generation. Exact-repeat median is **92 ms** and is reported separately. Dense and lexical baselines are included in `artifacts/evaluation/retrieval.json`.

These are single-workstation, sequential measurements, not a load-test guarantee. A separate cold API smoke request took 1.19 seconds for retrieval, illustrating startup/network effects.

A post-hoc GPT-5.4 audit of unchanged nano-generated answers scored **34/40 correct (85%)**, **36/37 produced answers fully citation-supported**, and **5/5 negative controls correctly refused**. The original nano judge scored 77.5% correctness but conflated supported extra detail with error; both judgments are retained. This rubric audit is not new unseen testing or human certification.

## Workflows and verification

Official OpenAI streaming, usage, tool calling and strict JSON probes passed. The actual `chat.py` Hello run reported 20 input tokens, 12 output tokens and 1,345 ms. History is capped at ten messages; endpoint-bound credentials and unknown-cost handling are tested.

The live planner called Open-Meteo and MediaWiki and produced a validated two-day itinerary at **NZ$436**, including food, transport and accommodation allowances. Prices and venue windows are explicitly estimates. Invalid attraction IDs, dates, travel gaps and excess budgets are rejected.

The code assistant freezes acceptance tests and permits three total attempts. Local generation succeeded; execution correctly reported unavailable because this workstation has no Docker engine. Linux CI passed **32/32**, including actual restricted-container tests (run 33937883023, core revision 9a0d55c). Local tests pass **30/30**, with two Docker integrations explicitly skipped. Six UI pages and the real QA HTTP endpoint passed smoke checks.

## Limits

Generation was tested against the user-confirmed official API, not the separate assessment gateway. QA semantic scores are labeled model judgments, not human certification. Reranking is optional and unbenchmarked. Cloud deployment, persistent cloud storage and an external isolated execution backend remain deployment responsibilities.
