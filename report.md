# Technical Assessment Report

Implemented streaming chat, an evaluated technical-document RAG service, a two-tool Auckland planner, a bounded Python code-repair workflow, and a Streamlit workbench. FastAPI exposes the workflows. Docker/Compose files are supplied locally; cloud deployment is left to the user.

## Retrieval

The corpus contains **8,183 documents, 53,486,002 parsed UTF-8 bytes, and 57,720 chunks**. It is a pinned Microsoft SQL documentation snapshot under CC-BY-4.0, with MIT code samples. Global exact-element deduplication precedes measurement and chunk overlap. Five loaders preserve source locations; scanned PDFs require explicit OCR.

The measured index uses official `text-embedding-3-small`, 384 dimensions, Chroma HNSW and SQLite BM25 with reciprocal-rank fusion. Local BGE remains configurable. Bulk import checkpoints embeddings and resumes interrupted work.

The fixed dataset has 50 evidence-backed questions: 10 development and 40 held-out, plus five separate unanswerable controls. Questions are AI-authored with verified original evidence, not independently human-graded.

On the 40 held-out questions, hybrid **source Hit@5 is 97.5% (39/40)**; strict evidence-span Hit@5 is **87.5% (35/40)**. With database/model connections warmed but each query-vector cache cleared, complete retrieval median is **241 ms**, p95 **296 ms**, including the embedding API call and excluding generation. Exact-repeat median is **85 ms** and is reported separately. Dense and lexical baselines are included in `artifacts/evaluation/retrieval.json`.

I also ran a small chunking ablation on the same fixed retrieval benchmark to test whether the submitted `320` token / `40` token-overlap setting was materially limiting performance. The committed comparison in `artifacts/evaluation/chunk_ablation/summary.json` keeps every other variable fixed and spot-checks three hybrid held-out indexes: `320/40` (57,720 chunks), `240/40` (78,290 chunks), and `320/20` (56,638 chunks). In that rerun, `240/40` improved strict evidence-span Hit@5 from **87.5% to 92.5%** with unchanged source Hit@5, while `320/20` improved source Hit@5 from **97.5% to 100%** and evidence-span Hit@5 to **90.0%**. Because these ablations were run later on a busier workstation, I treat their latency figures as relative comparisons rather than replacements for the main reported baseline, but they show that chunk granularity is a meaningful retrieval lever and that lower overlap is the best first follow-up candidate.

These are single-workstation, sequential measurements, not a load-test guarantee. A separate cold API smoke request took 1.19 seconds for retrieval, illustrating startup/network effects.

The current held-out QA summary was rerun end-to-end on official OpenAI `gpt-5.4-nano`, rather than the assessment gateway, because the gateway was too slow for full-batch QA generation. That clean rerun scores **32/40 correct (80%)**, with **65%** of answerable questions fully supported by their cited evidence, **95%** of answerable questions carrying citations for factual claims, and **5/5** negative controls correctly refused. A separate post-hoc audit of those unchanged answers, also on official OpenAI `gpt-5.4-nano`, scored **33/40 correct (82.5%)**, with **30/37** produced answers fully citation-supported and **31/37** produced answers carrying citations for all factual claims. These are model judgments, not human certification.

## Workflows and verification

Assessment-gateway generation was re-verified on 2026-09-05 using `gpt-5.4-nano`. Streaming and tool-calling probes passed against the supplied assessment endpoint, but full held-out QA generation was rerun on official OpenAI because the assessment gateway timed out repeatedly on long structured-answer requests. Official OpenAI is therefore the provider for the reported QA answer-generation and audit numbers, and also remains the embedding endpoint. The actual `chat.py` Hello run reported 20 input tokens, 12 output tokens and 1,345 ms. History is capped at ten messages; endpoint-bound credentials and unknown-cost handling are tested.

The live planner called Open-Meteo and MediaWiki and produced a validated two-day itinerary at **NZ$436**, including food, transport and accommodation allowances. Prices and venue windows are explicitly estimates. Invalid attraction IDs, dates, travel gaps and excess budgets are rejected.

The code assistant freezes acceptance tests and permits three total attempts. Local generation succeeded; execution correctly reported unavailable because this workstation has no Docker engine. Linux CI passed **32/32**, including actual restricted-container tests (run 33937883023, core revision 9a0d55c). The latest local pytest run passed **30 tests** with **2 Docker integrations skipped**. Six UI pages and the real QA HTTP endpoint passed smoke checks.

## Limits

The assessment gateway passed streaming and tool-calling smoke checks, but it was not reliable enough for full-batch QA scoring because long structured-answer requests timed out repeatedly; the published QA scores therefore come from official OpenAI instead. QA semantic scores are labeled model judgments, not human certification. Reranking is optional and unbenchmarked. Cloud deployment, persistent cloud storage and an external isolated execution backend remain deployment responsibilities.
