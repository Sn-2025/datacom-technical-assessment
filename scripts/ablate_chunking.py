"""Spot-check chunk-size and overlap sensitivity on the fixed retrieval benchmark."""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
from pathlib import Path

import numpy as np

from assessment.config import Settings
from assessment.documents import Document
from assessment.runtime import Runtime


def parse_variant(spec: str) -> tuple[str, int, int]:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError("Variants must be TOKEN_COUNT:OVERLAP[:LABEL]")
    chunk_tokens = int(parts[0])
    chunk_overlap = int(parts[1])
    label = parts[2] if len(parts) == 3 else f"{chunk_tokens}tok_{chunk_overlap}ov"
    return label, chunk_tokens, chunk_overlap


def build_settings(base: Settings, *, chunk_tokens: int, chunk_overlap: int) -> Settings:
    return Settings(
        profile=base.profile,
        openai_base_url=base.openai_base_url,
        openai_api_key=base.openai_api_key.get_secret_value(),
        openai_api_key_file=base.openai_api_key_file,
        model_name=base.model_name,
        runtime_dir=base.runtime_dir,
        model_cache_dir=base.model_cache_dir,
        chroma_host="",
        chroma_port=base.chroma_port,
        embedding_model=base.embedding_model,
        embedding_backend=base.embedding_backend,
        embedding_dimensions=base.embedding_dimensions,
        embedding_api_key=base.embedding_api_key.get_secret_value(),
        embedding_threads=base.embedding_threads,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
        retrieval_mode=base.retrieval_mode,
        sandbox_image=base.sandbox_image,
        sandbox_timeout_s=base.sandbox_timeout_s,
        sandbox_runner_url=base.sandbox_runner_url,
        app_access_token=base.app_access_token.get_secret_value(),
        tools_base_url=base.tools_base_url,
    )


def ingest_corpus(runtime: Runtime, corpus_path: Path) -> None:
    with corpus_path.open(encoding="utf-8") as source:
        documents = (Document.model_validate_json(line) for line in source)
        for _ in runtime.index.ingest_many(documents):
            pass


def evaluate_runtime(runtime: Runtime) -> dict:
    index = runtime.index
    assert index.stats()["unique_document_text_bytes"] >= 50 * 1024 * 1024, "Finish the full corpus import before benchmarking"
    questions = [
        json.loads(line)
        for line in Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["answerable"]
    ]
    assert len(questions) >= 20
    index.search("database consistency", mode="hybrid")
    rows = []
    random.Random(20260905).shuffle(questions)
    for mode in ("lexical", "dense", "hybrid"):
        for item in questions:
            cache = getattr(index.embedder, "query_cache", None)
            if cache is not None:
                cache.clear()
            initial = index.search(item["question"], mode=mode)
            warm = index.search(item["question"], mode=mode)
            gold = set(item["gold_source_ids"])
            hits = warm["hits"]
            source_hit = any(hit["chunk"]["source_id"] in gold for hit in hits)
            quote = " ".join(item["evidence_quote"].split())
            evidence_hit = any(quote in " ".join(hit["chunk"]["text"].split()) for hit in hits)
            rows.append(
                {
                    "id": item["id"],
                    "split": item["split"],
                    "mode": mode,
                    "source_hit_at_5": source_hit,
                    "evidence_hit_at_5": evidence_hit,
                    "first_query_ms": initial["latency_ms"],
                    "repeat_query_ms": warm["latency_ms"],
                    "first_timings": initial["timings"],
                    "repeat_timings": warm["timings"],
                    "top5_ids": [hit["chunk"]["id"] for hit in hits],
                }
            )
        print(mode, "complete", flush=True)
    summary = []
    for mode in ("lexical", "dense", "hybrid"):
        for split in ("dev", "heldout"):
            subset = [row for row in rows if row["mode"] == mode and row["split"] == split]
            if not subset:
                continue
            summary.append(
                {
                    "mode": mode,
                    "split": split,
                    "questions": len(subset),
                    "source_hit_at_5": statistics.mean(row["source_hit_at_5"] for row in subset),
                    "evidence_hit_at_5": statistics.mean(row["evidence_hit_at_5"] for row in subset),
                    **{
                        f"{phase}_{metric}_ms": float(function([row[f"{phase}_query_ms"] for row in subset]))
                        for phase in ("first", "repeat")
                        for metric, function in (
                            ("median", statistics.median),
                            ("p95", lambda values: np.percentile(values, 95)),
                        )
                    },
                }
            )
    return {
        "corpus": index.stats(),
        "config": runtime.settings.index_config(),
        "platform": platform.platform(),
        "protocol": "One warm-up, shuffled fixed questions. First: query embedding cache cleared, DB warm. "
        "Repeat: immediate exact repeat with bounded embedding cache. All measured totals include query embedding, "
        "vector/lexical retrieval, fusion and metadata; generation excluded. No answer/result cache.",
        "gold": "AI-authored, exact source evidence verified; source hit and strict evidence-span hit are distinct.",
        "summary": summary,
        "rows": rows,
    }


def hybrid_heldout_row(report: dict) -> dict:
    return next(item for item in report["summary"] if item["mode"] == "hybrid" and item["split"] == "heldout")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        dest="variants",
        help="TOKEN_COUNT:OVERLAP[:LABEL]. Default runs 320:40:baseline, 240:40:smaller_chunks, 320:20:lower_overlap.",
    )
    args = parser.parse_args()
    variants = args.variants or [
        ("baseline", 320, 40),
        ("smaller_chunks", 240, 40),
        ("lower_overlap", 320, 20),
    ]
    root = Path(__file__).resolve().parents[1]
    corpus_path = root / "data/corpus/documents.jsonl"
    output_root = root / "artifacts/evaluation/chunk_ablation"
    output_root.mkdir(parents=True, exist_ok=True)
    base = Settings()
    reports = []
    for label, chunk_tokens, chunk_overlap in variants:
        print(f"=== {label} ({chunk_tokens}/{chunk_overlap}) ===", flush=True)
        settings = build_settings(base, chunk_tokens=chunk_tokens, chunk_overlap=chunk_overlap)
        runtime = Runtime(settings)
        ingest_corpus(runtime, corpus_path)
        report = evaluate_runtime(runtime)
        destination = output_root / f"{label}.json"
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        selected = hybrid_heldout_row(report)
        reports.append(
            {
                "label": label,
                "chunk_tokens": chunk_tokens,
                "chunk_overlap": chunk_overlap,
                "index_id": report["corpus"]["index_id"],
                "chunks": report["corpus"]["chunks"],
                "documents": report["corpus"]["documents"],
                "source_hit_at_5": selected["source_hit_at_5"],
                "evidence_hit_at_5": selected["evidence_hit_at_5"],
                "first_median_ms": selected["first_median_ms"],
                "first_p95_ms": selected["first_p95_ms"],
                "repeat_median_ms": selected["repeat_median_ms"],
                "repeat_p95_ms": selected["repeat_p95_ms"],
                "artifact": str(destination.relative_to(root)),
            }
        )
    baseline = reports[0]
    comparison = []
    for item in reports:
        comparison.append(
            {
                **item,
                "delta_source_hit_at_5": item["source_hit_at_5"] - baseline["source_hit_at_5"],
                "delta_evidence_hit_at_5": item["evidence_hit_at_5"] - baseline["evidence_hit_at_5"],
                "delta_first_median_ms": item["first_median_ms"] - baseline["first_median_ms"],
                "delta_repeat_median_ms": item["repeat_median_ms"] - baseline["repeat_median_ms"],
            }
        )
    summary = {
        "question_set": "evals/questions.jsonl answerable subset; hybrid held-out row shown here. Full per-mode rows are in each per-variant artifact.",
        "baseline_label": baseline["label"],
        "variants": comparison,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
