"""Measure complete retrieval, with cache state explicit and no LLM in the timed path."""
import json
import platform
import random
import statistics
from pathlib import Path

import numpy as np

from assessment.runtime import Runtime


def main():
    runtime = Runtime()
    index = runtime.index
    assert index.stats()["unique_document_text_bytes"] >= 50 * 1024 * 1024, "Finish the full corpus import before benchmarking"
    questions = [json.loads(line) for line in Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["answerable"]]
    assert len(questions) >= 20
    index.search("database consistency", mode="hybrid")  # Load model, connection and DB pages.
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
            # Source hit is intentionally distinct from stricter evidence-span hit.
            source_hit = any(hit["chunk"]["source_id"] in gold for hit in hits)
            quote = " ".join(item["evidence_quote"].split())
            evidence_hit = any(quote in " ".join(hit["chunk"]["text"].split()) for hit in hits)
            rows.append({"id": item["id"], "split": item["split"], "mode": mode,
                "source_hit_at_5": source_hit, "evidence_hit_at_5": evidence_hit,
                "first_query_ms": initial["latency_ms"], "repeat_query_ms": warm["latency_ms"],
                "first_timings": initial["timings"], "repeat_timings": warm["timings"],
                "top5_ids": [hit["chunk"]["id"] for hit in hits]})
        print(mode, "complete", flush=True)
    summary = []
    for mode in ("lexical", "dense", "hybrid"):
        for split in ("dev", "heldout"):
            subset = [row for row in rows if row["mode"] == mode and row["split"] == split]
            if not subset:
                continue
            summary.append({"mode": mode, "split": split, "questions": len(subset),
                "source_hit_at_5": statistics.mean(row["source_hit_at_5"] for row in subset),
                "evidence_hit_at_5": statistics.mean(row["evidence_hit_at_5"] for row in subset),
                **{f"{phase}_{metric}_ms": float(function([row[f"{phase}_query_ms"] for row in subset]))
                   for phase in ("first", "repeat") for metric, function in
                   (("median", statistics.median), ("p95", lambda values: np.percentile(values, 95)))}})
    result = {"corpus": index.stats(), "config": runtime.settings.index_config(), "platform": platform.platform(),
        "protocol": "One warm-up, shuffled fixed questions. First: query embedding cache cleared, DB warm. "
                    "Repeat: immediate exact repeat with bounded embedding cache. All measured totals include query embedding, "
                    "vector/lexical retrieval, fusion and metadata; generation excluded. No answer/result cache.",
        "gold": "AI-authored, exact source evidence verified; source hit and strict evidence-span hit are distinct.",
        "summary": summary, "rows": rows}
    destination = Path("artifacts/evaluation")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
