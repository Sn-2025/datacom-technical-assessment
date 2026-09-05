"""Prepare existing downloaded SQL docs; never downloads additional material."""
from __future__ import annotations

import json
import re
from pathlib import Path

from assessment.documents import digest
from assessment.loaders import load_document


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "data/corpus"
    destination.mkdir(parents=True, exist_ok=True)
    seen, failures = set(), []
    count = size = 0
    with (destination / "documents.jsonl").open("w", encoding="utf-8") as output:
        records = [json.loads(line) for line in
            (root / "data/corpus_research/mssql/files.jsonl").read_text(encoding="utf-8").splitlines()]
        records.sort(key=lambda r: (not r["source_path"].startswith("docs/t-sql/"),
                                    "/system-" in r["source_path"], r["source_path"]))
        for record in records:
            path = root / record["raw_path"].replace("\\", "/")
            try:
                assert digest(path.read_bytes()) == record["sha256"]
                document = load_document(path, source_uri=record["source_uri"],
                    version=record["version"], license="CC-BY-4.0; code MIT")
                unique = []
                for element in document.elements:
                    key = digest(re.sub(r"\s+", " ", element.text).strip())
                    if key not in seen:
                        seen.add(key)
                        unique.append(element)
                document.elements = unique
                if not unique:
                    continue
                output.write(document.model_dump_json() + "\n")
                size += document.text_bytes
                count += 1
                if count % 500 == 0:
                    print(json.dumps({"documents": count, "unique_text_bytes": size}), flush=True)
                if size >= 51 * 1024 * 1024:
                    break
            except Exception as exc:
                failures.append({"source": record["source_path"], "error": type(exc).__name__})
    manifest = {"documents": count, "unique_text_bytes": size,
        "minimum_50_MiB_met": size >= 50 * 1024 * 1024,
        "deduplication": "Global whitespace-normalized exact element hashes; code retained",
        "source": "MicrosoftDocs/sql-docs", "license": "CC-BY-4.0; code MIT", "failures": failures}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
