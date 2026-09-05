"""Verify the complete prepared corpus against the committed canonical snapshot hash."""
import hashlib
import json
from pathlib import Path


def measure(path):
    checksum = hashlib.sha256()
    count = size = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            document = json.loads(line)
            canonical = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            checksum.update((canonical + "\n").encode("utf-8"))
            size += len("\n\n".join(e["text"] for e in document["elements"]).encode("utf-8"))
            count += 1
    return {"documents": count, "text_bytes": size, "canonical_sha256": checksum.hexdigest()}


if __name__ == "__main__":
    expected = json.loads(Path("configs/reproduction.json").read_text(encoding="utf-8"))["corpus"]
    actual = measure(Path("data/corpus/documents.jsonl"))
    if actual != expected:
        raise SystemExit(f"Corpus differs from the evaluated snapshot. Expected {expected}; got {actual}")
    print(json.dumps({"verified": True, **actual}, indent=2))
