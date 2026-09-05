"""Fetch an immutable, English Microsoft SQL documentation snapshot, text only."""
from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/corpus_research/mssql"
HEADERS = {"User-Agent": "TechnicalAssessment-CorpusResearch/0.1"}
LOCAL = threading.local()
PREFIXES = ("docs/t-sql/", "docs/relational-databases/", "docs/integration-services/", "docs/connect/",
            "docs/database-engine/", "docs/tools/", "docs/includes/", "docs/sql-server/", "azure-sql/")


def fetch(url):
    parsed = urllib.parse.urlsplit(url)
    if not hasattr(LOCAL, "connections"):
        LOCAL.connections = {}
    for attempt in range(3):
        try:
            connection = LOCAL.connections.setdefault(parsed.hostname,
                http.client.HTTPSConnection(parsed.hostname, timeout=45))
            target = parsed.path + ("?"+parsed.query if parsed.query else "")
            connection.request("GET", target, headers=HEADERS)
            response = connection.getresponse()
            payload = response.read()
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}")
            return payload
        except Exception:
            connection = LOCAL.connections.pop(parsed.hostname, None)
            if connection:
                connection.close()
            if attempt == 2:
                raise
            time.sleep(attempt+1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "snapshot.json"
    if lock.exists():
        metadata = json.loads(lock.read_text())
    else:
        commit = "4f78fa5f8e9f4272c016d2c0f95eca31de866c8b"
        tree = json.loads(fetch(f"https://api.github.com/repos/MicrosoftDocs/sql-docs/git/trees/{commit}?recursive=1"))
        if tree.get("truncated"):
            raise RuntimeError("Refuse an incomplete Git tree")
        files = [item for item in tree["tree"] if item["type"] == "blob" and item["path"].endswith(".md")
                 and item["path"].startswith(PREFIXES) and not any(s in item["path"].lower() for s in
                 ["release-notes", "release_notes", "whats-new", "/toc.", "/previous-versions/"])]
        metadata = {"project": "Microsoft SQL documentation", "repository": "MicrosoftDocs/sql-docs",
            "commit": commit, "license": "CC-BY-4.0 for documentation; MIT for code samples",
            "source_url": f"https://github.com/MicrosoftDocs/sql-docs/tree/{commit}", "files": files}
        lock.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"commit": metadata["commit"], "files": len(metadata["files"]),
                      "raw_bytes_expected": sum(f["size"] for f in metadata["files"])}), flush=True)
    for filename in ("LICENSE", "LICENSE-CODE"):
        (OUT / filename).write_bytes(fetch(f"https://raw.githubusercontent.com/MicrosoftDocs/sql-docs/{metadata['commit']}/{filename}"))
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)

    def download(item):
        path = raw / (hashlib.sha256(item["path"].encode()).hexdigest()+".md")
        url = f"https://raw.githubusercontent.com/MicrosoftDocs/sql-docs/{metadata['commit']}/{item['path']}"
        payload = path.read_bytes() if path.exists() else fetch(url)
        if hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest() != item["sha"]:
            raise ValueError("Git blob checksum mismatch")
        if not path.exists():
            path.write_bytes(payload)
        return {"source_path": item["path"], "raw_path": str(path.relative_to(ROOT)), "source_uri": url,
                "version": metadata["commit"], "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    failures, records = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download, item): item for item in metadata["files"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"path": futures[future]["path"], "error_type": type(exc).__name__})
            if (len(records)+len(failures)) % 200 == 0:
                print(json.dumps({"downloaded": len(records), "failed": len(failures)}), flush=True)
    records.sort(key=lambda r: r["source_path"])
    (OUT / "files.jsonl").write_text("".join(json.dumps(record)+"\n" for record in records), encoding="utf-8")
    result = {**{k: v for k, v in metadata.items() if k != "files"}, "downloaded_files": len(records),
              "raw_bytes": sum(r["bytes"] for r in records), "failures": failures,
              "measurement_status": "raw Markdown only; parsed/deduplicated text must be measured separately"}
    (OUT / "inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
