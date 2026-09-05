"""Reproducible local ingestion, search and application entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .documents import Document
from .loaders import load_document
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Ingest a document or prepared documents.jsonl; resumable")
    ingest.add_argument("path", type=Path)
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", choices=["dense", "lexical", "hybrid", "rerank"], default="hybrid")
    commands.add_parser("stats")
    args = parser.parse_args()
    runtime = Runtime()
    if args.command == "ingest":
        if args.path.suffix == ".jsonl":
            with args.path.open(encoding="utf-8") as source:
                documents = (Document.model_validate_json(line) for line in source)
                for result in runtime.index.ingest_many(documents):
                    print(json.dumps(result), flush=True)
        else:
            print(json.dumps(runtime.index.ingest(load_document(args.path))))
        print(json.dumps(runtime.index.stats()))
    elif args.command == "search":
        print(json.dumps(runtime.index.search(args.query, mode=args.mode), indent=2))
    else:
        print(json.dumps(runtime.index.stats(), indent=2))


if __name__ == "__main__":
    main()
