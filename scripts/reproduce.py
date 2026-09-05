"""Rebuild the pinned corpus/index and run fresh evaluations, without overwriting delivered evidence."""
import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse existing raw downloads; still prepare and verify them")
    parser.add_argument("--with-qa", action="store_true", help="Also make billable generation and model-judge requests")
    parser.add_argument("--dry-run", action="store_true", help="Show steps without downloading, building or calling any API")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/reproduction.json").read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment.update(config["environment"])
    environment["CHROMA_HOST"] = ""  # Reproduce the measured embedded store, not an inherited remote service.
    environment["RUNTIME_DIR"] = str(root / "data/reproduction/runtime")
    environment["MODEL_CACHE_DIR"] = str(root / "data/model_cache")
    environment["EVALUATION_DIR"] = str(root / "artifacts/reproduction" / uuid.uuid4().hex)
    steps = []
    if not args.skip_fetch:
        steps.append(["scripts/fetch_sql_corpus.py"])
    steps += [["scripts/prepare_corpus.py"], ["scripts/verify_corpus.py"], ["scripts/fetch_embedding.py"],
              ["-m", "assessment.cli", "ingest", "data/corpus/documents.jsonl"],
              ["scripts/evaluate_retrieval.py"]]
    if args.with_qa:
        steps += [["scripts/evaluate_qa.py"], ["-m", "scripts.audit_qa_scores", "--judge-model", "gpt-5.4-nano"]]
    print("Official OpenAI embeddings are billable and require your own credential. Results:", environment["EVALUATION_DIR"], flush=True)
    if not args.dry_run:
        subprocess.run([sys.executable, "-c", "from assessment.config import OFFICIAL_URL, Settings; s=Settings(); "
            "assert s.connection().api_key.get_secret_value(), 'Set OPENAI_API_KEY for the assessment gateway'; "
            "assert s.embedding_backend != 'openai' or s.embedding_api_key.get_secret_value() or "
            "s.openai_base_url.rstrip('/') == OFFICIAL_URL, "
            "'Set EMBEDDING_API_KEY for official OpenAI embeddings'"],
            cwd=root, env=environment, check=True)
    for step in steps:
        print("RUN", " ".join(["python", *step]), flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, *step], cwd=root, env=environment, check=True)
    if not args.dry_run:
        output = Path(environment["EVALUATION_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "reproduction-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
