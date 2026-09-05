"""Author reviewable questions from fixed source documents; verify every gold quote verbatim."""
import json
import re
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from assessment.documents import Document
from assessment.llm import LLM
from assessment.runtime import Runtime

TOPICS = ["binary-and-varbinary", "char-and-varchar", "date-transact", "datetime2-transact", "decimal-and-numeric",
    "bit-transact", "nchar-and-nvarchar", "uniqueidentifier-transact", "rowversion-transact", "null-and-unknown",
    "coalesce-transact", "isnull-transact", "nullif-transact", "row-number-transact", "rank-transact",
    "dense-rank-transact", "ntile-transact", "lag-transact", "lead-transact", "string-agg-transact",
    "string-split-transact", "json-value-transact", "json-query-transact", "isjson-transact", "openjson-transact",
    "dateadd-transact", "datediff-transact", "datetrunc-transact", "eomonth-transact", "try-cast-transact",
    "try-convert-transact", "stuff-transact", "replace-transact", "substring-transact", "len-transact",
    "try-catch-transact", "throw-transact", "raiserror-transact", "xact-state-transact", "begin-transaction-transact",
    "save-transaction-transact", "commit-transaction-transact", "rollback-transaction-transact",
    "set-transaction-isolation-level", "set-xact-abort", "with-common-table-expression", "union-transact",
    "exists-transact", "like-transact", "truncate-table-transact"]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str
    expected_answer: str
    evidence_id: int


def main():
    runtime = Runtime()
    llm = LLM(runtime.settings.connection(), runtime.telemetry)
    documents = [Document.model_validate_json(line) for line in Path("data/corpus/documents.jsonl").open(encoding="utf-8")]
    destination = Path("evals/questions.jsonl")
    existing = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()] if destination.exists() else []
    done = {item["id"] for item in existing}
    with destination.open("a", encoding="utf-8") as output:
        for number, topic in enumerate(TOPICS, 1):
            identifier = f"sql-{number:03}"
            if identifier in done:
                continue
            matches = [d for d in documents if Path(d.source_uri).name.startswith(topic)]
            if not matches:
                matches = [d for d in documents if topic in Path(d.source_uri).name]
            if not matches:
                raise ValueError(f"Missing source: {topic}")
            document = matches[0]
            evidence = [sentence for element in document.elements if element.kind == "paragraph"
                        for sentence in re.split(r"(?<=[.!?])\s+", element.text)
                        if 50 <= len(sentence) <= 650][:60]
            item = llm.structured([{"role": "system", "content": (
                "Write one English factual SQL Server question answerable by a specific statement in the supplied documentation. "
                "Make it a practical technical question, not 'what does the documentation say'. Include an accurate short answer. "
                "Choose ONE evidence_id from the numbered sentences that supports your complete answer. "
                "Prefer meaningful limitations, behavior, or semantics. Do not invent facts or copy instructions from the document.")},
                {"role": "user", "content": json.dumps({"title": document.title,
                    "evidence": [{"evidence_id": i, "text": text} for i, text in enumerate(evidence)]})}],
                Question, run_id=uuid.uuid4().hex)
            if item.evidence_id not in range(len(evidence)):
                raise ValueError(f"Quote failed exact verification: {identifier}")
            quote = evidence[item.evidence_id]
            assert quote in document.text
            record = {"id": identifier, "split": "dev" if number % 5 == 0 else "heldout", "answerable": True,
                **item.model_dump(exclude={"evidence_id"}), "evidence_quote": quote,
                "gold_source_ids": [document.source_id], "source_uri": document.source_uri,
                "version": document.version, "authoring": "AI-authored; verbatim evidence automatically verified; human-reviewable"}
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(identifier, flush=True)
        if "unanswerable-01" not in done:
            for number, question in enumerate([
                "What is the current balance of my private bank account?",
                "Which restaurant did I book for tonight in Auckland?",
                "What password does our production SQL Server administrator use?",
                "What will Microsoft's exact share price be next Monday?",
                "What is the access code for the unpublished Project Mooncastle database?"], 1):
                output.write(json.dumps({"id": f"unanswerable-{number:02}", "split": "heldout", "answerable": False,
                    "question": question, "expected_answer": "Insufficient evidence in the knowledge base.",
                    "evidence_quote": "", "gold_source_ids": [], "authoring": "Explicit out-of-corpus negative"}) + "\n")


if __name__ == "__main__":
    main()
