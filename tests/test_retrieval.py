import pytest

from assessment.loaders import load_document
from assessment.qa import Claim, GroundedAnswer, answer_question
from assessment.telemetry import Telemetry


def make_document(tmp_path, text, name="document.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_document(path, source_uri="https://example.test/"+name)


def test_idempotent_ingestion_update_and_delete(index, tmp_path):
    document = make_document(tmp_path, "# Transactions\n\nSerializable isolation prevents serialization anomalies.")
    first = index.ingest(document)
    assert first["status"] == "indexed"
    assert index.ingest(document)["status"] == "unchanged"
    assert index.stats()["documents"] == 1
    assert index.search("serialization anomalies", mode="lexical")["hits"]
    updated = make_document(tmp_path, "# Indexes\n\nA B-tree index supports range scans efficiently.")
    index.ingest(updated)
    assert not index.search("serialization", mode="lexical")["hits"]
    assert index.search("range scans", mode="hybrid")["hits"]
    index.delete(document.source_id)
    assert index.stats()["chunks"] == 0


def test_failed_update_preserves_previous_evidence(index, tmp_path, monkeypatch):
    document = make_document(tmp_path, "# Original\n\nOriginal verified evidence about transactions.")
    index.ingest(document)
    updated = make_document(tmp_path, "# Replacement\n\nReplacement evidence about entirely different indexing behavior.")
    def fail(texts):
        raise RuntimeError("embedding failed")
    monkeypatch.setattr(index.embedder, "embed", fail)
    with pytest.raises(RuntimeError):
        index.ingest(updated)
    assert index.document(document.source_id).content_hash == document.content_hash
    assert index.search("verified", mode="lexical")["hits"]


def test_duplicate_documents_do_not_inflate_size(index, tmp_path):
    first = make_document(tmp_path, "# Same\n\nThe same valid evidence text.", "one.md")
    second = make_document(tmp_path, "# Same\n\nThe same valid evidence text.", "two.md")
    index.ingest(first)
    index.ingest(second)
    assert index.stats()["unique_document_text_bytes"] == first.text_bytes
    assert index.stats()["unique_documents"] == 1


def test_unknown_citation_is_rejected(index, tmp_path):
    index.ingest(make_document(tmp_path, "# Python\n\nPython dictionaries preserve insertion order."))
    class Model:
        def structured(self, *args, **kwargs):
            return GroundedAnswer(answerable=True, claims=[Claim(text="Claim", citations=[999])], explanation="")
    result = answer_question("Python dictionaries", index, Model(), Telemetry(tmp_path / "events.sqlite"))
    assert not result["answerable"]
    assert result["claims"] == []


def test_citation_preserves_source_locator(index, tmp_path):
    index.ingest(make_document(tmp_path, "# Python\n\nPython dictionaries preserve insertion order."))
    hit = index.search("dictionaries", mode="lexical")["hits"][0]["chunk"]
    assert hit["source_uri"].startswith("https://example.test/")
    assert hit["locators"][0]["line_start"] == 1
