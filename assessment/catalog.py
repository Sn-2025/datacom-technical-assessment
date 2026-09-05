"""Read-only knowledge inventory; browsing never loads an embedding model."""
import json
import sqlite3
from collections import Counter
from urllib.parse import urlsplit


def inventory(settings):
    path = settings.runtime_dir / settings.index_id() / "knowledge.sqlite"
    if not path.exists():
        return {"documents": [], "chunks": 0, "text_bytes": 0, "sources": {}, "formats": {}}
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute("""SELECT source_id, json_extract(payload,'$.title'),
            json_extract(payload,'$.source_uri'), json_extract(payload,'$.format'),
            json_extract(payload,'$.version'), text_bytes FROM documents ORDER BY 2,1""").fetchall()
        size = connection.execute("SELECT coalesce(sum(n),0) FROM (SELECT max(text_bytes) n FROM documents GROUP BY content_hash)").fetchone()[0]
        chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()
    documents = [dict(zip(("source_id", "title", "source_uri", "format", "version", "text_bytes"), row, strict=True)) for row in rows]
    def source(uri):
        parsed = urlsplit(uri)
        if parsed.hostname == "raw.githubusercontent.com" and parsed.path.startswith("/MicrosoftDocs/sql-docs/"):
            return "Microsoft SQL documentation"
        return parsed.hostname or ("Uploaded documents" if parsed.scheme == "upload" else "Local documents")
    return {"documents": documents, "chunks": chunks, "text_bytes": size,
            "sources": dict(Counter(source(d["source_uri"]) for d in documents)),
            "formats": dict(Counter(d["format"] for d in documents))}


def document_preview(settings, source_id):
    path = settings.runtime_dir / settings.index_id() / "knowledge.sqlite"
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute("SELECT payload FROM documents WHERE source_id=?", (source_id,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        connection.close()
