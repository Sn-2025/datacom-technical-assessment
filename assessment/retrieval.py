"""Transactional document metadata, HNSW vectors and SQLite FTS5/BM25 retrieval."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import Settings
from .documents import Document, digest
from .embedding import chunk_document

STOP = set("a an and are as at be by can do does for from how i in is it of on or that the this to was what when where which who why will with".split())


class KnowledgeIndex:
    def __init__(self, settings: Settings, embedder, vector_client=None):
        self.settings, self.embedder = settings, embedder
        root = settings.runtime_dir / settings.index_id()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "knowledge.sqlite"
        self.lock = threading.RLock()
        if vector_client is None:
            import chromadb

            vector_client = (chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
                if settings.chroma_host else chromadb.PersistentClient(path=str(root / "chroma")))
        self.collection = vector_client.get_or_create_collection(
            name="knowledge_" + settings.index_id(), metadata={"hnsw:space": "cosine"})
        self.reranker = None
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, raw_hash TEXT NOT NULL,
                    text_bytes INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL, text_hash TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(id UNINDEXED, text, title, tokenize='porter unicode61');
                CREATE TABLE IF NOT EXISTS embedding_cache (text_hash TEXT PRIMARY KEY, vector BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            identity = getattr(embedder, "identity", settings.embedding_model)
            previous = db.execute("SELECT value FROM settings WHERE key='embedder_identity'").fetchone()
            if previous and previous[0] != identity:
                raise ValueError("Embedding assets changed; create a new index instead of mixing vectors")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('embedder_identity', ?)", (identity,))

    def connect(self):
        db = sqlite3.connect(self.path, timeout=60)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def ingest(self, document: Document) -> dict:
        with self.lock:
            with self.connect() as db:
                previous = db.execute("SELECT content_hash,raw_hash,payload FROM documents WHERE source_id=?",
                                      (document.source_id,)).fetchone()
                if (previous and previous[0] == document.content_hash and previous[1] == document.raw_hash
                        and json.loads(previous[2])["version"] == document.version
                        and json.loads(previous[2])["license"] == document.license):
                    return {"status": "unchanged", "source_id": document.source_id, "chunks": 0}
                old_ids = [row[0] for row in db.execute("SELECT id FROM chunks WHERE source_id=?", (document.source_id,))]
            chunks = chunk_document(document, self.embedder, self.settings)
            if not chunks:
                raise ValueError("A document must produce at least one chunk")
            new_ids = [chunk.id for chunk in chunks]
            fresh_ids = [i for i in new_ids if i not in set(old_ids)]
            try:
                for start in range(0, len(chunks), 32):
                    batch = chunks[start:start+32]
                    vectors, missing = {}, {}
                    with self.connect() as db:
                        for chunk in batch:
                            key = digest(chunk.text)
                            row = db.execute("SELECT vector FROM embedding_cache WHERE text_hash=?", (key,)).fetchone()
                            if row:
                                vectors[key] = np.frombuffer(row[0], dtype=np.float32).tolist()
                            else:
                                missing[key] = chunk.text
                    if missing:
                        generated = self.embedder.embed(list(missing.values()))
                        with self.connect() as db:
                            for key, vector in zip(missing, generated, strict=True):
                                vector = np.asarray(vector, dtype=np.float32)
                                vectors[key] = vector.tolist()
                                db.execute("INSERT OR IGNORE INTO embedding_cache VALUES (?,?)", (key, vector.tobytes()))
                    self.collection.upsert(ids=[c.id for c in batch],
                        embeddings=[vectors[digest(c.text)] for c in batch],
                        metadatas=[{"source_id": c.source_id} for c in batch])
                # Publish metadata only after all vectors exist. Searches share this process lock.
                with self.connect() as db:
                    for old_id in old_ids:
                        db.execute("DELETE FROM chunk_fts WHERE id=?", (old_id,))
                    db.execute("DELETE FROM chunks WHERE source_id=?", (document.source_id,))
                    for chunk in chunks:
                        db.execute("INSERT INTO chunks VALUES (?,?,?,?)", (chunk.id, document.source_id,
                            digest(chunk.text), chunk.model_dump_json()))
                        db.execute("INSERT INTO chunk_fts VALUES (?,?,?)", (chunk.id, chunk.text, chunk.title))
                    db.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?)", (document.source_id,
                        document.content_hash, document.raw_hash, document.text_bytes, document.model_dump_json()))
            except Exception:
                if fresh_ids:
                    self.collection.delete(ids=fresh_ids)
                raise
            retired = list(set(old_ids)-set(new_ids))
            if retired:
                self.collection.delete(ids=retired)
            return {"status": "indexed", "source_id": document.source_id, "chunks": len(chunks),
                    "text_bytes": document.text_bytes}

    def ingest_many(self, documents):
        """Bulk initial import; bounded batches, durable embedding cache and resumable documents."""
        pending = []
        for document in documents:
            previous = self.document(document.source_id)
            if previous and previous.model_dump() == document.model_dump():
                continue
            if previous:
                yield self.ingest(document)
                continue
            pending.append(document)
            if len(pending) >= 100:
                yield self._import_batch(pending)
                pending = []
        if pending:
            yield self._import_batch(pending)

    def _import_batch(self, documents):
        with self.lock:
            chunks = [chunk for document in documents for chunk in chunk_document(document, self.embedder, self.settings)]
            vectors, missing = {}, {}
            with self.connect() as db:
                for chunk in chunks:
                    key = digest(chunk.text)
                    cached = db.execute("SELECT vector FROM embedding_cache WHERE text_hash=?", (key,)).fetchone()
                    if cached:
                        vectors[key] = np.frombuffer(cached[0], dtype=np.float32).tolist()
                    else:
                        missing[key] = chunk.text
            entries = list(missing.items())
            batches = [entries[start:start+128] for start in range(0, len(entries), 128)]
            def embed_batch(batch):
                generated = self.embedder.embed([text for _, text in batch])
                with self.connect() as db:
                    for (key, _), vector in zip(batch, generated, strict=True):
                        db.execute("INSERT OR IGNORE INTO embedding_cache VALUES (?,?)",
                                   (key, np.asarray(vector, dtype=np.float32).tobytes()))
                return [(key, vector.tolist()) for (key, _), vector in zip(batch, generated, strict=True)]
            workers = 4 if self.settings.embedding_backend == "openai" else 1
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for batch in pool.map(embed_batch, batches):
                    vectors.update(batch)
            # An interrupted upsert leaves only invisible vectors. Re-running uses stable IDs and cached embeddings.
            for start in range(0, len(chunks), 2000):
                batch = chunks[start:start+2000]
                self.collection.upsert(ids=[c.id for c in batch], embeddings=[vectors[digest(c.text)] for c in batch],
                                       metadatas=[{"source_id": c.source_id} for c in batch])
            with self.connect() as db:
                for chunk in chunks:
                    db.execute("INSERT INTO chunks VALUES (?,?,?,?)", (chunk.id, chunk.source_id,
                               digest(chunk.text), chunk.model_dump_json()))
                    db.execute("INSERT INTO chunk_fts VALUES (?,?,?)", (chunk.id, chunk.text, chunk.title))
                for document in documents:
                    db.execute("INSERT INTO documents VALUES (?,?,?,?,?)", (document.source_id, document.content_hash,
                               document.raw_hash, document.text_bytes, document.model_dump_json()))
            return {"status": "indexed", "documents": len(documents), "chunks": len(chunks)}

    def delete(self, source_id: str):
        with self.lock, self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM chunks WHERE source_id=?", (source_id,))]
            for chunk_id in ids:
                db.execute("DELETE FROM chunk_fts WHERE id=?", (chunk_id,))
            db.execute("DELETE FROM chunks WHERE source_id=?", (source_id,))
            db.execute("DELETE FROM documents WHERE source_id=?", (source_id,))
        if ids:
            self.collection.delete(ids=ids)

    def stats(self) -> dict:
        with self.connect() as db:
            count = db.execute("SELECT count(*) FROM documents").fetchone()[0]
            unique = db.execute("SELECT count(DISTINCT content_hash) FROM documents").fetchone()[0]
            size = db.execute("SELECT coalesce(sum(n),0) FROM (SELECT max(text_bytes) n FROM documents GROUP BY content_hash)").fetchone()[0]
            chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"documents": count, "unique_documents": unique, "unique_document_text_bytes": size,
                "chunks": chunks, "index_id": self.settings.index_id()}

    def document(self, source_id: str) -> Document | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM documents WHERE source_id=?", (source_id,)).fetchone()
        return Document.model_validate_json(row[0]) if row else None

    def search(self, query: str, *, mode: str | None = None, k: int = 5) -> dict:
        mode = mode or self.settings.retrieval_mode
        if mode not in {"dense", "lexical", "hybrid", "rerank"}:
            raise ValueError("Unknown retrieval mode")
        if not query.strip() or len(query) > 4000:
            raise ValueError("Provide a non-empty query of at most 4000 characters")
        started = time.perf_counter()
        timings, rankings = {}, []
        with self.lock, self.connect() as db:
            total = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
            limit = min(max(k*6, 30), total)
            if not total:
                return {"hits": [], "latency_ms": (time.perf_counter()-started)*1000, "timings": {}, "mode": mode}
            if mode != "lexical":
                at = time.perf_counter()
                vector = self.embedder.query(query)
                timings["query_embedding_cache_hit"] = getattr(self.embedder, "last_query_cache_hit", False)
                timings["query_embedding_ms"] = (time.perf_counter()-at)*1000
                at = time.perf_counter()
                result = self.collection.query(query_embeddings=[vector], n_results=limit, include=["distances"])
                rankings.append(result["ids"][0])
                timings["vector_search_ms"] = (time.perf_counter()-at)*1000
            if mode != "dense":
                at = time.perf_counter()
                terms = [word for word in re.findall(r"[\w]+", query.lower()) if word not in STOP][:40]
                match = " OR ".join('"'+term+'"' for term in terms)
                rows = db.execute("SELECT id FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts,0,1,3) LIMIT ?",
                                  (match, limit)).fetchall() if match else []
                rankings.append([r[0] for r in rows])
                timings["lexical_search_ms"] = (time.perf_counter()-at)*1000
            scores = defaultdict(float)
            for ranking in rankings:
                for rank, chunk_id in enumerate(ranking, 1):
                    scores[chunk_id] += 1/(60+rank)
            hits, seen = [], set()
            for chunk_id in sorted(scores, key=lambda i: (-scores[i], i)):
                row = db.execute("SELECT text_hash,payload FROM chunks WHERE id=?", (chunk_id,)).fetchone()
                if row is None or row[0] in seen:
                    continue
                seen.add(row[0])
                hits.append({"chunk": json.loads(row[1]), "score": scores[chunk_id]})
                if len(hits) >= max(12, k):
                    break
            if mode == "rerank" and hits:
                at = time.perf_counter()
                if self.reranker is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self.reranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2",
                        cache_dir=str(self.settings.model_cache_dir), threads=self.settings.embedding_threads)
                values = list(self.reranker.rerank(query, [hit["chunk"]["text"] for hit in hits]))
                for hit, score in zip(hits, values, strict=True):
                    hit["rerank_score"] = float(score)
                hits.sort(key=lambda hit: hit["rerank_score"], reverse=True)
                timings["rerank_ms"] = (time.perf_counter()-at)*1000
        return {"hits": hits[:k], "latency_ms": (time.perf_counter()-started)*1000, "timings": timings, "mode": mode}
