# Corpus provenance and attribution

The evaluated corpus is an English technical knowledge base about SQL Server, Azure SQL, T-SQL, database engineering and data integration. It is a bounded subset of **Microsoft SQL documentation**, authored by Microsoft and the MicrosoftDocs contributors.

- Repository: https://github.com/MicrosoftDocs/sql-docs
- Immutable snapshot: `4f78fa5f8e9f4272c016d2c0f95eca31de866c8b`
- Documentation license: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Code samples: MIT. Original license texts are preserved in `docs/licenses/`.
- Original source links are included with every document and returned citation.
- Changes: Markdown/front matter/navigation parsing, global whitespace-normalized exact element deduplication, token-bounded chunking. Code examples are retained. Source documents are not represented as original work by this project's author.

The preparation script sorts T-SQL documentation first, then other engineering documentation, and stops after reaching 51 MiB of parsed, deduplicated text. Same-content elements across files contribute bytes only once. This conservative deduplication can remove a repeated heading from a later document; existing element locators still retain their original section path. Microsoft Docs include directives that depend on its publishing system remain source syntax rather than fabricated expanded content.

`data/corpus_research/mssql/files.jsonl` records per-file SHA-256, immutable URL and Git blob identity. The reproducible downloader validates the pinned Git blob before saving. `data/corpus/manifest.json` records the measured text size and parsing failures; the delivered copy is `artifacts/verification/corpus-manifest.json`. The repository commits the canonical prepared corpus, while exploratory research downloads, runtime indexes, and model caches remain excluded from Git.

Earlier exploration also downloaded Python, pandas, NumPy, PostgreSQL, DuckDB, Kubernetes, Spark, Airflow and Flink documentation. These remain local research assets and are **not counted** toward the evaluated corpus. No additional corpus expansion is required.

BGE-small-en-v1.5 is an optional MIT-licensed local model. The pinned quantized ONNX distribution is `qdrant/bge-small-en-v1.5-onnx-q` at revision `52398278842ec682c6f32300af41344b1c0b0bb2`, weight SHA-256 `51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431`. The evaluated official embedding model and dimensions are recorded separately in each evaluation artifact.
