"""A common, provenance-preserving representation across document formats."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


class Locator(BaseModel):
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    element: int | None = None
    anchor: str | None = None
    section: list[str] = Field(default_factory=list)


class Element(BaseModel):
    kind: Literal["heading", "paragraph", "code", "table", "list"]
    text: str
    locator: Locator


class Document(BaseModel):
    source_id: str
    source_uri: str
    version: str
    title: str
    format: str
    raw_hash: str
    elements: list[Element]
    license: str = "user-provided"
    warnings: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(element.text for element in self.elements)

    @property
    def content_hash(self) -> str:
        # Ignore incidental formatting for exact-document deduplication.
        return digest(re.sub(r"\s+", " ", self.text).strip())

    @property
    def text_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


class Chunk(BaseModel):
    id: str
    source_id: str
    source_uri: str
    version: str
    title: str
    text: str
    locators: list[Locator]
    ordinal: int
    license: str


class ParseError(ValueError):
    pass


class NeedsOCR(ParseError):
    pass


def safe_relative(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Path escapes its permitted directory")
    return candidate
