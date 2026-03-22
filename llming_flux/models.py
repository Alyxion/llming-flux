"""Pydantic models for document-knowledge MCP agents."""

from pydantic import BaseModel
from typing import Any


class Document(BaseModel):
    """A markdown document from a knowledge base."""
    id: int | None = None
    filename: str
    title: str | None = None
    page_number: int | None = None
    source: str | None = None
    product_name: str | None = None
    tags: list[str] = []
    content: str = ""
    sections: dict[str, str] = {}
    source_pdf: str | None = None


class Table(BaseModel):
    """A data table from a knowledge base."""
    id: int | None = None
    filename: str
    title: str | None = None
    headers: list[str] = []
    row_count: int = 0
    sample_values: dict[str, list[str]] = {}


class TableRow(BaseModel):
    """A single row from a table."""
    table_id: int | None = None
    table_name: str | None = None
    row_number: int
    data: dict[str, Any]


class Image(BaseModel):
    """An image from a knowledge base."""
    id: int | None = None
    filename: str
    source_document: str | None = None
    image_type: str | None = None
    language: str | None = None
    tags: list[str] = []
    local_path: str = ""


class KnowledgeOverview(BaseModel):
    """Overview of available knowledge."""
    document_count: int
    table_count: int
    image_count: int
    total_table_rows: int
    topics: list[str] = []
    tables: list[dict[str, Any]] = []
    image_types: list[str] = []


class SearchResult(BaseModel):
    """A search result from any content type."""
    result_type: str
    id: int
    title: str
    snippet: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = {}


class TableQueryResult(BaseModel):
    """Result of a table query."""
    table_name: str
    headers: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    returned_rows: int
