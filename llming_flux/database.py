"""SQLite + FTS5 database for document-knowledge MCP agents."""

import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Any

from .query_expander import QueryExpander, build_fts_query
from .models import (
    Document, Table, Image, KnowledgeOverview, SearchResult, TableQueryResult,
)


class Database:
    """SQLite database with FTS5 full-text search.

    Args:
        db_path: Path to the SQLite database file.
        query_expander: Optional query expander for bilingual / synonym search.
            Defaults to the base ``QueryExpander`` (tokenise + stopwords only).
    """

    def __init__(self, db_path: Path | str, query_expander: QueryExpander | None = None):
        self.db_path = Path(db_path)
        self._qe = query_expander or QueryExpander()
        self._connection: sqlite3.Connection | None = None

    @contextmanager
    def connection(self):
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        try:
            yield self._connection
        finally:
            pass

    def close(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    # ── Schema ─────────────────────────────────────────────

    def create_tables(self):
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    filename TEXT UNIQUE NOT NULL,
                    title TEXT,
                    page_number INTEGER,
                    source_ref TEXT,
                    product_name TEXT,
                    tags TEXT,
                    content TEXT,
                    sections TEXT,
                    source_pdf TEXT
                );
                CREATE TABLE IF NOT EXISTS tables (
                    id INTEGER PRIMARY KEY,
                    filename TEXT UNIQUE NOT NULL,
                    title TEXT,
                    headers TEXT NOT NULL,
                    row_count INTEGER,
                    sample_values TEXT
                );
                CREATE TABLE IF NOT EXISTS table_rows (
                    id INTEGER PRIMARY KEY,
                    table_id INTEGER REFERENCES tables(id),
                    row_number INTEGER,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_document TEXT,
                    image_type TEXT,
                    language TEXT,
                    tags TEXT,
                    local_path TEXT
                );
                CREATE TABLE IF NOT EXISTS pdf_page_map (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    page_number INTEGER,
                    source_pdf TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
                CREATE INDEX IF NOT EXISTS idx_documents_page ON documents(page_number);
                CREATE INDEX IF NOT EXISTS idx_tables_filename ON tables(filename);
                CREATE INDEX IF NOT EXISTS idx_table_rows_table ON table_rows(table_id);
                CREATE INDEX IF NOT EXISTS idx_images_type ON images(image_type);
                CREATE INDEX IF NOT EXISTS idx_images_source ON images(source_document);
                CREATE INDEX IF NOT EXISTS idx_pdf_page_map ON pdf_page_map(filename);
            """)
            conn.commit()

    def create_fts_index(self):
        with self.connection() as conn:
            conn.executescript("""
                DROP TABLE IF EXISTS doc_search;
                DROP TABLE IF EXISTS table_search;
                DROP TABLE IF EXISTS row_search;
                CREATE VIRTUAL TABLE doc_search USING fts5(
                    filename, title, content, tags, sections,
                    content='documents', content_rowid='id');
                INSERT INTO doc_search(rowid, filename, title, content, tags, sections)
                    SELECT id, filename, title, content, tags, sections FROM documents;
                CREATE VIRTUAL TABLE table_search USING fts5(
                    filename, title, headers, sample_values,
                    content='tables', content_rowid='id');
                INSERT INTO table_search(rowid, filename, title, headers, sample_values)
                    SELECT id, filename, title, headers, sample_values FROM tables;
                CREATE VIRTUAL TABLE row_search USING fts5(
                    data, content='table_rows', content_rowid='id');
                INSERT INTO row_search(rowid, data) SELECT id, data FROM table_rows;
            """)
            conn.commit()

    # ── Insert ─────────────────────────────────────────────

    def insert_document(self, doc: Document) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(filename,title,page_number,source_ref,product_name,tags,content,sections,source_pdf) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (doc.filename, doc.title, doc.page_number, doc.source,
                 doc.product_name, json.dumps(doc.tags), doc.content,
                 json.dumps(doc.sections), doc.source_pdf))
            conn.commit()
            return cur.lastrowid

    def insert_table(self, table: Table) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT OR REPLACE INTO tables (filename,title,headers,row_count,sample_values) "
                "VALUES (?,?,?,?,?)",
                (table.filename, table.title, json.dumps(table.headers),
                 table.row_count, json.dumps(table.sample_values)))
            conn.commit()
            return cur.lastrowid

    def insert_table_row(self, table_id: int, row_number: int, data: dict[str, Any]) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT INTO table_rows (table_id,row_number,data) VALUES (?,?,?)",
                (table_id, row_number, json.dumps(data, default=str)))
            conn.commit()
            return cur.lastrowid

    def insert_image(self, img: Image) -> int:
        with self.connection() as conn:
            cur = conn.execute(
                "INSERT INTO images (filename,source_document,image_type,language,tags,local_path) "
                "VALUES (?,?,?,?,?,?)",
                (img.filename, img.source_document, img.image_type,
                 img.language, json.dumps(img.tags), img.local_path))
            conn.commit()
            return cur.lastrowid

    def insert_pdf_page_map(self, filename: str, page_number: int, source_pdf: str):
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pdf_page_map (filename,page_number,source_pdf) VALUES (?,?,?)",
                (filename, page_number, source_pdf))
            conn.commit()

    # ── Query ──────────────────────────────────────────────

    def get_knowledge_overview(self) -> KnowledgeOverview:
        with self.connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            table_count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
            row_count = conn.execute("SELECT COUNT(*) FROM table_rows").fetchone()[0]
            image_count = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            topics: set[str] = set()
            for row in conn.execute("SELECT tags FROM documents WHERE tags IS NOT NULL"):
                if row[0]:
                    topics.update(json.loads(row[0]))
            tables = []
            for row in conn.execute("SELECT filename,title,headers,row_count FROM tables ORDER BY filename"):
                tables.append({"filename": row["filename"], "title": row["title"],
                               "columns": json.loads(row["headers"]), "rows": row["row_count"]})
            image_types = [r[0] for r in conn.execute(
                "SELECT DISTINCT image_type FROM images WHERE image_type IS NOT NULL")]
            return KnowledgeOverview(
                document_count=doc_count, table_count=table_count,
                image_count=image_count, total_table_rows=row_count,
                topics=sorted(topics), tables=tables, image_types=image_types)

    def search(self, query: str, limit: int = 20, search_type: str | None = None) -> list[SearchResult]:
        expanded = self._qe.expand(query)
        fts = build_fts_query(expanded)
        results: list[SearchResult] = []
        with self.connection() as conn:
            if search_type is None or search_type == "document":
                try:
                    for row in conn.execute(
                        "SELECT d.id,d.filename,d.title,d.content,d.page_number,d.source_pdf "
                        "FROM doc_search ds JOIN documents d ON ds.rowid=d.id "
                        "WHERE doc_search MATCH ? ORDER BY rank LIMIT ?", (fts, limit)):
                        results.append(SearchResult(
                            result_type="document", id=row["id"],
                            title=row["title"] or row["filename"],
                            snippet=self._extract_snippet(row["content"], query),
                            metadata={"filename": row["filename"], "page": row["page_number"],
                                      "source_pdf": row["source_pdf"]}))
                except sqlite3.OperationalError:
                    pass
            if search_type is None or search_type == "table":
                try:
                    for row in conn.execute(
                        "SELECT t.id,t.filename,t.title,t.headers,t.row_count "
                        "FROM table_search ts JOIN tables t ON ts.rowid=t.id "
                        "WHERE table_search MATCH ? ORDER BY rank LIMIT ?", (fts, limit)):
                        headers = json.loads(row["headers"])
                        results.append(SearchResult(
                            result_type="table", id=row["id"],
                            title=row["title"] or row["filename"],
                            snippet=f"Columns: {', '.join(headers[:5])}{'...' if len(headers)>5 else ''}",
                            metadata={"filename": row["filename"], "columns": headers,
                                      "rows": row["row_count"]}))
                except sqlite3.OperationalError:
                    pass
            if search_type is None or search_type == "row":
                try:
                    for row in conn.execute(
                        "SELECT tr.id,tr.table_id,tr.row_number,tr.data,t.filename,t.title "
                        "FROM row_search rs JOIN table_rows tr ON rs.rowid=tr.id "
                        "JOIN tables t ON tr.table_id=t.id "
                        "WHERE row_search MATCH ? ORDER BY rank LIMIT ?", (fts, limit)):
                        data = json.loads(row["data"])
                        results.append(SearchResult(
                            result_type="table_row", id=row["id"],
                            title=f"{row['title'] or row['filename']} - Row {row['row_number']}",
                            snippet=self._format_row_snippet(data, query),
                            metadata={"table": row["filename"], "row_number": row["row_number"],
                                      "data": data}))
                except sqlite3.OperationalError:
                    pass
            if search_type is None or search_type == "image":
                for row in conn.execute(
                    "SELECT * FROM images WHERE filename LIKE ? OR source_document LIKE ? "
                    "OR image_type LIKE ? LIMIT ?",
                    (f"%{query}%", f"%{query}%", f"%{query}%", limit)):
                    results.append(SearchResult(
                        result_type="image", id=row["id"], title=row["filename"],
                        snippet=f"Type: {row['image_type'] or 'unknown'}",
                        metadata={"filename": row["filename"], "type": row["image_type"],
                                  "language": row["language"], "path": row["local_path"]}))
        return results[:limit]

    @staticmethod
    def _extract_snippet(content: str, query: str, ctx: int = 100) -> str:
        if not content:
            return ""
        pos = content.lower().find(query.lower())
        if pos == -1:
            return (content[:200] + "...") if len(content) > 200 else content
        s = max(0, pos - ctx)
        e = min(len(content), pos + len(query) + ctx)
        snip = content[s:e]
        if s > 0: snip = "..." + snip
        if e < len(content): snip += "..."
        return snip.replace("\n", " ")

    @staticmethod
    def _format_row_snippet(data: dict, query: str, max_items: int = 3) -> str:
        items: list[str] = []
        ql = query.lower()
        for k, v in data.items():
            vs = str(v)
            if ql in k.lower() or ql in vs.lower():
                items.insert(0, f"{k}: {vs[:50]}")
            elif len(items) < max_items:
                items.append(f"{k}: {vs[:50]}")
        return "; ".join(items[:max_items])

    def get_document(self, identifier: str) -> Document | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE filename=? OR filename LIKE ? OR title LIKE ? LIMIT 1",
                (identifier, f"%{identifier}%", f"%{identifier}%")).fetchone()
            if not row:
                return None
            source_pdf = row["source_pdf"]
            if not source_pdf:
                pm = conn.execute("SELECT source_pdf FROM pdf_page_map WHERE filename=?",
                                  (row["filename"],)).fetchone()
                if pm:
                    source_pdf = pm[0]
            return Document(
                id=row["id"], filename=row["filename"], title=row["title"],
                page_number=row["page_number"], source=row["source_ref"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                content=row["content"] or "",
                sections=json.loads(row["sections"]) if row["sections"] else {},
                source_pdf=source_pdf)

    def get_document_section(self, identifier: str, section_name: str) -> str | None:
        doc = self.get_document(identifier)
        if not doc:
            return None
        for name, content in doc.sections.items():
            if name.lower() == section_name.lower():
                return content
        return None

    def list_documents(self, limit: int = 100) -> list[Document]:
        with self.connection() as conn:
            return [Document(
                id=r["id"], filename=r["filename"], title=r["title"],
                page_number=r["page_number"],
                tags=json.loads(r["tags"]) if r["tags"] else [],
                source_pdf=r["source_pdf"],
            ) for r in conn.execute(
                "SELECT id,filename,title,page_number,tags,source_pdf FROM documents "
                "ORDER BY page_number,filename LIMIT ?", (limit,))]

    def get_table(self, identifier: str) -> Table | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM tables WHERE filename=? OR filename LIKE ? OR title LIKE ? LIMIT 1",
                (identifier, f"%{identifier}%", f"%{identifier}%")).fetchone()
            if not row:
                return None
            return Table(
                id=row["id"], filename=row["filename"], title=row["title"],
                headers=json.loads(row["headers"]), row_count=row["row_count"],
                sample_values=json.loads(row["sample_values"]) if row["sample_values"] else {})

    def query_table(self, table_identifier: str, filters: dict[str, Any] | None = None,
                    columns: list[str] | None = None, limit: int = 100, offset: int = 0
                    ) -> TableQueryResult | None:
        table = self.get_table(table_identifier)
        if not table:
            return None
        with self.connection() as conn:
            rows, matched = [], 0
            for row in conn.execute(
                "SELECT row_number,data FROM table_rows WHERE table_id=? ORDER BY row_number",
                (table.id,)):
                data = json.loads(row["data"])
                if filters:
                    ok = True
                    for k, v in filters.items():
                        if k not in data or str(v).lower() not in str(data[k]).lower():
                            ok = False; break
                    if not ok:
                        continue
                matched += 1
                if matched <= offset:
                    continue
                rows.append({k: v for k, v in data.items() if k in columns} if columns else data)
                if len(rows) >= limit:
                    break
            return TableQueryResult(
                table_name=table.filename,
                headers=columns if columns else table.headers,
                rows=rows, total_rows=table.row_count, returned_rows=len(rows))

    def list_tables(self) -> list[Table]:
        with self.connection() as conn:
            return [Table(
                id=r["id"], filename=r["filename"], title=r["title"],
                headers=json.loads(r["headers"]), row_count=r["row_count"],
            ) for r in conn.execute("SELECT * FROM tables ORDER BY filename")]

    def get_image(self, identifier: str, image_type: str | None = None,
                  language: str | None = None) -> Image | None:
        with self.connection() as conn:
            q = "SELECT * FROM images WHERE (filename LIKE ? OR source_document LIKE ? OR local_path LIKE ?)"
            p: list = [f"%{identifier}%", f"%{identifier}%", f"%{identifier}%"]
            if image_type:
                q += " AND image_type=?"; p.append(image_type)
            if language:
                q += " AND (language=? OR language IS NULL)"; p.append(language)
            q += " ORDER BY CASE WHEN language=? THEN 0 ELSE 1 END LIMIT 1"
            p.append(language or "en")
            row = conn.execute(q, p).fetchone()
            if not row:
                return None
            return Image(
                id=row["id"], filename=row["filename"],
                source_document=row["source_document"], image_type=row["image_type"],
                language=row["language"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                local_path=row["local_path"])

    def search_images(self, query: str, image_type: str | None = None,
                      limit: int = 20) -> list[Image]:
        with self.connection() as conn:
            q = "SELECT * FROM images WHERE (filename LIKE ? OR source_document LIKE ?)"
            p: list = [f"%{query}%", f"%{query}%"]
            if image_type:
                q += " AND image_type=?"; p.append(image_type)
            q += " LIMIT ?"; p.append(limit)
            return [Image(
                id=r["id"], filename=r["filename"],
                source_document=r["source_document"], image_type=r["image_type"],
                language=r["language"],
                tags=json.loads(r["tags"]) if r["tags"] else [],
                local_path=r["local_path"],
            ) for r in conn.execute(q, p)]

    def get_pdf_for_page(self, filename: str) -> tuple[str | None, int | None]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT source_pdf,page_number FROM pdf_page_map WHERE filename=?",
                (filename,)).fetchone()
            return (row["source_pdf"], row["page_number"]) if row else (None, None)
