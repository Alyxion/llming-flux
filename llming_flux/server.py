"""Factory for standalone MCP servers (stdio transport) with resources.

Usage in a domain package::

    from llming_flux.server import create_document_kb_server, run_server

    server = create_document_kb_server(
        name="linus",
        prefix="linus",
        data_dir=Path("data"),
        tools=TOOLS,
        query_expander=MyExpander(),
    )
"""

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ImageContent, Resource, ResourceTemplate, TextContent,
    BlobResourceContents, TextResourceContents, Tool,
)

from .database import Database
from .query_expander import QueryExpander
from .media_router import guess_mime

logger = logging.getLogger(__name__)


def create_document_kb_server(
    name: str,
    prefix: str,
    data_dir: Path,
    tools: list[dict[str, Any]],
    query_expander: QueryExpander | None = None,
    metadata_loader: Callable[[str], dict] | None = None,
    uri_scheme: str | None = None,
) -> Server:
    """Create a standalone MCP server for a document knowledge base.

    Args:
        name: Server name (e.g. ``"agenticlisa"``).
        prefix: Tool name prefix (e.g. ``"lisa"``).
        data_dir: Path to the data directory (must contain a ``.db`` file).
        tools: Tool definitions list (with ``name``, ``description``, ``inputSchema``).
        query_expander: Optional domain-specific query expander.
        metadata_loader: Optional ``filename -> dict`` for JSON sidecar enrichment.
        uri_scheme: Resource URI scheme (default: ``{prefix}://``).
    """
    scheme = uri_scheme or f"{prefix}://"
    server = Server(name)

    # Find database
    db_path = data_dir / f"{prefix}.db"
    if not db_path.exists():
        db_path = data_dir / "lisa.db"
    if not db_path.exists():
        dbs = list(data_dir.glob("*.db"))
        if not dbs:
            raise RuntimeError(f"No database found in {data_dir}")
        db_path = dbs[0]

    db = Database(db_path, query_expander or QueryExpander())
    images_dir = data_dir / "images"
    pdfs_dir = data_dir / "pdfs"
    _load_meta = metadata_loader or (lambda _fn: {})

    # Load image hints
    hints_file = data_dir / "image_hints.json"
    image_hints: dict[str, str] = {}
    if hints_file.exists():
        try:
            image_hints = json.loads(hints_file.read_text())
        except Exception:
            pass

    # ── Tools ──────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                for t in tools]

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
        try:
            if tool_name == f"{prefix}_get_image":
                return await _handle_get_image(arguments)
            result = _dispatch(tool_name, arguments)
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as e:
            logger.exception(f"Tool error: {tool_name}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    async def _handle_get_image(args: dict) -> list[TextContent | ImageContent]:
        img = db.get_image(args["query"], args.get("image_type"), args.get("language"))
        if not img:
            return [TextContent(type="text", text=json.dumps({"error": f"Image not found for '{args['query']}'."}))]
        meta = {"filename": img.filename, "type": img.image_type, "language": img.language,
                "source": img.source_document, "hint": image_hints.get(img.local_path, ""),
                "resource_uri": f"{scheme}image/{img.local_path}"}
        contents: list[TextContent | ImageContent] = [
            TextContent(type="text", text=json.dumps(meta, indent=2))]
        fp = images_dir / img.local_path
        if fp.exists() and fp.stat().st_size < 5 * 1024 * 1024:
            contents.append(ImageContent(
                type="image", data=base64.b64encode(fp.read_bytes()).decode(),
                mimeType=guess_mime(fp.name)))
        return contents

    def _dispatch(tool_name: str, args: dict) -> Any:
        """Route tool calls to the database."""
        p = prefix
        if tool_name == f"{p}_overview":
            ov = db.get_knowledge_overview()
            return {"documents": ov.document_count, "tables": ov.table_count,
                    "table_rows": ov.total_table_rows, "images": ov.image_count,
                    "topics": ov.topics[:50], "tables_summary": ov.tables[:20],
                    "image_types": ov.image_types}
        elif tool_name == f"{p}_search":
            query, limit = args["query"], args.get("limit", 20)
            seen: set[tuple] = set()
            combined: list[dict] = []
            for r in db.search(query, limit * 2, args.get("type")):
                fn = r.metadata.get("filename") or r.metadata.get("table") or ""
                key = (r.result_type, fn)
                if key in seen or not fn:
                    continue
                seen.add(key)
                m = _load_meta(fn)
                entry: dict[str, Any] = {"type": r.result_type, "filename": fn,
                                          "title": m.get("title", r.title),
                                          "summary": m.get("summary", r.snippet or ""),
                                          "source_pdf": r.metadata.get("source_pdf"),
                                          "page": r.metadata.get("page")}
                if r.result_type == "table":
                    tbl = db.get_table(fn)
                    if tbl:
                        entry.update(columns=tbl.headers, row_count=tbl.row_count, sample_values=tbl.sample_values)
                elif r.result_type == "table_row":
                    entry["matched_row"] = r.metadata.get("data", {})
                combined.append(entry)
            return combined[:limit]
        elif tool_name == f"{p}_batch_fetch":
            batch: dict[str, list] = {"documents": [], "tables": []}
            for dr in args.get("documents", []):
                idf, sec = dr["identifier"], dr.get("section")
                if sec:
                    c = db.get_document_section(idf, sec)
                    batch["documents"].append({"filename": idf, "title": _load_meta(idf).get("title", ""),
                                               "section": sec, "content": c} if c else
                                              {"filename": idf, "error": f"Section '{sec}' not found"})
                else:
                    doc = db.get_document(idf)
                    if doc:
                        batch["documents"].append({
                            "filename": doc.filename, "title": _load_meta(doc.filename).get("title", doc.title),
                            "page": doc.page_number, "source_pdf": doc.source_pdf,
                            "sections": list(doc.sections.keys()), "content": doc.content})
                    else:
                        batch["documents"].append({"filename": idf, "error": "Document not found"})
            for tr in args.get("tables", []):
                idf = tr["identifier"]
                qr = db.query_table(idf, tr.get("filters"), tr.get("columns"), tr.get("limit", 20))
                if qr:
                    batch["tables"].append({
                        "filename": qr.table_name, "title": _load_meta(qr.table_name).get("title", ""),
                        "columns": qr.headers, "total_rows": qr.total_rows, "returned_rows": qr.returned_rows,
                        "filters_applied": tr.get("filters") or {}, "rows": qr.rows})
                else:
                    batch["tables"].append({"filename": idf, "error": "Table not found"})
            return batch
        elif tool_name == f"{p}_get_document":
            sec = args.get("section")
            if sec:
                c = db.get_document_section(args["identifier"], sec)
                return {"section": sec, "content": c} if c else {"error": f"Section '{sec}' not found"}
            doc = db.get_document(args["identifier"])
            if not doc:
                return {"error": f"Document '{args['identifier']}' not found"}
            m = _load_meta(doc.filename)
            return {"filename": doc.filename, "title": m.get("title", doc.title),
                    "page": doc.page_number, "source_pdf": doc.source_pdf,
                    "tags": doc.tags, "sections": list(doc.sections.keys()), "content": doc.content}
        elif tool_name == f"{p}_get_table":
            if args.get("info_only"):
                tbl = db.get_table(args["identifier"])
                if not tbl:
                    return {"error": f"Table not found"}
                m = _load_meta(tbl.filename)
                return {"filename": tbl.filename, "title": m.get("title", tbl.title),
                        "columns": tbl.headers, "row_count": tbl.row_count, "sample_values": tbl.sample_values}
            qr = db.query_table(args["identifier"], args.get("filters"), args.get("columns"), args.get("limit", 20))
            if not qr:
                return {"error": f"Table not found"}
            m = _load_meta(qr.table_name)
            return {"table": qr.table_name, "title": m.get("title", ""), "columns": qr.headers,
                    "total_rows": qr.total_rows, "returned_rows": qr.returned_rows, "rows": qr.rows}
        elif tool_name == f"{p}_list_content":
            ct, lim = args.get("content_type", "documents"), args.get("limit", 50)
            if ct == "documents":
                return [{"filename": d.filename, "title": _load_meta(d.filename).get("title", d.title),
                         "page": d.page_number, "tags": d.tags} for d in db.list_documents(lim)]
            elif ct == "tables":
                return [{"filename": t.filename, "title": _load_meta(t.filename).get("title", t.title),
                         "columns": t.headers, "rows": t.row_count} for t in db.list_tables()[:lim]]
            elif ct == "images":
                return [{"filename": i.filename, "type": i.image_type, "language": i.language,
                         "resource_uri": f"{scheme}image/{i.local_path}"}
                        for i in db.search_images("", limit=lim)]
            return {"error": f"Unknown content type: {ct}"}
        return {"error": f"Unknown tool: {tool_name}"}

    # ── Resources ──────────────────────────────────────────

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return [
            ResourceTemplate(uriTemplate=f"{scheme}image/{{path}}", name="Image",
                             description="Images from the knowledge base", mimeType="image/png"),
            ResourceTemplate(uriTemplate=f"{scheme}pdf/{{filename}}", name="Source PDF",
                             description="Source PDF documents", mimeType="application/pdf"),
            ResourceTemplate(uriTemplate=f"{scheme}document/{{filename}}", name="Document",
                             description="Markdown knowledge documents", mimeType="text/markdown"),
        ]

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        resources: list[Resource] = []
        if pdfs_dir.exists():
            for pdf in sorted(pdfs_dir.glob("*.pdf")):
                resources.append(Resource(uri=f"{scheme}pdf/{pdf.name}", name=pdf.name,
                                          description=f"Source PDF: {pdf.name}",
                                          mimeType="application/pdf", size=pdf.stat().st_size))
        products_dir = images_dir / "products"
        if products_dir.exists():
            for img in sorted(products_dir.glob("*"))[:50]:
                resources.append(Resource(uri=f"{scheme}image/products/{img.name}", name=img.name,
                                          description=image_hints.get(f"products/{img.name}", img.stem),
                                          mimeType=guess_mime(img.name), size=img.stat().st_size))
        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> list[TextResourceContents | BlobResourceContents]:
        u = str(uri)
        if u.startswith(f"{scheme}image/"):
            path = u[len(f"{scheme}image/"):]
            if ".." in path:
                raise ValueError("Invalid path")
            fp = images_dir / path
            if not fp.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return [BlobResourceContents(uri=u, blob=base64.b64encode(fp.read_bytes()).decode(),
                                          mimeType=guess_mime(fp.name))]
        elif u.startswith(f"{scheme}pdf/"):
            fn = u[len(f"{scheme}pdf/"):]
            if ".." in fn or "/" in fn:
                raise ValueError("Invalid filename")
            fp = pdfs_dir / fn
            if not fp.exists():
                raise FileNotFoundError(f"PDF not found: {fn}")
            return [BlobResourceContents(uri=u, blob=base64.b64encode(fp.read_bytes()).decode(),
                                          mimeType="application/pdf")]
        elif u.startswith(f"{scheme}document/"):
            fn = u[len(f"{scheme}document/"):]
            doc = db.get_document(fn)
            if not doc:
                raise FileNotFoundError(f"Document not found: {fn}")
            return [TextResourceContents(uri=u, text=doc.content, mimeType="text/markdown")]
        raise ValueError(f"Unknown URI scheme: {u}")

    return server


async def run_server(server: Server):
    """Run an MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
