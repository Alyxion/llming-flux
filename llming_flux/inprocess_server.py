"""Base InProcessMCPServer for document-knowledge agents.

Subclass ``DocumentKBServer`` and override the hooks to build a
domain-specific MCP agent (tool prefix, prompt, metadata loader, renderers).
"""

import json
import logging
from pathlib import Path
from typing import Any

try:
    from llming_models.tools.mcp.connection import InProcessMCPServer
except ImportError:
    from abc import ABC, abstractmethod

    class InProcessMCPServer(ABC):
        @abstractmethod
        async def list_tools(self) -> list[dict]: ...
        @abstractmethod
        async def call_tool(self, name: str, arguments: dict) -> str: ...
        async def get_prompt_hints(self) -> list[str]:
            return []
        async def get_client_renderers(self) -> list[dict[str, str]]:
            return []

from .database import Database

logger = logging.getLogger(__name__)


_FLUX_SOURCES_MARKER = "\n\n__FLUX_SOURCES__\n"


def parse_tool_result(result: str) -> tuple[str, list[dict] | None]:
    """Split a call_tool() result into (json_str, sources_or_None).

    Helper for consumers that need to json.loads() the main result while
    also extracting the optional ``__FLUX_SOURCES__`` trailer.
    """
    if _FLUX_SOURCES_MARKER in result:
        json_part, sources_json = result.split(_FLUX_SOURCES_MARKER, 1)
        try:
            return json_part, json.loads(sources_json)
        except (json.JSONDecodeError, TypeError):
            return json_part, None
    return result, None


def _estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def make_tool_definitions(prefix: str, description_context: str = "knowledge base") -> list[dict[str, Any]]:
    """Generate the standard 7-tool set with the given prefix.

    Tool names become ``{prefix}_overview``, ``{prefix}_search``, etc.
    """
    ctx = description_context
    return [
        {"name": f"{prefix}_overview", "displayName": f"{prefix.upper()} Overview", "icon": "info",
         "description": f"Get an overview of the {ctx}. Returns counts of documents, tables, images, topics.",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": f"{prefix}_search", "displayName": f"{prefix.upper()} Search", "icon": "search",
         "description": f"Search the {ctx} across documents, tables, and images. "
                        "Supports bilingual queries. Use this FIRST, then use batch_fetch.",
         "inputSchema": {"type": "object", "properties": {
             "query": {"type": "string", "description": "Search term"},
             "type": {"type": "string", "enum": ["document", "table", "row", "image"],
                      "description": "Limit to content type (optional)"},
             "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
         }, "required": ["query"]}},
        {"name": f"{prefix}_batch_fetch", "displayName": f"{prefix.upper()} Batch Fetch", "icon": "download",
         "description": f"Fetch multiple documents/tables from the {ctx} in a SINGLE call.",
         "inputSchema": {"type": "object", "properties": {
             "documents": {"type": "array", "items": {"type": "object", "properties": {
                 "identifier": {"type": "string"}, "section": {"type": "string"}}, "required": ["identifier"]}},
             "tables": {"type": "array", "items": {"type": "object", "properties": {
                 "identifier": {"type": "string"},
                 "filters": {"type": "object", "additionalProperties": {"type": "string"}},
                 "columns": {"type": "array", "items": {"type": "string"}},
                 "limit": {"type": "integer"}}, "required": ["identifier"]}},
         }}},
        {"name": f"{prefix}_get_document", "displayName": f"{prefix.upper()} Get Document", "icon": "article",
         "description": "Get a document by filename or title. Can retrieve a specific section.",
         "inputSchema": {"type": "object", "properties": {
             "identifier": {"type": "string", "description": "Filename or title (partial match)"},
             "section": {"type": "string", "description": "Specific section name"},
         }, "required": ["identifier"]}},
        {"name": f"{prefix}_get_table", "displayName": f"{prefix.upper()} Get Table", "icon": "table_chart",
         "description": "Query a table with filters. ALWAYS use filters.",
         "inputSchema": {"type": "object", "properties": {
             "identifier": {"type": "string"}, "info_only": {"type": "boolean", "default": False},
             "filters": {"type": "object", "additionalProperties": {"type": "string"}},
             "columns": {"type": "array", "items": {"type": "string"}},
             "limit": {"type": "integer", "default": 20},
         }, "required": ["identifier"]}},
        {"name": f"{prefix}_get_image", "displayName": f"{prefix.upper()} Get Image", "icon": "image",
         "description": "Get image information and URL for embedding.",
         "inputSchema": {"type": "object", "properties": {
             "query": {"type": "string"}, "image_type": {"type": "string"},
             "language": {"type": "string", "enum": ["en", "de"]},
         }, "required": ["query"]}},
        {"name": f"{prefix}_list_content", "displayName": f"{prefix.upper()} List Content", "icon": "list",
         "description": f"List documents, tables, or images in the {ctx}.",
         "inputSchema": {"type": "object", "properties": {
             "content_type": {"type": "string", "enum": ["documents", "tables", "images"], "default": "documents"},
             "limit": {"type": "integer", "default": 50},
         }}},
    ]


def _extract_sources(result: Any, media_prefix: str) -> list[dict]:
    """Extract source attribution entries from a tool result.

    Scans the result dict/list for ``source_pdf``, ``page``, ``filename``,
    ``columns``, ``rows`` fields and returns a compact list for the UI footer.
    """
    sources: list[dict] = []
    if isinstance(result, dict) and "error" in result:
        return sources

    def _add_doc(item: dict) -> None:
        pdf = item.get("source_pdf")
        if not pdf:
            return
        sources.append({
            "type": "doc",
            "title": item.get("title") or item.get("filename") or "",
            "page": item.get("page"),
            "pdf": pdf,
            "media_prefix": media_prefix,
        })

    def _add_table(item: dict) -> None:
        fn = item.get("filename") or item.get("table") or ""
        cols = item.get("columns")
        rows = item.get("rows")
        if not fn or not cols:
            return
        # Limit preview rows for storage
        preview_rows = rows[:50] if isinstance(rows, list) else None
        # Fallback: build preview rows from sample_values (info_only responses)
        if not preview_rows and isinstance(item.get("sample_values"), dict):
            sv = item["sample_values"]
            # sample_values is {col: [val1, val2, ...]} — transpose to rows
            max_samples = max((len(v) for v in sv.values() if isinstance(v, list)), default=0)
            if max_samples:
                preview_rows = []
                for i in range(min(max_samples, 5)):
                    row = {}
                    for col in cols:
                        vals = sv.get(col, [])
                        row[col] = vals[i] if i < len(vals) else ""
                    preview_rows.append(row)
        sources.append({
            "type": "table",
            "title": item.get("title") or fn,
            "filename": fn,
            "columns": cols,
            "total_rows": item.get("total_rows") or item.get("row_count") or item.get("returned_rows") or (len(rows) if isinstance(rows, list) else 0),
            "preview_rows": (preview_rows[:5] if preview_rows else None),
            "rows": preview_rows,
        })

    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                _add_doc(item)
                _add_table(item)
    elif isinstance(result, dict):
        # Single doc/table result
        _add_doc(result)
        _add_table(result)
        # Batch fetch: {documents: [...], tables: [...]}
        docs = result.get("documents", [])
        if isinstance(docs, list):
            for doc in docs:
                if isinstance(doc, dict):
                    _add_doc(doc)
        tbls = result.get("tables", [])
        if isinstance(tbls, list):
            for tbl in tbls:
                if isinstance(tbl, dict):
                    _add_table(tbl)

    return sources


# ---------------------------------------------------------------------------
# Source footer renderer (JS + CSS) — injected via get_client_renderers()
# ---------------------------------------------------------------------------

_FLUX_SOURCE_FOOTER_CSS = """\
.flux-source-footer{display:flex;flex-wrap:wrap;align-items:baseline;gap:3px 6px;
  margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,.05)}
.flux-source-footer-label{font-size:10px;color:var(--hub-text-muted,#6b7280);
  font-weight:500;letter-spacing:.4px;text-transform:uppercase;margin-right:1px;
  flex-shrink:0}
.flux-src-chip{display:inline-flex;align-items:center;gap:3px;
  padding:1px 7px 1px 5px;border-radius:6px;font-size:11px;font-weight:400;
  background:transparent;
  border:1px solid rgba(255,255,255,.06);
  color:var(--hub-text-muted,#8890a0);cursor:pointer;
  transition:background .15s,border-color .15s,color .15s;position:relative;
  text-decoration:none!important;margin:0;
  max-width:calc(50% - 16px);min-width:0}
.flux-src-chip:hover{background:var(--hub-hover-bg,rgba(60,63,90,.5));
  border-color:rgba(255,255,255,.15);color:var(--hub-text,#e0e0e0)}
.flux-src-chip .flux-src-icon{font-size:12px;opacity:.5;flex-shrink:0}
.flux-src-chip .flux-src-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.flux-src-popup{position:fixed;z-index:9999;
  background:var(--hub-overlay-bg,rgba(26,28,46,.92));
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  border:1px solid var(--hub-overlay-border,rgba(255,255,255,.10));
  border-radius:var(--hub-radius,14px);
  box-shadow:var(--hub-overlay-shadow,0 8px 32px rgba(0,0,0,.45));
  overflow:hidden;pointer-events:auto;animation:flux-popup-in .18s ease-out;
  padding:16px}
@keyframes flux-popup-in{from{opacity:0;transform:translateY(4px) scale(.98)}to{opacity:1;transform:none}}
.flux-src-popup-doc{width:580px;max-width:90vw}
.flux-src-popup-doc iframe{width:100%;aspect-ratio:1/1.414;max-height:70vh;
  border:none;border-radius:8px;background:#1a1c2e}
.flux-src-popup-doc .flux-src-popup-title{font-size:14px;font-weight:600;
  color:var(--hub-text,#e0e0e0);margin-bottom:8px;display:flex;align-items:center;gap:8px}
.flux-src-popup-doc .flux-src-popup-meta{font-size:11px;color:var(--hub-text-muted,#8890a0);
  margin-top:8px;display:flex;align-items:center;justify-content:space-between}
.flux-src-popup-doc .flux-src-open-full{font-size:12px;color:var(--hub-accent,#60a5fa);
  cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;gap:3px}
.flux-src-popup-doc .flux-src-open-full:hover{text-decoration:underline}
.flux-src-popup-table{width:640px;max-width:90vw}
.flux-src-popup-table .flux-src-popup-title{font-size:14px;font-weight:600;
  color:var(--hub-text,#e0e0e0);margin-bottom:8px}
.flux-src-popup-table table{width:100%;border-collapse:collapse;font-size:12.5px}
.flux-src-popup-table th{text-align:left;padding:5px 10px;font-weight:600;
  color:var(--hub-text,#e0e0e0);border-bottom:1px solid rgba(255,255,255,.12)}
.flux-src-popup-table td{padding:5px 10px;color:var(--hub-text-muted,#8890a0);
  border-bottom:1px solid rgba(255,255,255,.05)}
.flux-src-popup-table .flux-src-popup-meta{font-size:11px;color:var(--hub-text-muted,#8890a0);margin-top:8px}
"""

_FLUX_SOURCE_FOOTER_JS = r"""
(function(registry) {
  var _popup = null, _timer = null;

  function _esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  function hidePopup() {
    clearTimeout(_timer);
    if (_popup) { _popup.remove(); _popup = null; }
  }

  function positionPopup(chip, p) {
    var rect = chip.getBoundingClientRect();
    var pw = p.offsetWidth || 420, ph = p.offsetHeight || 200;
    var vw = window.innerWidth, vh = window.innerHeight, gap = 8;
    var left = rect.left; if (left + pw > vw - gap) left = vw - pw - gap; if (left < gap) left = gap;
    var top = rect.bottom + 6;
    if (top + ph > vh - gap) top = rect.top - ph - 6; if (top < gap) top = gap;
    p.style.left = left + 'px'; p.style.top = top + 'px';
  }

  function _getPdfUrl(src) {
    return '/api/' + (src.media_prefix || 'kb-media') + '/pdf/' + src.pdf;
  }

  function showDocPopup(chip, src) {
    hidePopup();
    var p = document.createElement('div');
    p.className = 'flux-src-popup flux-src-popup-doc';
    var pdfUrl = _getPdfUrl(src);
    var html = '<div class="flux-src-popup-title">' +
      '<span class="material-icons" style="font-size:16px;opacity:.6">description</span> ' +
      _esc(src.title || 'Document') + '</div>';
    html += '<iframe src="' + _esc(pdfUrl + '#page=' + (src.page || 1) + '&toolbar=0&navpanes=0&view=FitH') + '"></iframe>';
    html += '<div class="flux-src-popup-meta"><span>';
    var meta = [];
    if (src.page) meta.push('Page ' + src.page);
    if (src.pdf) meta.push(src.pdf);
    html += _esc(meta.join(' \u2022 '));
    html += '</span><span class="flux-src-open-full" data-action="open-full">' +
      '<span class="material-icons" style="font-size:14px">open_in_full</span> Open full viewer</span></div>';
    p.innerHTML = html;
    p.querySelector('[data-action="open-full"]').addEventListener('click', function(e) {
      e.stopPropagation();
      hidePopup();
      _openPdf(src);
    });
    document.body.appendChild(p);
    _popup = p;
    positionPopup(chip, p);
    p.addEventListener('mouseenter', function() { clearTimeout(_timer); });
    p.addEventListener('mouseleave', hidePopup);
  }

  function showTablePopup(chip, src) {
    hidePopup();
    var p = document.createElement('div');
    p.className = 'flux-src-popup flux-src-popup-table';
    var html = '<div class="flux-src-popup-title">' +
      '<span class="material-icons" style="font-size:16px;opacity:.6">table_chart</span> ' +
      _esc(src.title || 'Table') + '</div>';
    var cols = (src.columns || []).slice(0, 6);
    var rows = (src.preview_rows || []).slice(0, 5);
    if (cols.length) {
      html += '<table><thead><tr>';
      for (var c = 0; c < cols.length; c++) html += '<th>' + _esc(String(cols[c])) + '</th>';
      html += '</tr></thead><tbody>';
      for (var r = 0; r < rows.length; r++) {
        html += '<tr>';
        if (Array.isArray(rows[r])) {
          for (var ci = 0; ci < cols.length; ci++) html += '<td>' + _esc(String(rows[r][ci] != null ? rows[r][ci] : '')) + '</td>';
        } else if (typeof rows[r] === 'object') {
          for (var ci2 = 0; ci2 < cols.length; ci2++) html += '<td>' + _esc(String(rows[r][cols[ci2]] != null ? rows[r][cols[ci2]] : '')) + '</td>';
        }
        html += '</tr>';
      }
      html += '</tbody></table>';
    }
    var total = src.total_rows || 0;
    if (total) html += '<div class="flux-src-popup-meta">' + total + ' rows total</div>';
    p.innerHTML = html;
    document.body.appendChild(p);
    _popup = p;
    positionPopup(chip, p);
    p.addEventListener('mouseenter', function() { clearTimeout(_timer); });
    p.addEventListener('mouseleave', hidePopup);
  }

  function openFullTable(src) {
    // Use tablePlugin modal if available, else simple overlay
    if (registry._openTableViewer) {
      registry._openTableViewer(src);
      return;
    }
    // Fallback: full-screen modal
    var overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.7);display:flex;align-items:center;justify-content:center;padding:24px';
    overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
    var card = document.createElement('div');
    card.style.cssText = 'background:var(--hub-overlay-bg,#1a1c2e);border-radius:14px;padding:20px;max-width:90vw;max-height:80vh;overflow:auto;color:var(--hub-text,#e0e0e0)';
    var cols = src.columns || [];
    var rows = src.rows || [];
    var html = '<h3 style="margin:0 0 12px">' + _esc(src.title || 'Table') + '</h3><table style="width:100%;border-collapse:collapse;font-size:13px"><thead><tr>';
    for (var c = 0; c < cols.length; c++) html += '<th style="text-align:left;padding:6px 10px;border-bottom:1px solid rgba(255,255,255,.15);font-weight:600">' + _esc(String(cols[c])) + '</th>';
    html += '</tr></thead><tbody>';
    for (var r = 0; r < rows.length; r++) {
      html += '<tr>';
      if (Array.isArray(rows[r])) {
        for (var ci = 0; ci < cols.length; ci++) html += '<td style="padding:4px 10px;border-bottom:1px solid rgba(255,255,255,.05)">' + _esc(String(rows[r][ci] != null ? rows[r][ci] : '')) + '</td>';
      } else if (typeof rows[r] === 'object') {
        for (var ci2 = 0; ci2 < cols.length; ci2++) html += '<td style="padding:4px 10px;border-bottom:1px solid rgba(255,255,255,.05)">' + _esc(String(rows[r][cols[ci2]] != null ? rows[r][cols[ci2]] : '')) + '</td>';
      }
      html += '</tr>';
    }
    html += '</tbody></table>';
    var total = src.total_rows || 0;
    if (total) html += '<div style="margin-top:8px;font-size:12px;color:var(--hub-text-muted,#8890a0)">' + total + ' rows total</div>';
    card.innerHTML = html;
    // Close button
    var close = document.createElement('button');
    close.style.cssText = 'position:absolute;top:12px;right:12px;background:none;border:none;color:var(--hub-text,#e0e0e0);cursor:pointer;font-size:20px';
    close.innerHTML = '<span class="material-icons">close</span>';
    close.addEventListener('click', function() { overlay.remove(); });
    card.style.position = 'relative';
    card.appendChild(close);
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    var onKey = function(e) { if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', onKey); } };
    document.addEventListener('keydown', onKey);
  }

  function _openPdf(src) {
    var url = _getPdfUrl(src);
    var page = src.page || 1;
    // Use the ChatApp PDF viewer (same as file attachments — full-screen, maximize, new tab)
    if (window.__chatApp && window.__chatApp._openPdfViewer) {
      window.__chatApp._openPdfViewer(url, page);
    } else if (registry && registry._openPdfViewer) {
      registry._openPdfViewer(url, page);
    } else {
      window.open(url + '#page=' + page, '_blank');
    }
  }

  function buildFooter(sources) {
    var bar = document.createElement('div');
    bar.className = 'flux-source-footer';

    // Deduplicate
    var seen = {};
    var deduped = [];
    for (var i = 0; i < sources.length; i++) {
      var s = sources[i];
      var key = s.type === 'doc' ? ('doc:' + s.pdf + ':' + (s.page || '')) : ('tbl:' + s.filename);
      if (seen[key]) continue;
      seen[key] = true;
      deduped.push(s);
    }
    if (!deduped.length) return null;

    var label = document.createElement('span');
    label.className = 'flux-source-footer-label';
    label.textContent = 'Sources';
    bar.appendChild(label);

    for (var j = 0; j < deduped.length; j++) {
      (function(src) {
        var chip = document.createElement('span');
        chip.className = 'flux-src-chip';
        if (src.type === 'doc') {
          chip.innerHTML = '<span class="material-icons flux-src-icon">description</span>' +
            '<span class="flux-src-label">' + _esc(src.title || 'Document') + (src.page ? ', p.' + _esc(String(src.page)) : '') + '</span>';
          chip.addEventListener('mouseenter', function() { clearTimeout(_timer); _timer = setTimeout(function() { showDocPopup(chip, src); }, 300); });
          chip.addEventListener('mouseleave', function() { clearTimeout(_timer); _timer = setTimeout(hidePopup, 200); });
          chip.addEventListener('click', function(e) {
            e.preventDefault(); hidePopup();
            _openPdf(src);
          });
        } else {
          chip.innerHTML = '<span class="material-icons flux-src-icon">table_chart</span>' +
            '<span class="flux-src-label">' + _esc(src.title || 'Table') + '</span>';
          chip.addEventListener('mouseenter', function() { clearTimeout(_timer); _timer = setTimeout(function() { showTablePopup(chip, src); }, 300); });
          chip.addEventListener('mouseleave', function() { clearTimeout(_timer); _timer = setTimeout(hidePopup, 200); });
          chip.addEventListener('click', function(e) {
            e.preventDefault(); hidePopup();
            openFullTable(src);
          });
        }
        bar.appendChild(chip);
      })(deduped[j]);
    }

    return bar;
  }

  // Inline renderer: reads data-flux-sources from the parent .cv2-msg-assistant
  registry.registerInline(async function(container) {
    var msgEl = container.closest('.cv2-msg-assistant');
    if (!msgEl) return;
    var raw = msgEl.getAttribute('data-flux-sources');
    if (!raw) return;
    // Only render once
    if (container.querySelector('.flux-source-footer')) return;
    try {
      var sources = JSON.parse(raw);
      if (!sources || !sources.length) return;
      var footer = buildFooter(sources);
      if (footer) container.appendChild(footer);
    } catch(e) {
      console.error('[FluxSources] Failed to render footer:', e);
    }
  });

})(registry);
"""


class DocumentKBServer(InProcessMCPServer):
    """Base class for document-knowledge MCP servers.

    Subclasses must set ``prefix`` and override at least ``get_prompt_hints()``.
    Override ``load_metadata()`` for JSON-sidecar enrichment.
    """

    prefix: str = "kb"  # tool name prefix — override in subclass
    description_context: str = "knowledge base"
    media_url_prefix: str = "kb-media"  # /api/{media_url_prefix}/image/...

    def __init__(self, data_dir: Path, db: Database | None = None):
        self._data_dir = data_dir
        self._db = db
        self._image_hints: dict | None = None
        self._tools: list[dict] | None = None
        self._blob_checked = False
        self._blob_ready = True  # Assumed ready until proven otherwise

    async def _ensure_blob_data(self) -> None:
        """Pull data from blob storage if local DB is missing.

        Runs the download in a background task so it never blocks the event
        loop. Returns immediately — ``call_tool`` checks ``_blob_ready``
        and returns a user-friendly message while data is loading.
        """
        if self._blob_checked:
            return
        self._blob_checked = True
        if any(self._data_dir.glob("*.db")):
            self._blob_ready = True
            return  # Local data exists
        # Start download in background — don't block the event loop
        self._blob_ready = False
        import asyncio
        asyncio.create_task(self._background_blob_pull())

    async def _background_blob_pull(self) -> None:
        """Download data from blob storage in the background."""
        try:
            from .blob_sync import ensure_data
            logger.info("[%s] Starting background data download...", self.prefix.upper())
            await ensure_data(self.prefix, self._data_dir)
            self._blob_ready = True
            logger.info("[%s] Background data download complete.", self.prefix.upper())
        except Exception as e:
            logger.warning("[%s] Blob sync failed: %s", self.prefix.upper(), e)
            self._blob_ready = True  # Allow fallback to error in _get_db

    def _get_db(self) -> Database:
        if self._db is None:
            db_path = self._data_dir / "lisa.db"
            # Try generic name first
            if not db_path.exists():
                db_path = self._data_dir / f"{self.prefix}.db"
            if not db_path.exists():
                # Find any .db file
                dbs = list(self._data_dir.glob("*.db"))
                if dbs:
                    db_path = dbs[0]
                else:
                    raise RuntimeError(f"No database found in {self._data_dir}")
            self._db = Database(db_path, self._create_query_expander())
        return self._db

    def _create_query_expander(self):
        """Override to return a domain-specific QueryExpander."""
        from .query_expander import QueryExpander
        return QueryExpander()

    def _get_image_hints(self) -> dict:
        if self._image_hints is None:
            hints_file = self._data_dir / "image_hints.json"
            self._image_hints = {}
            if hints_file.exists():
                try:
                    import json
                    self._image_hints = json.loads(hints_file.read_text())
                except Exception:
                    pass
        return self._image_hints

    def load_metadata(self, filename: str) -> dict:
        """Load JSON sidecar metadata for a file. Override in subclasses."""
        return {}

    def _get_tools(self) -> list[dict]:
        if self._tools is None:
            self._tools = make_tool_definitions(self.prefix, self.description_context)
        return self._tools

    async def list_tools(self) -> list[dict]:
        return self._get_tools()

    def _rewrite_image_urls(self, content: str) -> str:
        """Replace raw image paths in document content with media URLs.

        Transforms ``![alt](images/048_idk_product.png)`` →
        ``![alt](/api/{media_url_prefix}/image/products/048_idk_product.png)``
        by looking up each image in the DB to get the correct subdirectory.
        """
        import re
        media = self.media_url_prefix
        def _replace(m):
            alt, raw_path = m.group(1), m.group(2)
            fname = raw_path.rsplit("/", 1)[-1]
            db = self._get_db()
            img = db.get_image(fname.rsplit(".", 1)[0])
            if img and img.local_path:
                return f"![{alt}](/api/{media}/image/{img.local_path})"
            return m.group(0)
        return re.sub(r'!\[([^\]]*)\]\((images/[^)]+)\)', _replace, content)

    # Tools that actually fetch content (not search/overview/list which return pointers)
    _SOURCE_TOOLS = {"get_document", "get_table", "batch_fetch"}

    async def call_tool(self, name: str, arguments: dict) -> str:
        try:
            await self._ensure_blob_data()
            if not getattr(self, '_blob_ready', True):
                return json.dumps({
                    "status": "loading",
                    "message": f"Knowledge base is being prepared (downloading data). Please try again in a moment.",
                })
            db = self._get_db()
            result = self._dispatch(db, name, arguments)
            result_str = json.dumps(result, indent=2, default=str)
            # Only extract sources from tools that fetch actual content
            suffix = name.removeprefix(self.prefix + "_")
            if suffix in self._SOURCE_TOOLS:
                sources = _extract_sources(result, self.media_url_prefix)
                if sources:
                    result_str += "\n\n__FLUX_SOURCES__\n" + json.dumps(sources, default=str)
            return result_str
        except Exception as e:
            logger.exception(f"[{self.prefix.upper()}] Tool error: {name}")
            return json.dumps({"error": str(e)})

    def _dispatch(self, db: Database, name: str, args: dict) -> Any:
        p = self.prefix
        media = self.media_url_prefix

        if name == f"{p}_overview":
            ov = db.get_knowledge_overview()
            return {"documents": ov.document_count, "tables": ov.table_count,
                    "table_rows": ov.total_table_rows, "images": ov.image_count,
                    "topics": ov.topics[:50], "tables_summary": ov.tables[:20],
                    "image_types": ov.image_types}

        elif name == f"{p}_search":
            query, limit = args["query"], args.get("limit", 20)
            seen: set[tuple] = set()
            combined: list[dict] = []
            for r in db.search(query, limit * 2, args.get("type")):
                fn = r.metadata.get("filename") or r.metadata.get("table") or ""
                key = (r.result_type, fn)
                if key in seen or not fn:
                    continue
                seen.add(key)
                meta = self.load_metadata(fn)
                entry: dict[str, Any] = {
                    "type": r.result_type, "filename": fn,
                    "title": meta.get("title", r.title),
                    "summary": meta.get("summary", r.snippet or ""),
                    "source_pdf": r.metadata.get("source_pdf"),
                    "page": r.metadata.get("page")}
                if r.result_type == "table":
                    tbl = db.get_table(fn)
                    if tbl:
                        entry.update(columns=tbl.headers, row_count=tbl.row_count,
                                     sample_values=tbl.sample_values)
                elif r.result_type == "table_row":
                    entry["matched_row"] = r.metadata.get("data", {})
                combined.append(entry)
            return combined[:limit]

        elif name == f"{p}_batch_fetch":
            batch: dict[str, list] = {"documents": [], "tables": []}
            for dr in args.get("documents", []):
                idf, sec = dr["identifier"], dr.get("section")
                if sec:
                    c = db.get_document_section(idf, sec)
                    if c:
                        batch["documents"].append({"filename": idf, "title": self.load_metadata(idf).get("title", ""),
                                                   "section": sec, "content": self._rewrite_image_urls(c)})
                    else:
                        batch["documents"].append({"filename": idf, "error": f"Section '{sec}' not found"})
                else:
                    doc = db.get_document(idf)
                    if doc:
                        meta = self.load_metadata(doc.filename)
                        batch["documents"].append({
                            "filename": doc.filename, "title": meta.get("title", doc.title),
                            "page": doc.page_number, "source_pdf": doc.source_pdf,
                            "sections": list(doc.sections.keys()), "content": self._rewrite_image_urls(doc.content)})
                    else:
                        batch["documents"].append({"filename": idf, "error": "Document not found"})
            for tr in args.get("tables", []):
                idf = tr["identifier"]
                qr = db.query_table(idf, tr.get("filters"), tr.get("columns"), tr.get("limit", 20))
                if qr:
                    meta = self.load_metadata(qr.table_name)
                    batch["tables"].append({
                        "filename": qr.table_name, "title": meta.get("title", ""),
                        "columns": qr.headers, "total_rows": qr.total_rows,
                        "returned_rows": qr.returned_rows,
                        "filters_applied": tr.get("filters") or {}, "rows": qr.rows})
                else:
                    batch["tables"].append({"filename": idf, "error": "Table not found"})
            return batch

        elif name == f"{p}_get_document":
            sec = args.get("section")
            if sec:
                c = db.get_document_section(args["identifier"], sec)
                return {"section": sec, "content": self._rewrite_image_urls(c)} if c else {"error": f"Section '{sec}' not found"}
            doc = db.get_document(args["identifier"])
            if not doc:
                return {"error": f"Document '{args['identifier']}' not found"}
            meta = self.load_metadata(doc.filename)
            return {"filename": doc.filename, "title": meta.get("title", doc.title),
                    "page": doc.page_number, "source_pdf": doc.source_pdf,
                    "tags": doc.tags, "sections": list(doc.sections.keys()), "content": self._rewrite_image_urls(doc.content)}

        elif name == f"{p}_get_table":
            if args.get("info_only"):
                tbl = db.get_table(args["identifier"])
                if not tbl:
                    return {"error": f"Table '{args['identifier']}' not found"}
                meta = self.load_metadata(tbl.filename)
                return {"filename": tbl.filename, "title": meta.get("title", tbl.title),
                        "columns": tbl.headers, "row_count": tbl.row_count, "sample_values": tbl.sample_values}
            qr = db.query_table(args["identifier"], args.get("filters"), args.get("columns"), args.get("limit", 20))
            if not qr:
                return {"error": f"Table '{args['identifier']}' not found"}
            meta = self.load_metadata(qr.table_name)
            return {"table": qr.table_name, "title": meta.get("title", ""),
                    "columns": qr.headers, "total_rows": qr.total_rows,
                    "returned_rows": qr.returned_rows, "rows": qr.rows}

        elif name == f"{p}_get_image":
            img = db.get_image(args["query"], args.get("image_type"), args.get("language"))
            if not img:
                return {"error": f"Image not found for '{args['query']}'"}
            hint = self._get_image_hints().get(img.local_path, "")
            return {"filename": img.filename, "type": img.image_type, "language": img.language,
                    "source": img.source_document,
                    "url": f"/api/{media}/image/{img.local_path}", "hint": hint}

        elif name == f"{p}_list_content":
            ct, lim = args.get("content_type", "documents"), args.get("limit", 50)
            if ct == "documents":
                return [{"filename": d.filename, "title": self.load_metadata(d.filename).get("title", d.title),
                         "page": d.page_number, "tags": d.tags, "source_pdf": d.source_pdf}
                        for d in db.list_documents(lim)]
            elif ct == "tables":
                return [{"filename": t.filename, "title": self.load_metadata(t.filename).get("title", t.title),
                         "columns": t.headers, "rows": t.row_count} for t in db.list_tables()[:lim]]
            elif ct == "images":
                return [{"filename": i.filename, "type": i.image_type, "language": i.language,
                         "url": f"/api/{media}/image/{i.local_path}"}
                        for i in db.search_images("", limit=lim)]
            return {"error": f"Unknown content type: {ct}"}

        return {"error": f"Unknown tool: {name}"}

    async def get_client_renderers(self) -> list[dict[str, str]]:
        return [
            {"type": "inline", "js": _FLUX_SOURCE_FOOTER_JS, "css": _FLUX_SOURCE_FOOTER_CSS},
        ]
