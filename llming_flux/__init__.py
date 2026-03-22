"""llming-flux — Base library for building document-knowledge MCP agents.

MIT Licensed.
"""

__version__ = "0.1.0"

from llming_flux.database import Database
from llming_flux.query_expander import QueryExpander, build_fts_query, clean_word
from llming_flux.models import Document, Table, TableRow, Image, KnowledgeOverview, SearchResult, TableQueryResult
from llming_flux.inprocess_server import DocumentKBServer, make_tool_definitions
from llming_flux.media_router import build_media_router, guess_mime
