"""Tests for the base DocumentKBServer and standalone server factory."""

import asyncio
import json
import pytest
from pathlib import Path

from llming_flux.testing.fixtures import synthetic_db
from llming_flux.inprocess_server import DocumentKBServer, make_tool_definitions, parse_tool_result


class TestMakeToolDefinitions:
    def test_prefix(self):
        tools = make_tool_definitions("mybot")
        names = [t["name"] for t in tools]
        assert "mybot_search" in names
        assert "mybot_batch_fetch" in names
        assert "mybot_get_document" in names
        assert "mybot_get_table" in names
        assert "mybot_get_image" in names
        assert "mybot_list_content" in names
        assert "mybot_overview" in names

    def test_custom_context(self):
        tools = make_tool_definitions("x", "industrial product catalog")
        assert "industrial product catalog" in tools[0]["description"]

    def test_schemas_valid(self):
        tools = make_tool_definitions("t")
        for t in tools:
            assert "inputSchema" in t
            assert t["inputSchema"]["type"] == "object"


class TestDocumentKBServer:
    @pytest.fixture
    def server(self, tmp_path):
        db = synthetic_db(tmp_path)

        class TestServer(DocumentKBServer):
            prefix = "test"
            description_context = "test knowledge base"
            media_url_prefix = "test-media"

        srv = TestServer(data_dir=tmp_path, db=db)
        return srv

    def test_list_tools(self, server):
        tools = asyncio.run(server.list_tools())
        names = [t["name"] for t in tools]
        assert "test_search" in names
        assert "test_batch_fetch" in names

    def test_search(self, server):
        result = asyncio.run(server.call_tool("test_search", {"query": "Alpha"}))
        data = json.loads(parse_tool_result(result)[0])
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_document(self, server):
        result = asyncio.run(server.call_tool("test_get_document", {"identifier": "doc_alpha.md"}))
        data = json.loads(parse_tool_result(result)[0])
        assert data["filename"] == "doc_alpha.md"
        assert data["source_pdf"] == "catalog-2024-en.pdf"

    def test_get_document_section(self, server):
        result = asyncio.run(server.call_tool("test_get_document",
                                              {"identifier": "doc_alpha.md", "section": "Advantages"}))
        data = json.loads(parse_tool_result(result)[0])
        assert "accuracy" in data["content"]

    def test_get_table(self, server):
        result = asyncio.run(server.call_tool("test_get_table",
                                              {"identifier": "alpha_data.xlsx", "filters": {"Color": "Blue"}}))
        data = json.loads(parse_tool_result(result)[0])
        assert data["returned_rows"] == 1

    def test_get_image(self, server):
        result = asyncio.run(server.call_tool("test_get_image", {"query": "alpha"}))
        data = json.loads(parse_tool_result(result)[0])
        assert "/api/test-media/image/" in data["url"]

    def test_list_content_documents(self, server):
        result = asyncio.run(server.call_tool("test_list_content", {"content_type": "documents"}))
        data = json.loads(parse_tool_result(result)[0])
        assert len(data) == 3

    def test_list_content_tables(self, server):
        result = asyncio.run(server.call_tool("test_list_content", {"content_type": "tables"}))
        data = json.loads(parse_tool_result(result)[0])
        assert len(data) == 1

    def test_overview(self, server):
        result = asyncio.run(server.call_tool("test_overview", {}))
        data = json.loads(parse_tool_result(result)[0])
        assert data["documents"] == 3

    def test_batch_fetch(self, server):
        result = asyncio.run(server.call_tool("test_batch_fetch", {
            "documents": [{"identifier": "doc_alpha.md"}],
            "tables": [{"identifier": "alpha_data.xlsx", "filters": {"Color": "Red"}}],
        }))
        data = json.loads(parse_tool_result(result)[0])
        assert len(data["documents"]) == 1
        assert len(data["tables"]) == 1
        assert data["documents"][0]["filename"] == "doc_alpha.md"
        assert data["tables"][0]["returned_rows"] == 1

    def test_unknown_tool(self, server):
        result = asyncio.run(server.call_tool("test_nonexistent", {}))
        data = json.loads(parse_tool_result(result)[0])
        assert "error" in data

    def test_custom_query_expander(self, tmp_path):
        """Verify pluggable query expander works."""
        from llming_flux.query_expander import QueryExpander

        class SynonymExpander(QueryExpander):
            def expand(self, query):
                terms = super().expand(query)
                if "alpha" in terms:
                    terms.append("bravo")  # synonym
                return terms

        db = synthetic_db(tmp_path, prefix="syn")

        class SynServer(DocumentKBServer):
            prefix = "syn"
            def _create_query_expander(self):
                return SynonymExpander()

        srv = SynServer(data_dir=tmp_path, db=db)
        # "bravo" should also find Alpha docs because of the synonym expansion
        result = asyncio.run(srv.call_tool("syn_search", {"query": "alpha"}))
        data = json.loads(parse_tool_result(result)[0])
        assert len(data) > 0


class TestStandaloneServerFactory:
    def test_creates_server(self, tmp_path):
        from llming_flux.server import create_document_kb_server
        from llming_flux.inprocess_server import make_tool_definitions

        synthetic_db(tmp_path)
        tools = make_tool_definitions("test")
        server = create_document_kb_server("test-server", "test", tmp_path, tools)
        assert server.name == "test-server"

    def test_handlers_registered(self, tmp_path):
        from llming_flux.server import create_document_kb_server
        from llming_flux.inprocess_server import make_tool_definitions
        from mcp.types import (
            ListToolsRequest, CallToolRequest,
            ListResourceTemplatesRequest, ListResourcesRequest,
            ReadResourceRequest,
        )

        synthetic_db(tmp_path)
        server = create_document_kb_server("t", "t", tmp_path, make_tool_definitions("t"))
        registered = set(server.request_handlers.keys())
        assert ListToolsRequest in registered
        assert CallToolRequest in registered
        assert ListResourceTemplatesRequest in registered
        assert ListResourcesRequest in registered
        assert ReadResourceRequest in registered
