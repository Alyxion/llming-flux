"""Database tests with synthetic data."""

import json
import pytest
from llming_flux.testing.fixtures import synthetic_db


@pytest.fixture
def db(tmp_path):
    return synthetic_db(tmp_path)


class TestInsertAndQuery:
    def test_knowledge_overview(self, db):
        ov = db.get_knowledge_overview()
        assert ov.document_count == 3
        assert ov.table_count == 1
        assert ov.image_count == 1
        assert ov.total_table_rows == 3

    def test_search_document(self, db):
        results = db.search("Alpha")
        assert len(results) > 0
        assert any(r.result_type == "document" for r in results)

    def test_search_table(self, db):
        results = db.search("Alpha", search_type="table")
        assert len(results) > 0

    def test_search_row(self, db):
        results = db.search("A-200", search_type="row")
        assert len(results) > 0

    def test_search_image(self, db):
        results = db.search("alpha", search_type="image")
        assert len(results) > 0

    def test_search_no_results(self, db):
        results = db.search("xyznonexistent12345")
        assert len(results) == 0

    def test_get_document_by_filename(self, db):
        doc = db.get_document("doc_alpha.md")
        assert doc is not None
        assert doc.title == "Alpha Product Series"
        assert doc.page_number == 10
        assert doc.source_pdf == "catalog-2024-en.pdf"

    def test_get_document_partial_match(self, db):
        doc = db.get_document("alpha")
        assert doc is not None

    def test_get_document_not_found(self, db):
        assert db.get_document("nonexistent.md") is None

    def test_get_document_section(self, db):
        content = db.get_document_section("doc_alpha.md", "Advantages")
        assert content is not None
        assert "accuracy" in content

    def test_get_document_section_case_insensitive(self, db):
        content = db.get_document_section("doc_alpha.md", "advantages")
        assert content is not None

    def test_list_documents(self, db):
        docs = db.list_documents()
        assert len(docs) == 3

    def test_get_table(self, db):
        tbl = db.get_table("alpha_data.xlsx")
        assert tbl is not None
        assert tbl.row_count == 3

    def test_query_table_no_filter(self, db):
        result = db.query_table("alpha_data.xlsx")
        assert result is not None
        assert result.returned_rows == 3

    def test_query_table_with_filter(self, db):
        result = db.query_table("alpha_data.xlsx", filters={"Color": "Blue"})
        assert result is not None
        assert result.returned_rows == 1
        assert result.rows[0]["Model"] == "A-200"

    def test_query_table_with_columns(self, db):
        result = db.query_table("alpha_data.xlsx", columns=["Model", "Flow rate (l/min)"])
        assert result is not None
        assert "Color" not in result.rows[0]

    def test_query_table_with_limit(self, db):
        result = db.query_table("alpha_data.xlsx", limit=1)
        assert result.returned_rows == 1

    def test_list_tables(self, db):
        tables = db.list_tables()
        assert len(tables) == 1

    def test_get_image(self, db):
        img = db.get_image("alpha", image_type="product")
        assert img is not None
        assert img.local_path == "products/alpha_product.png"

    def test_get_image_not_found(self, db):
        assert db.get_image("nonexistent") is None

    def test_search_images(self, db):
        imgs = db.search_images("alpha")
        assert len(imgs) == 1

    def test_pdf_page_map(self, db):
        pdf, page = db.get_pdf_for_page("doc_alpha.md")
        assert pdf == "catalog-2024-en.pdf"
        assert page == 10

    def test_pdf_page_map_not_found(self, db):
        pdf, page = db.get_pdf_for_page("nonexistent.md")
        assert pdf is None


class TestFTSEdgeCases:
    def test_search_with_special_chars(self, db):
        # Should not crash on FTS special characters
        results = db.search("alpha (test)")
        assert isinstance(results, list)

    def test_search_empty_query(self, db):
        results = db.search("")
        # Empty query after cleaning should return nothing
        assert isinstance(results, list)

    def test_search_stopwords_only(self, db):
        results = db.search("the and for")
        assert isinstance(results, list)
