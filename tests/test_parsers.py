"""Parser tests with synthetic markdown and xlsx files."""

import pytest
from pathlib import Path

from llming_flux.parsers.markdown import parse_markdown_file, extract_all_sections, ParsedDocument
from llming_flux.parsers.xlsx import parse_xlsx_file, derive_title, ParsedTable


class TestMarkdownParser:
    @pytest.fixture
    def sample_md(self, tmp_path):
        content = """\
@@page 42
@@source CATALOG2024
@@product Alpha Series
@@tags search;document;retrieval

# Alpha Series Products

Introduction text about Alpha products.

![product](images/alpha_product.png)
![dimensions](images/alpha_dimensions_en.png)

## Advantages

- High precision
- Low drift

## Technical Data

* Size: 01-10
* Material: ceramic, POM

## Application

For plant protection.
"""
        p = tmp_path / "042_alpha_en.md"
        p.write_text(content)
        return p

    def test_parse_markers(self, sample_md):
        doc = parse_markdown_file(sample_md)
        assert doc is not None
        assert doc.page_number == 42
        assert doc.source_ref == "CATALOG2024"
        assert doc.product_name == "Alpha Series"

    def test_parse_tags(self, sample_md):
        doc = parse_markdown_file(sample_md)
        assert "search" in doc.tags
        assert "document" in doc.tags
        assert "drift" in doc.tags

    def test_parse_title(self, sample_md):
        doc = parse_markdown_file(sample_md)
        assert doc.title == "Alpha Series Products"

    def test_parse_sections(self, sample_md):
        doc = parse_markdown_file(sample_md)
        assert "Advantages" in doc.sections
        assert "Technical Data" in doc.sections
        assert "Application" in doc.sections
        assert "precision" in doc.sections["Advantages"]

    def test_parse_images(self, sample_md):
        doc = parse_markdown_file(sample_md)
        assert len(doc.images) == 2
        types = [img.image_type for img in doc.images]
        assert "product" in types
        assert "dimensions" in types

    def test_parse_subdirectory(self, sample_md):
        doc = parse_markdown_file(sample_md, subdirectory="TestDir")
        assert doc.subdirectory == "TestDir"

    def test_parse_nonexistent_file(self, tmp_path):
        result = parse_markdown_file(tmp_path / "nonexistent.md")
        assert result is None

    def test_custom_image_classifier(self, sample_md):
        def my_classifier(alt, path):
            if "pattern" in alt.lower() or "pattern" in path.lower():
                return "image_pattern"
            return "generic"

        doc = parse_markdown_file(sample_md, image_classifier=my_classifier)
        # Our test file doesn't have pattern images, so both should be "generic"
        assert all(img.image_type == "generic" for img in doc.images)


class TestExtractSections:
    def test_basic(self):
        content = "# Title\n\nIntro\n\n## Section A\n\nContent A\n\n## Section B\n\nContent B"
        sections = extract_all_sections(content)
        assert "Section A" in sections
        assert "Section B" in sections
        assert "Content A" in sections["Section A"]

    def test_no_sections(self):
        sections = extract_all_sections("# Just a title\n\nNo sections here.")
        assert len(sections) == 0


class TestXlsxParser:
    @pytest.fixture
    def sample_xlsx(self, tmp_path):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Model", "Color", "Pressure"])
        ws.append(["A-100", "Red", 2.0])
        ws.append(["A-200", "Blue", 3.0])
        ws.append(["A-300", "Green", 3.0])
        p = tmp_path / "042_alpha_table.xlsx"
        wb.save(p)
        return p

    def test_parse_headers(self, sample_xlsx):
        table = parse_xlsx_file(sample_xlsx)
        assert table is not None
        assert table.headers == ["Model", "Color", "Pressure"]

    def test_parse_rows(self, sample_xlsx):
        table = parse_xlsx_file(sample_xlsx)
        assert len(table.rows) == 3
        assert table.rows[0]["Model"] == "A-100"
        assert table.rows[1]["Color"] == "Blue"

    def test_parse_title(self, sample_xlsx):
        table = parse_xlsx_file(sample_xlsx)
        assert table.title == "Alpha Table"

    def test_empty_xlsx(self, tmp_path):
        import openpyxl
        wb = openpyxl.Workbook()
        p = tmp_path / "empty.xlsx"
        wb.save(p)
        assert parse_xlsx_file(p) is None


class TestDeriveTitle:
    def test_with_page_number(self):
        assert derive_title("042_alpha_table.xlsx") == "Alpha Table"

    def test_without_page_number(self):
        assert derive_title("alpha_data.xlsx") == "Alpha Data"
