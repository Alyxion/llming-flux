"""Synthetic test data fixtures for document-knowledge databases."""

from pathlib import Path

from ..database import Database
from ..models import Document, Table, Image


def synthetic_db(tmp_path: Path, prefix: str = "test") -> Database:
    """Create a Database with synthetic sample data for unit testing.

    Returns a ready-to-query Database with FTS indices, containing:
    - 3 documents (product doc, reference doc, Q&A doc)
    - 1 table with 3 rows
    - 1 image
    - 1 pdf_page_map entry
    """
    db_path = tmp_path / f"{prefix}.db"
    db = Database(db_path)
    db.create_tables()

    # Documents
    db.insert_document(Document(
        filename="doc_alpha.md",
        title="Alpha Product Series",
        page_number=10,
        source="CATALOG2024",
        tags=["product", "alpha"],
        content="# Alpha Series\n\nThe Alpha series is designed for precision.\n\n## Advantages\n\n- High accuracy\n- Low drift\n- Durable materials",
        sections={"Advantages": "- High accuracy\n- Low drift\n- Durable materials"},
        source_pdf="catalog-2024-en.pdf",
    ))
    db.insert_document(Document(
        filename="ref_materials.md",
        title="Material Reference Guide",
        tags=["reference", "materials"],
        content="# Material Reference\n\nAvailable materials: ceramic, stainless steel, POM, brass.\n\n## Ceramic\n\nMost durable option.",
        sections={"Ceramic": "Most durable option."},
    ))
    db.insert_document(Document(
        filename="qa_faq.md",
        title="Frequently Asked Questions",
        tags=["Q&A", "faq"],
        content="# FAQ\n\nQ: How do I identify product sizes?\nA: By color coding per ISO standard.",
        sections={},
    ))

    # Table
    table_id = db.insert_table(Table(
        filename="alpha_data.xlsx",
        title="Alpha Series Data",
        headers=["Model", "Color", "Pressure (bar)", "Flow rate (l/min)"],
        row_count=3,
        sample_values={"Model": ["A-100", "A-200"], "Color": ["Red", "Blue"]},
    ))
    db.insert_table_row(table_id, 1, {"Model": "A-100", "Color": "Red", "Pressure (bar)": 2.0, "Flow rate (l/min)": 0.5})
    db.insert_table_row(table_id, 2, {"Model": "A-200", "Color": "Blue", "Pressure (bar)": 3.0, "Flow rate (l/min)": 1.0})
    db.insert_table_row(table_id, 3, {"Model": "A-300", "Color": "Green", "Pressure (bar)": 3.0, "Flow rate (l/min)": 1.5})

    # Image
    db.insert_image(Image(
        filename="alpha_product.png",
        source_document="Alpha",
        image_type="product",
        language="en",
        tags=["product"],
        local_path="products/alpha_product.png",
    ))

    # PDF page map
    db.insert_pdf_page_map("doc_alpha.md", 10, "catalog-2024-en.pdf")

    # FTS
    db.create_fts_index()
    return db
