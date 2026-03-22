"""Generic markdown document parser.

Extracts @@page, @@source, @@product, @@tags markers, ## sections,
and image references from markdown files.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ImageRef:
    """Reference to an image in a markdown file."""
    alt_text: str
    path: str
    image_type: str


@dataclass
class ParsedDocument:
    """A complete parsed markdown document."""
    filename: str
    page_number: int | None = None
    source_ref: str | None = None
    product_name: str | None = None
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    content: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    images: list[ImageRef] = field(default_factory=list)
    subdirectory: str = ""


def classify_image_type_default(alt_text: str, path: str) -> str:
    """Default image classifier — override for domain-specific types."""
    al = alt_text.lower()
    pl = path.lower()
    if "product" in al or "_product" in pl:
        return "product"
    if "dimension" in al or "_dimension" in pl:
        return "dimensions"
    if "table" in al or "_table" in pl:
        return "table"
    if "logo" in al or "_logo" in pl:
        return "logo"
    return "other"


def parse_markdown_file(
    filepath: Path,
    subdirectory: str = "",
    image_classifier: callable = None,
) -> ParsedDocument | None:
    """Parse a single markdown file and extract all content.

    Args:
        filepath: Path to the .md file.
        subdirectory: Label for which subdirectory this came from.
        image_classifier: ``(alt_text, path) -> str`` to classify images.
    """
    classify = image_classifier or classify_image_type_default
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    doc = ParsedDocument(filename=filepath.name, content=content, subdirectory=subdirectory)

    m = re.search(r"@@page\s+(\d+)", content)
    if m:
        doc.page_number = int(m.group(1))

    m = re.search(r"@@source\s+(.+?)(?:\n|$)", content)
    if m:
        doc.source_ref = m.group(1).strip()

    m = re.search(r"@@product\s+(.+?)(?:\n|$)", content)
    if m:
        doc.product_name = m.group(1).strip()

    for m in re.finditer(r"@@tags\s+(.+?)(?:\n|$)", content):
        doc.tags.extend(t.strip() for t in m.group(1).split(";") if t.strip())

    m = re.search(r"@@sub\s+(.+?)(?:\n|$)", content)
    if m:
        doc.tags.extend(s.strip() for s in m.group(1).split(";") if s.strip())

    m = re.search(r"^#\s+(.+?)(?:\n|$)", content, re.MULTILINE)
    if m:
        title = re.sub(r"<br\s*/?>", " ", m.group(1).strip())
        doc.title = title.strip()

    doc.sections = extract_all_sections(content)

    for m in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", content):
        doc.images.append(ImageRef(
            alt_text=m.group(1), path=m.group(2),
            image_type=classify(m.group(1), m.group(2))))

    return doc


def extract_all_sections(content: str) -> dict[str, str]:
    """Extract all ``## heading`` sections from markdown."""
    sections = {}
    for heading, body in re.findall(
        r"^##\s+(.+?)(?:\n|$)(.*?)(?=^##|\Z)", content, re.MULTILINE | re.DOTALL
    ):
        sections[heading.strip()] = body.strip()
    return sections
