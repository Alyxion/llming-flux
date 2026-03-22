"""Generate JSON sidecar files for documents, tables, images, and PDFs.

Uses ``claude -p`` (Claude Code CLI) — no API key needed.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def _call_claude(prompt: str, timeout: int = 90) -> str:
    """Call claude CLI and return response text."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"[Error: {result.stderr.strip()[:200]}]"
    except subprocess.TimeoutExpired:
        return "[Error: timeout]"
    except FileNotFoundError:
        return "[Error: claude CLI not found]"


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from claude response text."""
    # Try to find JSON in the response
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Document summarization ────────────────────────────────

def summarize_markdown(content: str, filename: str) -> dict:
    """Generate summary, products, and hashtags for a markdown file."""
    prompt = f"""Analyze this document and output JSON only (no explanation):
{{
  "title": "short title (max 10 words)",
  "summary": "what this document contains in ~30 words",
  "products": ["product names/codes mentioned (NOT image filenames)"],
  "hashtags": ["top 10 relevant keywords for categorization"]
}}

File: {filename}

{content[:6000]}"""

    response = _call_claude(prompt)
    data = _extract_json(response)
    if data:
        return data
    return {"title": "", "summary": response[:200] if not response.startswith("[Error") else "",
            "products": [], "hashtags": []}


def summarize_xlsx(content_preview: str, filename: str) -> dict:
    """Generate summary for an xlsx file (pass header + sample rows as text)."""
    prompt = f"""Analyze this table and output JSON only (no explanation):
{{
  "title": "short title (max 10 words)",
  "summary": "what data this table contains in ~30 words",
  "header_row": 1,
  "product_column": "column name containing product names/codes, or null",
  "hashtags": ["top 10 relevant keywords"]
}}

File: {filename}

{content_preview[:6000]}"""

    response = _call_claude(prompt)
    data = _extract_json(response)
    if data:
        return data
    return {"title": "", "summary": "", "header_row": 1, "product_column": None, "hashtags": []}


def _read_xlsx_preview(filepath: Path) -> str:
    """Read xlsx file and return header + sample rows as text."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return "Empty table"
        headers = None
        data_rows = []
        for row in rows:
            if any(cell is not None for cell in row):
                if headers is None:
                    headers = [str(c) if c else "" for c in row]
                else:
                    data_rows.append([str(c) if c else "" for c in row])
        if not headers:
            return "Empty table"
        content = f"Columns: {', '.join(h for h in headers if h)}\n\n"
        content += f"Total rows: {len(data_rows)}\n\nSample rows:\n"
        for row in data_rows[:5]:
            content += f"  {' | '.join(row[:len(headers)])}\n"
        wb.close()
        return content
    except Exception as e:
        return f"Error reading xlsx: {e}"


# ── Image summarization ───────────────────────────────────

def summarize_image_from_filename(filepath: Path) -> str:
    """Generate a hint from filename patterns (no API call)."""
    name = filepath.stem.lower()
    parts = name.split('_')
    page_num = parts[0] if parts and parts[0].isdigit() else ""

    series = ""
    for p in parts[1:4]:
        if re.match(r'^[a-z]{1,5}$', p) or re.match(r'^[a-z]+\d+$', p):
            series = p.upper()
            break

    parent = filepath.parent.name
    if "product" in name or parent == "products":
        img_type = "Product photo"
    elif "dimension" in name or parent == "dimensions":
        lang = "German" if "_de" in name else "English" if "_en" in name else ""
        img_type = f"Technical dimension drawing{' (' + lang + ')' if lang else ''}"
    elif "pattern" in name:
        img_type = "Pattern visualization"
    elif "flow" in name:
        img_type = "Flow diagram"
    elif "logo" in name:
        img_type = "Logo"
    elif "table" in name:
        img_type = "Data table image"
    elif "silver" in name:
        img_type = "Product photo (silver variant)"
    elif "adapter" in name:
        img_type = "Adapter diagram"
    elif "distribution" in name:
        img_type = "Distribution pattern"
    elif "cleaning" in name or "cycle" in name:
        img_type = "Cleaning cycle diagram"
    else:
        img_type = "Technical image"

    if series:
        return f"{img_type} of {series} series"
    elif page_num:
        return f"{img_type} from catalog page {int(page_num)}"
    return img_type


def summarize_image_with_claude(filepath: Path) -> str:
    """Use claude CLI with vision to describe an image."""
    # claude -p can't do vision directly, fall back to filename-based
    return summarize_image_from_filename(filepath)


# ── PDF summarization ─────────────────────────────────────

def summarize_pdf(pdf_path: Path, max_pages: int = 120) -> dict:
    """Generate global summary + per-page summaries for a PDF.

    Returns: {"filename": "...", "global_summary": "...",
              "page_count": N, "pages": [{"page": 1, "summary": "..."}, ...]}
    """
    # Extract text per page
    pages_text = _extract_pdf_pages(pdf_path, max_pages)
    if not pages_text:
        return {"filename": pdf_path.name, "global_summary": "Could not extract text",
                "page_count": 0, "pages": []}

    # Generate per-page summaries in batches
    page_summaries = []
    batch_size = 10
    for i in range(0, len(pages_text), batch_size):
        batch = pages_text[i:i + batch_size]
        batch_text = "\n\n".join(
            f"--- PAGE {pg['page']} ---\n{pg['text'][:800]}" for pg in batch
        )
        prompt = f"""For each page below, write a 1-sentence summary of its content.
Output JSON array only:
[{{"page": N, "summary": "..."}}]

{batch_text}"""

        response = _call_claude(prompt, timeout=120)
        # Try to parse as JSON array
        try:
            arr = json.loads(response)
            if isinstance(arr, list):
                page_summaries.extend(arr)
                continue
        except json.JSONDecodeError:
            pass
        # Try to extract from response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group())
                if isinstance(arr, list):
                    page_summaries.extend(arr)
                    continue
            except json.JSONDecodeError:
                pass
        # Fallback: one summary per page from batch
        for pg in batch:
            page_summaries.append({"page": pg["page"], "summary": pg["text"][:100].strip()})

    # Generate global summary from page summaries
    pages_preview = "\n".join(
        f"p.{ps['page']}: {ps['summary'][:80]}" for ps in page_summaries[:40]
    )
    prompt = f"""Summarize this document in 2-3 sentences based on the page summaries below.
Output only the summary text.

Filename: {pdf_path.name}
Total pages: {len(pages_text)}

{pages_preview}"""

    global_summary = _call_claude(prompt)
    if global_summary.startswith("[Error"):
        global_summary = f"PDF with {len(pages_text)} pages"

    return {
        "filename": pdf_path.name,
        "global_summary": global_summary,
        "page_count": len(pages_text),
        "pages": page_summaries,
    }


def _extract_pdf_pages(pdf_path: Path, max_pages: int = 120) -> list[dict]:
    """Extract text from each page of a PDF."""
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages], 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i, "text": text.strip()})
    except ImportError:
        # Fallback: use subprocess with python and PyPDF2 or just note it
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages[:max_pages], 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i, "text": text.strip()})
        except ImportError:
            # Last resort: use claude to describe the PDF
            pages.append({"page": 1, "text": f"[Could not extract text from {pdf_path.name} — install pdfplumber]"})
    except Exception as e:
        pages.append({"page": 1, "text": f"[Error extracting PDF: {e}]"})
    return pages


# ── Batch processing ──────────────────────────────────────

def process_directory(
    directory: Path,
    force: bool = False,
    parallel: int = 3,
    file_types: tuple[str, ...] = (".md", ".xlsx"),
) -> dict[str, int]:
    """Process all md/xlsx files in a directory, generating JSON sidecars.

    Returns counts: {"processed": N, "skipped": N, "errors": N}
    """
    files = []
    for ext in file_types:
        files.extend(directory.glob(f"*{ext}"))
    files = sorted(files)

    if not force:
        files = [f for f in files if not f.with_suffix(".json").exists()]

    if not files:
        print(f"All {len(list(directory.glob('*.json')))} files already have sidecars.")
        return {"processed": 0, "skipped": 0, "errors": 0}

    print(f"Processing {len(files)} files (parallel={parallel})...")
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    def _process_one(filepath: Path) -> tuple[str, bool]:
        try:
            if filepath.suffix == ".xlsx":
                preview = _read_xlsx_preview(filepath)
                data = summarize_xlsx(preview, filepath.name)
            else:
                content = filepath.read_text(errors="ignore")
                data = summarize_markdown(content, filepath.name)

            json_path = filepath.with_suffix(".json")
            json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            return filepath.name, True
        except Exception as e:
            return f"{filepath.name}: {e}", False

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(_process_one, f): f for f in files}
        done = 0
        for future in as_completed(futures):
            done += 1
            name, ok = future.result()
            if ok:
                stats["processed"] += 1
                print(f"  [{done}/{len(files)}] {name}: OK")
            else:
                stats["errors"] += 1
                print(f"  [{done}/{len(files)}] ERROR: {name}")

    return stats


def generate_image_hints(
    images_dir: Path,
    output_file: Path,
    force: bool = False,
) -> int:
    """Generate image hints for all images in a directory.

    Returns number of new hints generated.
    """
    existing = {}
    if output_file.exists() and not force:
        try:
            existing = json.loads(output_file.read_text())
        except Exception:
            pass

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(images_dir.rglob(ext))
    images = sorted(images)

    new_count = 0
    for img in images:
        try:
            key = str(img.relative_to(images_dir))
        except ValueError:
            key = img.name

        if key not in existing or force:
            existing[key] = summarize_image_from_filename(img)
            new_count += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"Image hints: {len(existing)} total, {new_count} new")
    return new_count
