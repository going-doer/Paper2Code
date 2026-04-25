"""
pdf_to_json.py - Convert a PDF file into the s2orc-compatible JSON format
expected by 0_pdf_process.py, without needing GROBID.

Uses PyMuPDF (fitz) for text extraction.

Install:  pip install pymupdf

Output schema (mirrors s2orc raw JSON):
{
  "paper_id": "<stem>",
  "title": "<str>",
  "abstract": "<str>",
  "pdf_parse": {
    "paper_id": "<stem>",
    "abstract": [{"text": "<str>", "cite_spans": [], "ref_spans": [],
                  "eq_spans": [], "section": "Abstract", "sec_num": null}],
    "body_text": [{"text": "<str>", "cite_spans": [], "ref_spans": [],
                   "eq_spans": [], "section": "<heading>", "sec_num": null}, ...],
    "back_matter": [],
    "bib_entries": {},
    "ref_entries": {}
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print(
        "[ERROR] PyMuPDF is not installed. Run: pip install pymupdf",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Heuristics for section heading detection
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"""^
    (?:
        (?:\d+\.?)+\s+[A-Z]      # "1 Introduction" / "2.1 Background"
        | [A-Z][A-Z\s]{3,}$       # ALL-CAPS heading
        | (?:Abstract|Introduction|Related\s+Work|Background|Method(?:ology)?|
             Experiment(?:s|al\s+Setup)?|Results?|Discussion|Conclusion(?:s)?|
             References?|Appendix|Acknowledgements?)\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Sections that are typically back-matter
_BACK_MATTER_RE = re.compile(
    r"^(?:References?|Bibliography|Acknowledgements?|Appendix)\b",
    re.IGNORECASE,
)


def _is_heading(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 120:
        return False
    return bool(_HEADING_RE.match(text))


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

_JUNK_TITLE_RE = re.compile(
    r"arXiv|doi\.org|preprint|©|copyright|\d{4}\s+IEEE"
    r"|proceedings of|workshop on|journal of|vol\.\s*\d|pages?\s+\d+",
    re.IGNORECASE,
)


def _extract_title(doc: fitz.Document) -> str:
    """Best-effort title: PDF metadata -> largest non-junk font cluster on page 1."""
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if meta_title and len(meta_title) > 8 and not _JUNK_TITLE_RE.search(meta_title):
        return meta_title

    # Collect all (font_size, y_position, text) spans from page 1
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    spans: list[tuple[float, float, str]] = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                t = span["text"].strip()
                if t:
                    spans.append((span["size"], span["origin"][1], t))

    if not spans:
        return ""

    # Sort by font size descending, skip junk patterns and single chars
    spans.sort(key=lambda x: -x[0])
    clean = [(sz, y, t) for sz, y, t in spans
             if not _JUNK_TITLE_RE.search(t) and len(t) >= 3]
    if not clean:
        clean = spans  # fall back if everything looked like junk

    # The title is the largest-font cluster (within 2pt of the top clean size)
    top_size = clean[0][0]
    title_spans = [(sz, y, t) for sz, y, t in clean if abs(sz - top_size) < 2.0]
    # Sort by y-position (top to bottom) to preserve reading order
    title_spans.sort(key=lambda x: x[1])
    return " ".join(t for _, _, t in title_spans).strip()


def _extract_abstract(pages_text: list[str]) -> str:
    """Pull the abstract paragraph from the first two pages."""
    combined = "\n".join(pages_text[:3])
    # Try to find "Abstract" heading followed by text
    m = re.search(
        r"Abstract[.\s\n]*([A-Z].*?)(?=\n\s*(?:\d+\.?\s+[A-Z]|Introduction\b|Keywords?\b))",
        combined,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def pdf_to_json(pdf_path: str) -> dict:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc = fitz.open(pdf_path)

    # --- per-page plain text ---
    pages_text = [page.get_text("text") for page in doc]

    title = _extract_title(doc)
    abstract_text = _extract_abstract(pages_text)

    # --- build body paragraphs ---
    full_text = "\n".join(pages_text)
    # Split by blank lines to get paragraphs
    raw_paragraphs = re.split(r"\n{2,}", full_text)

    current_section = "Introduction"
    abstract_blocks: list[dict] = []
    body_blocks: list[dict] = []
    back_matter_blocks: list[dict] = []

    in_abstract = False
    past_abstract = False
    in_back_matter = False

    for para in raw_paragraphs:
        para = para.strip()
        if not para or len(para) < 20:
            continue

        first_line = para.split("\n")[0].strip()

        # Detect abstract section
        if re.match(r"^Abstract\b", first_line, re.IGNORECASE) and not past_abstract:
            in_abstract = True
            body_of_para = "\n".join(para.split("\n")[1:]).strip()
            if body_of_para:
                abstract_blocks.append(_make_block(body_of_para, "Abstract", None))
            continue

        if in_abstract and _is_heading(first_line):
            in_abstract = False
            past_abstract = True

        if in_abstract:
            abstract_blocks.append(_make_block(para, "Abstract", None))
            continue

        # Detect back matter
        if _BACK_MATTER_RE.match(first_line):
            in_back_matter = True
            current_section = first_line
            continue

        # Detect regular section headings
        if _is_heading(first_line) and not in_back_matter:
            current_section = re.sub(r"\s+", " ", first_line).strip()
            body_of_para = "\n".join(para.split("\n")[1:]).strip()
            if body_of_para:
                body_blocks.append(_make_block(body_of_para, current_section, None))
            continue

        text_clean = re.sub(r"\s+", " ", para).strip()
        block = _make_block(text_clean, current_section, None)

        if in_back_matter:
            back_matter_blocks.append(block)
        else:
            body_blocks.append(block)

    # If we got no abstract blocks from section detection, use the extracted text
    if not abstract_blocks and abstract_text:
        abstract_blocks = [_make_block(abstract_text, "Abstract", None)]

    return {
        "paper_id": stem,
        "title": title,
        "abstract": abstract_text,
        "pdf_parse": {
            "paper_id": stem,
            "_pdf_hash": "",
            "abstract": abstract_blocks,
            "body_text": body_blocks,
            "back_matter": back_matter_blocks,
            "bib_entries": {},
            "ref_entries": {},
        },
    }


def _make_block(text: str, section: str, sec_num) -> dict:
    return {
        "text": text,
        "cite_spans": [],
        "ref_spans": [],
        "eq_spans": [],
        "section": section,
        "sec_num": sec_num,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF to s2orc-compatible JSON (no GROBID required)."
    )
    parser.add_argument("--pdf_path", required=True, help="Path to input PDF file")
    parser.add_argument("--output_json_path", required=True, help="Path for output JSON")
    args = parser.parse_args()

    print(f"[pdf_to_json] Reading {args.pdf_path} ...")
    data = pdf_to_json(args.pdf_path)
    with open(args.output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[pdf_to_json] Saved -> {args.output_json_path}")
    print(f"  title    : {data['title'][:80]}")
    print(f"  abstract : {data['abstract'][:120]}...")
    print(f"  body paragraphs : {len(data['pdf_parse']['body_text'])}")


if __name__ == "__main__":
    main()
