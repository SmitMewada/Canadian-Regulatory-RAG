import fitz  # PyMuPDF
import re
from pathlib import Path
from src.models.schemas import DocumentRecord


def extract_text_from_pdf(file_path: str) -> tuple[list[dict], int]:
    """
    Extract text from a PDF file page by page.
    Returns a list of page dicts and total page count.
    """
    pdf = fitz.open(file_path)
    pages = []
    total_pages = len(pdf)

    for page_num in range(total_pages):
        page = pdf[page_num]
        text = page.get_text("text")

        # Clean the text
        text = clean_text(text)

        if not text.strip():
            continue  # Skip blank pages

        section_heading = extract_section_heading(text)

        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "section_heading": section_heading,
        })

    pdf.close()
    return pages, total_pages


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    - Remove excessive whitespace
    - Fix common encoding artifacts
    - Remove page numbers standing alone
    - Normalize line breaks
    """
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode dashes and quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    # Remove standalone page numbers (e.g., "\n12\n")
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def extract_section_heading(text: str) -> str | None:
    """
    Try to extract a section heading from the beginning of page text.
    Looks for patterns like:
        "1. Introduction"
        "Section 4.2 — Model Validation"
        "PART A — SCOPE"
    Returns None if no heading found.
    """
    lines = text.strip().split('\n')

    for line in lines[:5]:  # Check first 5 lines only
        line = line.strip()

        # Match numbered sections: "1.", "4.2", "A."
        if re.match(r'^(\d+\.?\d*|[A-Z]\.)\s+[A-Z]', line):
            return line[:100]  # Cap at 100 chars

        # Match ALL CAPS headings (common in regulatory docs)
        if line.isupper() and 5 < len(line) < 80:
            return line

        # Match "Section X" pattern
        if re.match(r'^(Section|SECTION|Part|PART|Appendix|APPENDIX)', line):
            return line[:100]

    return None


def extract_all(records: list[DocumentRecord]) -> list[dict]:
    """
    Extract text from all registered documents.
    Returns list of page dicts with document_id attached.
    """
    all_pages = []

    for record in records:
        print(f"Extracting: {record.title}")

        pages, total_pages = extract_text_from_pdf(record.file_path)

        for page in pages:
            page["document_id"] = record.id

        all_pages.extend(pages)
        print(f"  Pages extracted: {total_pages} total, {len(pages)} non-empty")

    print(f"\nTotal pages extracted: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.models.schemas import DocumentRecord

    # Load manifest
    manifest = json.loads(Path("data/manifest.json").read_text())
    records = [DocumentRecord(**v) for v in manifest.values()]

    # Extract
    pages = extract_all(records)

    # Quick sanity check — print first page of first doc
    print("\n--- Sample Page ---")
    sample = pages[0]
    print(f"Document: {sample['document_id']}")
    print(f"Page: {sample['page_number']}")
    print(f"Heading: {sample['section_heading']}")
    print(f"Text preview: {sample['text'][:300]}")