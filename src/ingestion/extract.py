import pdfplumber
import re
from pathlib import Path
from src.models.schemas import DocumentRecord


def extract_text_from_pdf(file_path: str) -> tuple[list[dict], int]:
    """
    Extract text from a PDF file page by page using pdfplumber.
    Handles both regular text and tables cleanly.
    Returns list of page dicts and total page count.
    """
    pages = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            # Extract tables first — convert to readable text
            table_text = extract_tables_as_text(page)

            # Extract regular text
            raw_text = page.extract_text() or ""

            # Combine: regular text + table text
            combined = raw_text
            if table_text:
                combined = raw_text + "\n\n" + table_text

            cleaned = clean_text(combined)

            if not cleaned.strip():
                continue

            section_heading = extract_section_heading(cleaned)

            pages.append({
                "page_number": page_num + 1,
                "text": cleaned,
                "section_heading": section_heading,
            })

    return pages, total_pages


def extract_tables_as_text(page) -> str:
    """
    Extract tables from a page and convert to readable text.
    Important for regulatory docs with structured requirements tables.
    """
    tables = page.extract_tables()
    if not tables:
        return ""

    table_texts = []
    for table in tables:
        rows = []
        for row in table:
            # Filter None cells and join with pipe separator
            clean_row = [str(cell).strip() if cell else "" for cell in row]
            if any(clean_row):  # Skip completely empty rows
                rows.append(" | ".join(clean_row))
        if rows:
            table_texts.append("\n".join(rows))

    return "\n\n".join(table_texts)


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.
    """
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode dashes and quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    # Remove standalone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def extract_section_heading(text: str) -> str | None:
    """
    Extract section heading from page text.
    Looks for numbered sections and ALL CAPS headings
    common in Canadian regulatory documents.
    """
    lines = text.strip().split('\n')

    for line in lines[:5]:
        line = line.strip()

        # Numbered sections: "1.", "4.2", "A."
        if re.match(r'^(\d+\.?\d*|[A-Z]\.)\s+[A-Z]', line):
            return line[:100]

        # ALL CAPS headings
        if line.isupper() and 5 < len(line) < 80:
            return line

        # "Section/Part/Appendix" patterns
        if re.match(r'^(Section|SECTION|Part|PART|Appendix|APPENDIX)', line):
            return line[:100]

    return None


def extract_all(records: list[DocumentRecord]) -> list[dict]:
    """
    Extract text from all registered documents.
    """
    all_pages = []

    for record in records:
        print(f"Extracting: {record.title}")
        pages, total_pages = extract_text_from_pdf(record.file_path)

        for page in pages:
            page["document_id"] = record.id

        all_pages.extend(pages)
        print(f"  Pages: {total_pages} total, {len(pages)} non-empty")

    print(f"\nTotal pages extracted: {len(all_pages)}")
    return all_pages


if __name__ == "__main__":
    import json
    from pathlib import Path
    from src.models.schemas import DocumentRecord

    manifest = json.loads(Path("data/manifest.json").read_text())
    records = [DocumentRecord(**v) for v in manifest.values()]

    pages = extract_all(records)

    print("\n--- Sample Page ---")
    sample = pages[0]
    print(f"Document: {sample['document_id']}")
    print(f"Page: {sample['page_number']}")
    print(f"Heading: {sample['section_heading']}")
    print(f"Text preview:\n{sample['text'][:400]}")