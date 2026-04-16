import pdfplumber
import re
from pathlib import Path
from src.models.schemas import DocumentRecord

# Patterns that indicate a TOC page
TOC_INDICATORS = [
    r'Table of Contents',
    r'On this page\s*\n',
]
NOISE_LINES = [
    r'^Canada\.ca\s*$',
    r'^Canada\.ca\s+',
    r'How government works',
    r'About government',
    r'Government in a digital age',
    r'Digital government innovation',
    r'Responsible use of artificial intelligence in government',
    r'^On this page\s*$',
    r'^View complete hierarchy\s*$',
    r'^View all inactive instruments\s*$',
    r'^Print-friendly XML\s*$',
    r'^Expand all\s+Collapse all\s*$',
    r'^ View complete hierarchy\s*$',
    r'^Archives\s*$',
    r'^Date modified:',
    r'^Supporting tools\s*$',
    r'^Tools:\s*$',
    r'^More information\s*$',
    r'^Policy:\s*$',
    r'^Topic:\s*$',
    r'^Hierarchy\s*$',
    # OSFI repeating footer
    r'^Guideline E-23',
    r'^Office of the Superintendent of Financial Institutions\s*$',
    r'^Page \d+\s*$',
    # B-13 footer
    r'^Technology and Cyber Risk Management\s*$',
    r'^Table of Contents\s*$',
]


def is_toc_line(line: str) -> bool:
    """
    Returns True if line looks like a TOC entry.
    Matches patterns like:
        A. Overview
        A.1 Purpose
        B.1 Organizational enablement
        C.3 Risk management intensity
        Appendix 1: ...
        Footnotes
    """
    line = line.strip()
    if not line or len(line) > 80:
        return False

    # Matches: A., A.1, B.2, C.3, D.1 etc followed by text
    if re.match(r'^[A-Z]\d*\.?\d*\s+\w', line):
        return True

    # Matches: Appendix, Footnotes
    if re.match(r'^(Appendix|Footnotes)', line):
        return True

    return False

def remove_noise_lines(text: str) -> str:
    """
    Remove noise lines anywhere in the text.
    Handles website chrome, navigation elements, and TOC entries
    that appear mixed with real content.
    """
    lines = text.split('\n')
    cleaned = []
    
    # Track consecutive TOC lines — if we see 3+ in a row, strip the block
    toc_buffer = []
    
    for line in lines:
        # Check hard noise patterns
        is_noise = any(re.search(pattern, line) for pattern in NOISE_LINES)
        if is_noise:
            continue
            
        # Check TOC lines
        if is_toc_line(line):
            toc_buffer.append(line)
            continue
        else:
            # If we had fewer than 3 TOC lines, they might be real content
            # (e.g. a section heading that looks like a TOC entry)
            if len(toc_buffer) < 3:
                cleaned.extend(toc_buffer)
            # If 3 or more consecutive TOC lines — it's a real TOC block, discard
            toc_buffer = []
            cleaned.append(line)
    
    # Handle any remaining buffer
    if len(toc_buffer) < 3:
        cleaned.extend(toc_buffer)
    
    return '\n'.join(cleaned)


def should_skip_page(text: str) -> bool:
    """
    Returns True only if the ENTIRE page is navigation with zero content.
    After noise removal, if less than 100 chars remain — skip it.
    """
    after_cleaning = remove_noise_lines(text)
    return len(after_cleaning.strip()) < 100


def extract_text_from_pdf(file_path: str) -> tuple[list[dict], int]:
    pages = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            table_text = extract_tables_as_text(page)
            raw_text = page.extract_text() or ""

            combined = raw_text
            if table_text:
                combined = raw_text + "\n\n" + table_text

            # Remove noise lines
            combined = remove_noise_lines(combined)

            # Skip if nothing meaningful remains
            if should_skip_page(combined):
                print(f"    Skipping empty/nav page {page_num + 1}")
                continue

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