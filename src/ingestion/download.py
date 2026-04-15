import hashlib
import json
from datetime import datetime
from pathlib import Path
from src.models.schemas import DocumentRecord
import fitz


RAW_DATA_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.json")

# Map your exact filenames to clean document IDs
DOCUMENTS = [
    {
        "id": "osfi_e23",
        "title": "Guideline E-23 Model Risk Management 2027",
        "filename": "Guideline E23  Model Risk Management 2027.pdf",
        "source_url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/guideline-e-23-model-risk-management-2027",
    },
    {
        "id": "osfi_e23_letter",
        "title": "Guideline E-23 Model Risk Management 2027 Letter",
        "filename": "Guideline E23  Model Risk Management 2027  Letter.pdf",
        "source_url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/guideline-e-23-model-risk-management-2027-letter",
    },
    {
        "id": "tb_adm",
        "title": "Directive on Automated Decision-Making",
        "filename": "Directive on Automated Decision-Making- Canada.ca.pdf",
        "source_url": "https://www.tbs-sct.canada.ca/pol/doc-eng.aspx?id=32592",
    },
    {
        "id": "tb_adm_guide",
        "title": "Guide on the Scope of the Directive on Automated Decision-Making",
        "filename": "Guide on the Scope of the Directive on Automated Decision-Making - Canada.ca.pdf",
        "source_url": "https://www.canada.ca/en/government/system/digital-government/digital-government-innovations/responsible-use-ai/guide-scope-directive-automated-decision-making.html",
    },
    {
        "id": "pipeda",
        "title": "Personal Information Protection and Electronic Documents Act",
        "filename": "Personal Information Protection and Electronic Documents Act.pdf",
        "source_url": "https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/",
    },
    {
        "id": "aia",
        "title": "Algorithmic Impact Assessment Tool",
        "filename": "Algorithmic Impact Assessment tool - Canada.ca.pdf",
        "source_url": "https://www.canada.ca/en/government/system/digital-government/digital-government-innovations/responsible-use-ai/algorithmic-impact-assessment.html",
    },
    {
        "id": "osfi_b13",
        "title": "Technology and Cyber Risk Management",
        "filename": "Technology and Cyber Risk Management.pdf",
        "source_url": "https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/technology-and-cyber-risk-management-guideline",
    },
]


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for version detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> dict:
    """Load existing manifest or return empty dict."""
    if MANIFEST_PATH.exists():
        content = MANIFEST_PATH.read_text().strip()
        if not content:
            return {}
        return json.loads(content)
    return {}


def save_manifest(manifest: dict) -> None:
    """Save manifest to disk."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def register_existing_documents() -> list[DocumentRecord]:
    """
    Register already-downloaded documents by computing their hashes
    and building the manifest. No network calls needed.
    """
    manifest = load_manifest()
    registered = []

    for doc in DOCUMENTS:
        file_path = RAW_DATA_DIR / doc["filename"]

        if not file_path.exists():
            print(f"  MISSING: {doc['filename']}")
            continue

        sha256_hash = compute_sha256(file_path)
        file_size_kb = file_path.stat().st_size // 1024

        # Count pages for PDFs
        total_pages = None
        if file_path.suffix.lower() == ".pdf":
            pdf = fitz.open(str(file_path))
            total_pages = len(pdf)
            pdf.close()

        record = DocumentRecord(
            id=doc["id"],
            title=doc["title"],
            source_url=doc["source_url"],
            file_path=str(file_path),
            sha256_hash=sha256_hash,
            date_accessed=datetime.now(),
            total_pages=total_pages,
        )

        manifest[record.id] = record.model_dump()
        registered.append(record)

        print(f"  Registered: {doc['id']} ({file_size_kb} KB) pages={total_pages} hash={sha256_hash[:12]}...")

    save_manifest(manifest)  # THIS WAS MISSING
    print(f"\nRegistered {len(registered)}/{len(DOCUMENTS)} documents")
    return registered


if __name__ == "__main__":
    register_existing_documents()