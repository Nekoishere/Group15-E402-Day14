"""Ingest VinLex PDFs vào ChromaDB để lấy real chunk IDs."""
import hashlib
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from backend.vector_store import VectorStore
from config import CHROMA_DIR, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS

try:
    from pypdf import PdfReader
except ImportError:
    print("❌ pypdf not installed. Run: pip install pypdf")
    sys.exit(1)


def chunk_text(text: str, source_pdf: str, page: int) -> list[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE_CHARS
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = hashlib.md5(f"{source_pdf}:p{page}:c{chunk_idx}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source_pdf": source_pdf,
                    "page": page,
                    "chunk_index": chunk_idx,
                    "pdf_id": hashlib.md5(source_pdf.encode()).hexdigest(),
                },
            })
            chunk_idx += 1
        start += CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS
    return chunks


def ingest_pdf(pdf_path: Path, vs: VectorStore) -> list[dict]:
    print(f"  📄 Ingesting: {pdf_path.name}")
    reader = PdfReader(str(pdf_path))
    all_chunks = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 50:
            continue
        chunks = chunk_text(text, pdf_path.name, page_num)
        all_chunks.extend(chunks)

    vs.add_chunks(all_chunks)
    print(f"     → {len(all_chunks)} chunks added")
    return all_chunks


def main():
    pdfs_dir = Path("data/pdfs")
    pdf_files = list(pdfs_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDFs found in data/pdfs/")
        return

    print(f"🚀 Ingesting {len(pdf_files)} PDF(s) into ChromaDB...")
    vs = VectorStore(str(CHROMA_DIR))

    all_chunks = []
    for pdf_path in pdf_files:
        chunks = ingest_pdf(pdf_path, vs)
        all_chunks.extend(chunks)

    print(f"\n✅ Total chunks ingested: {vs.count()}")
    print("💾 ChromaDB updated at:", CHROMA_DIR)


if __name__ == "__main__":
    main()
