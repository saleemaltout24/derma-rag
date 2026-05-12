import json
import re
from pathlib import Path

from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "textbooks"
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

MIN_CHARS = 800
MAX_CHARS = 1800
OVERLAP_CHARS = 250


def detect_book_language(filename: str) -> str:
    name = filename.lower()

    if "_tr" in name or name.endswith("tr.pdf") or "turk" in name or "türk" in name:
        return "tr"

    if "_en" in name or name.endswith("en.pdf") or "english" in name:
        return "en"

    return "en"


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\u2010", "-")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    return text.strip()


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()

    if len(p) < 3 or len(p) > 120:
        return False

    if "\n" in p:
        return False

    if re.match(r"^\d+(\.\d+)*\s+[A-ZÇĞİÖŞÜ]", p):
        return True

    if re.match(r"^[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9 \-,:()/]{3,}$", p):
        return True

    if p.endswith(":") and len(p) < 80:
        return True

    return False


def split_into_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    cleaned = []

    for part in parts:
        p = clean_text(part)
        if p:
            cleaned.append(p)

    return cleaned


def split_long_paragraph(paragraph: str, max_chars: int = 900) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces


def build_semantic_chunks(paragraphs: list[str]) -> list[dict]:
    chunks = []
    current_chunk = ""
    current_section = "Unknown section"

    for raw_paragraph in paragraphs:
        if is_heading(raw_paragraph):
            if current_chunk.strip():
                chunks.append({
                    "section_title": current_section,
                    "text": current_chunk.strip(),
                })
                current_chunk = ""

            current_section = raw_paragraph.strip()
            continue

        sub_parts = split_long_paragraph(raw_paragraph)

        for paragraph in sub_parts:
            candidate = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph

            if len(candidate) <= MAX_CHARS:
                current_chunk = candidate
            else:
                if current_chunk.strip():
                    chunks.append({
                        "section_title": current_section,
                        "text": current_chunk.strip(),
                    })

                    overlap = current_chunk[-OVERLAP_CHARS:] if len(current_chunk) > OVERLAP_CHARS else current_chunk
                    current_chunk = f"{overlap}\n\n{paragraph}".strip()
                else:
                    chunks.append({
                        "section_title": current_section,
                        "text": paragraph.strip(),
                    })
                    current_chunk = ""

    if current_chunk.strip():
        chunks.append({
            "section_title": current_section,
            "text": current_chunk.strip(),
        })

    merged = []
    for chunk in chunks:
        if merged and len(chunk["text"]) < MIN_CHARS:
            merged[-1]["text"] = f'{merged[-1]["text"]}\n\n{chunk["text"]}'.strip()
        else:
            merged.append(chunk)

    return merged


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text()
        except Exception:
            extracted = ""

        if extracted:
            pages.append(extracted)

        if i % 20 == 0:
            print(f"  Read {i} pages from {pdf_path.name}...")

    return clean_text("\n\n".join(pages))


def main():
    print(f"Looking for PDFs in: {INPUT_DIR}")
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s).")

    if not pdf_files:
        print("No PDF files found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    global_chunk_id = 1

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        book_language = detect_book_language(pdf_file.name)
        text = extract_pdf_text(pdf_file)

        if not text:
            print(f"  No extractable text found in {pdf_file.name}")
            continue

        paragraphs = split_into_paragraphs(text)
        semantic_chunks = build_semantic_chunks(paragraphs)

        print(f"  Created {len(semantic_chunks)} semantic chunks")

        for local_id, chunk in enumerate(semantic_chunks, start=1):
            all_chunks.append({
                "chunk_id": global_chunk_id,
                "source": pdf_file.name,
                "source_path": str(pdf_file),
                "local_chunk_id": local_id,
                "language": book_language,
                "section_title": chunk["section_title"],
                "text": chunk["text"],
            })
            global_chunk_id += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()