"""
OCR the extracted WHO skin tumours chapters PDF and save as a text-based PDF
that process_books.py can read normally.

Requirements:
    pip install pymupdf pytesseract pillow

Usage (from project root):
    python scripts/ocr_who_chapters.py
"""

import sys
from pathlib import Path

import pytesseract
import fitz  # pymupdf
from PIL import Image
import io

# Hardcoded Tesseract path — avoids PATH issues on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PDF  = BASE_DIR / "data" / "textbooks" / "who_skin_tumours_chapters_en.pdf"
OUTPUT_TXT = BASE_DIR / "data" / "textbooks" / "who_skin_tumours_chapters_en_ocr.txt"

DPI = 200  # higher = better quality but slower; 200 is a good balance


def ocr_pdf(input_path: Path, output_path: Path) -> None:
    print(f"Opening: {input_path}")
    doc = fitz.open(str(input_path))
    total = len(doc)
    print(f"Total pages to OCR: {total}")

    all_text = []

    for i, page in enumerate(doc, start=1):
        # Render page to image
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        # OCR the image — English only (WHO book is English)
        text = pytesseract.image_to_string(img, lang="eng")
        all_text.append(text)

        if i % 10 == 0:
            print(f"  OCR'd {i}/{total} pages...")

    full_text = "\n\n".join(all_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\nDone. OCR text saved → {output_path}")
    print(f"Total characters extracted: {len(full_text):,}")


if __name__ == "__main__":
    if not INPUT_PDF.exists():
        print(f"ERROR: Input PDF not found: {INPUT_PDF}")
        sys.exit(1)
    ocr_pdf(INPUT_PDF, OUTPUT_TXT)
