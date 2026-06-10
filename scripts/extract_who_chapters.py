"""
Extract relevant chapters from the WHO Classification of Skin Tumours book.
Keeps only content relevant to the 8 classifier classes:
  MEL, NV  → Chapter 2: Melanocytic tumours       (pages 65–151)
  BCC, AK,
  SCC, BKL → Chapter 1: Keratinocytic tumours      (pages 23–63)
  DF, VASC → Chapter 5: Soft tissue tumours        (pages 291–350)

Output: data/textbooks/who_skin_tumours_chapters_en.pdf
"""

from pathlib import Path
from pypdf import PdfReader, PdfWriter

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PDF = BASE_DIR / "data" / "textbooks" / "who_skin_tumours_classification_en.pdf"
OUTPUT_PDF = BASE_DIR / "data" / "textbooks" / "who_skin_tumours_chapters_en.pdf"

# Page ranges to keep (0-indexed, PDF has 1 extra cover page so subtract 2
# from printed page numbers instead of 1)
# Printed pages 23-63   → indices 21-61
# Printed pages 65-151  → indices 63-149
# Printed pages 291-350 → indices 289-348
RANGES = [
    (21, 61),    # Chapter 1: Keratinocytic — BCC, AK, SCC, BKL
    (63, 149),   # Chapter 2: Melanocytic  — MEL, NV
    (289, 348),  # Chapter 5: Soft tissue  — DF, VASC
]


def main():
    print(f"Reading: {INPUT_PDF}")
    reader = PdfReader(str(INPUT_PDF))
    total = len(reader.pages)
    print(f"Total pages in book: {total}")

    writer = PdfWriter()
    pages_added = 0

    for start, end in RANGES:
        actual_end = min(end, total - 1)
        for i in range(start, actual_end + 1):
            writer.add_page(reader.pages[i])
            pages_added += 1
        print(f"  Added pages {start+1}–{actual_end+1} ({actual_end - start + 1} pages)")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PDF, "wb") as f:
        writer.write(f)

    print(f"\nDone. Extracted {pages_added} pages → {OUTPUT_PDF}")
    print("You can now delete who_skin_tumours_classification_en.pdf from textbooks/")


if __name__ == "__main__":
    main()
