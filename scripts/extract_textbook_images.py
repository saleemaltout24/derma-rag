import sys
from pathlib import Path
import json

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.config import TEXTBOOK_DIR, TEXTBOOK_IMAGE_DIR, IMAGE_METADATA_PATH


DPI_SCALE = 2.0
MIN_REGION_WIDTH = 180
MIN_REGION_HEIGHT = 180
MAX_REGION_COVERAGE = 0.45


def pil_to_cv(pil_img: Image.Image):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_text_block_rects(page, scale: float):
    blocks = page.get_text("blocks")
    rects = []

    for block in blocks:
        x0, y0, x1, y1, text = block[:5]

        if not str(text).strip():
            continue

        rects.append((
            int(x0 * scale),
            int(y0 * scale),
            int(x1 * scale),
            int(y1 * scale),
        ))

    return rects


def remove_text_regions(mask, text_rects):
    for x0, y0, x1, y1 in text_rects:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 0, thickness=-1)
    return mask


def is_photo_like(crop: Image.Image) -> bool:
    img = np.array(crop.convert("RGB"))
    h, w, _ = img.shape

    if w < MIN_REGION_WIDTH or h < MIN_REGION_HEIGHT:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1. Text-heavy regions tend to have lots of sharp black/white transitions
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.mean(edges > 0)

    # 2. Real clinical photos usually have more color variation
    color_std = float(np.std(img))

    # 3. Text blocks often have very high white background ratio
    white_ratio = np.mean(np.all(img > 235, axis=2))

    # 4. Text-like areas often have many tiny connected components
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    small_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 5 <= area <= 150:
            small_components += 1

    component_density = small_components / max((w * h) / 10000, 1)

    # Heuristics:
    # reject text/table-like crops
    if white_ratio > 0.60 and edge_ratio > 0.08:
        return False

    if color_std < 18:
        return False

    if component_density > 35:
        return False

    return True


def find_candidate_regions(page_image: Image.Image, text_rects):
    img = pil_to_cv(page_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold content
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        12,
    )

    thresh = remove_text_regions(thresh, text_rects)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_h, page_w = gray.shape
    page_area = page_w * page_h

    regions = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        coverage = area / page_area

        if w < MIN_REGION_WIDTH or h < MIN_REGION_HEIGHT:
            continue

        if coverage > MAX_REGION_COVERAGE:
            continue

        aspect_ratio = w / max(h, 1)
        if aspect_ratio < 0.25 or aspect_ratio > 4.5:
            continue

        regions.append((x, y, w, h))

    regions.sort(key=lambda r: (r[1], r[0]))
    return regions


def extract_regions_from_page(doc, page_num: int):
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI_SCALE, DPI_SCALE), alpha=False)
    page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    text_rects = get_text_block_rects(page, DPI_SCALE)
    regions = find_candidate_regions(page_image, text_rects)

    valid_crops = []

    for x, y, w, h in regions:
        crop = page_image.crop((x, y, x + w, y + h))

        if not is_photo_like(crop):
            continue

        valid_crops.append((crop, (x, y, w, h), page_image.size))

    return valid_crops


def main():
    pdf_files = list(TEXTBOOK_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/textbooks")
        return

    TEXTBOOK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata = []
    image_id = 1

    for pdf_path in pdf_files:
        print(f"Processing pages from: {pdf_path.name}")
        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            try:
                crops = extract_regions_from_page(doc, page_num)
            except Exception as e:
                print(f"Skipping page {page_num + 1} in {pdf_path.name}: {e}")
                continue

            for img_idx, (crop, region, page_size) in enumerate(crops, start=1):
                file_name = f"{pdf_path.stem}_page{page_num + 1}_crop{img_idx}.png"
                image_path = TEXTBOOK_IMAGE_DIR / file_name
                crop.save(image_path)

                x, y, w, h = region

                metadata.append({
                    "image_id": image_id,
                    "source": pdf_path.name,
                    "source_pdf": pdf_path.name,
                    "source_stem": pdf_path.stem,
                    "page": page_num + 1,
                    "image_path": str(image_path),
                    "file_name": file_name,
                    "crop_box": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                    },
                    "page_width": page_size[0],
                    "page_height": page_size[1],
                    "disease": "",
                    "body_site": "",
                    "caption": "",
                    "section_title": "",
                    "nearby_text": "",
                })

                image_id += 1

            if (page_num + 1) % 25 == 0:
                print(f"  Processed {page_num + 1} pages from {pdf_path.name}...")

        doc.close()

    with open(IMAGE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Extracted {len(metadata)} photo-like image crops.")
    print(f"Saved metadata to: {IMAGE_METADATA_PATH}")


if __name__ == "__main__":
    main()