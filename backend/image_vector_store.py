import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from backend.config import IMAGE_INDEX_PATH, IMAGE_METADATA_PATH
from backend.image_embedder import embed_image

image_index = None
image_metadata = []
load_error = None


def _load_resources() -> None:
    global image_index, image_metadata, load_error
    if image_index is not None and image_metadata:
        return

    if not Path(IMAGE_INDEX_PATH).exists():
        load_error = (
            f"Image index is missing at '{IMAGE_INDEX_PATH}'. "
            "Run: python scripts/extract_textbook_images.py && python scripts/create_image_embeddings.py"
        )
        return

    if not Path(IMAGE_METADATA_PATH).exists():
        load_error = (
            f"Image metadata is missing at '{IMAGE_METADATA_PATH}'. "
            "Run: python scripts/extract_textbook_images.py"
        )
        return

    try:
        image_index = faiss.read_index(str(IMAGE_INDEX_PATH))
        with open(IMAGE_METADATA_PATH, "r", encoding="utf-8") as f:
            image_metadata = json.load(f)
        load_error = None
    except Exception as exc:
        load_error = f"Failed to load image vector resources: {exc}"


def _normalize_source(value: str | None) -> str:
    if not value:
        return ""
    return Path(value).name.lower().strip()


def _build_allowed_page_lookup(
    allowed_source_pages: Iterable[tuple[str, int]] | None,
) -> set[tuple[str, int]]:
    if not allowed_source_pages:
        return set()
    normalized = set()
    for source, page in allowed_source_pages:
        normalized.add((_normalize_source(source), int(page)))
    return normalized


def search_similar_images(
    image_path: str,
    k: int = 3,
    body_site: str | None = None,
    allowed_source_pages: Iterable[tuple[str, int]] | None = None,
):
    _load_resources()
    if load_error:
        raise RuntimeError(load_error)

    q = embed_image(image_path)
    q = np.array(q, dtype=np.float32)

    candidate_pool = max(k * 8, 50)
    distances, indices = image_index.search(q, candidate_pool)
    allowed_lookup = _build_allowed_page_lookup(allowed_source_pages)

    results = []
    scoped_results = []

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        item = dict(image_metadata[idx])
        score = float(distances[0][rank])
        item_source = _normalize_source(item.get("source") or item.get("source_pdf"))
        item_page = int(item.get("page", 0))

        # body-site boost
        if body_site and item.get("body_site"):
            if item["body_site"].lower() == body_site.lower():
                score += 0.15

        item["score"] = score
        results.append(item)
        if allowed_lookup and (item_source, item_page) in allowed_lookup:
            scoped_results.append(item)

    selected = scoped_results if scoped_results else results
    selected.sort(key=lambda x: x["score"], reverse=True)

    return selected[:k]