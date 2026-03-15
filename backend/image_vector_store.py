import json
import faiss
import numpy as np

from backend.config import IMAGE_INDEX_PATH, IMAGE_METADATA_PATH
from backend.image_embedder import embed_image

image_index = faiss.read_index(str(IMAGE_INDEX_PATH))

with open(IMAGE_METADATA_PATH, "r", encoding="utf-8") as f:
    image_metadata = json.load(f)


def search_similar_images(image_path: str, k: int = 3, body_site: str | None = None):
    q = embed_image(image_path)
    q = np.array(q, dtype=np.float32)

    distances, indices = image_index.search(q, k * 3)  # retrieve more candidates

    results = []

    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        item = dict(image_metadata[idx])
        score = float(distances[0][rank])

        # body-site boost
        if body_site and item.get("body_site"):
            if item["body_site"].lower() == body_site.lower():
                score += 0.15

        item["score"] = score
        results.append(item)

    # sort by boosted score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:k]