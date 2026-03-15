import sys
from pathlib import Path
import json

import faiss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.config import IMAGE_METADATA_PATH, IMAGE_INDEX_PATH
from backend.image_embedder import embed_image


def main():
    with open(IMAGE_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not metadata:
        print("No image metadata found.")
        return

    valid_metadata = []
    vectors = []

    for i, item in enumerate(metadata, start=1):
        image_path = item["image_path"]

        try:
            vec = embed_image(image_path)
            vectors.append(vec[0])
            valid_metadata.append(item)
        except Exception as e:
            print(f"Skipping image {image_path}: {e}")

        if i % 20 == 0:
            print(f"Processed {i}/{len(metadata)} images")

    if not vectors:
        print("No valid image embeddings were created.")
        return

    embeddings = np.array(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    IMAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(IMAGE_INDEX_PATH))

    with open(IMAGE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(valid_metadata, f, ensure_ascii=False, indent=2)

    print("Image FAISS index created successfully.")
    print(f"Saved to: {IMAGE_INDEX_PATH}")


if __name__ == "__main__":
    main()