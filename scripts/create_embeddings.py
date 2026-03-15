import sys
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.config import CHUNKS_PATH, TEXT_INDEX_PATH, EMBED_MODEL


def main():
    print(f"Loading chunks from: {CHUNKS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks if c.get("text") and c["text"].strip()]

    print(f"Loaded {len(texts)} chunks.")
    print(f"Loading model: {EMBED_MODEL}")

    model = SentenceTransformer(EMBED_MODEL)

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    TEXT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving index to: {TEXT_INDEX_PATH}")

    faiss.write_index(index, str(TEXT_INDEX_PATH))

    print("Text vector database created successfully.")
    print(f"Saved to: {TEXT_INDEX_PATH}")


if __name__ == "__main__":
    main()