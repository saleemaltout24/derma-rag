import json
from pathlib import Path

import faiss
import numpy as np

from backend.config import CHUNKS_PATH, TEXT_INDEX_PATH
from backend.embedder import embed

index = None
chunks = []
load_error = None


def _load_resources() -> None:
    global index, chunks, load_error
    if index is not None and chunks:
        return

    if not Path(TEXT_INDEX_PATH).exists():
        load_error = (
            f"Text index is missing at '{TEXT_INDEX_PATH}'. "
            "Run: python scripts/process_books.py && python scripts/create_embeddings.py"
        )
        return

    if not Path(CHUNKS_PATH).exists():
        load_error = (
            f"Chunks file is missing at '{CHUNKS_PATH}'. "
            "Run: python scripts/process_books.py"
        )
        return

    try:
        index = faiss.read_index(str(TEXT_INDEX_PATH))
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        load_error = None
    except Exception as exc:
        load_error = f"Failed to load vector store resources: {exc}"


def search(query: str, k: int = 4):
    _load_resources()
    if load_error:
        raise RuntimeError(load_error)

    q = embed(query)
    q = np.array(q, dtype=np.float32)

    distances, indices = index.search(q, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        item = dict(chunks[idx])
        dist = float(distances[0][rank])
        item["score"] = dist
        item["distance"] = dist
        results.append(item)

    return results
