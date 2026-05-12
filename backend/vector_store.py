import json
import faiss
import numpy as np

from backend.config import CHUNKS_PATH, TEXT_INDEX_PATH
from backend.embedder import embed

index = faiss.read_index(str(TEXT_INDEX_PATH))

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)


def search(query: str, k: int = 4):
    q = embed(query)
    q = np.array(q, dtype=np.float32)

    distances, indices = index.search(q, k)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        item = dict(chunks[idx])
        item["score"] = float(distances[0][rank])
        results.append(item)

    return results