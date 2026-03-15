from sentence_transformers import SentenceTransformer
from backend.config import EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL)


def embed(text: str):
    return model.encode([text], convert_to_numpy=True)