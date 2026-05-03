import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
from backend.config import EMBED_MODEL

model = SentenceTransformer(EMBED_MODEL, device="cpu")

def embed(text: str):
    return model.encode(
        [text],
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=1
    )