import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer
import torch

from backend.config import EMBED_MODEL, HARDWARE_PROFILE

if HARDWARE_PROFILE == "gpu" and torch.cuda.is_available():
    MODEL_DEVICE = "cuda"
else:
    MODEL_DEVICE = "cpu"

model = SentenceTransformer(EMBED_MODEL, device=MODEL_DEVICE)

def embed(text: str):
    return model.encode(
        [text],
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=1
    )
