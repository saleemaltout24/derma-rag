import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from backend.config import CLIP_MODEL

print("[ImageEmbedder] Loading CLIP model at startup...")
_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
_model = CLIPModel.from_pretrained(CLIP_MODEL)
_model.eval()
print("[ImageEmbedder] CLIP model ready.")

def embed_image(image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = _processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            outputs = _model.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            features = _model.visual_projection(pooled)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype("float32")
    except Exception as e:
        print(f"[ImageEmbedder] Error: {e}")
        return None
