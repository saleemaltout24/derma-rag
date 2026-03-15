from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from backend.config import CLIP_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
model.eval()


def embed_image(image_path: str):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output
        features = model.visual_projection(pooled_output)

    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")