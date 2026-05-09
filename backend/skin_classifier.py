import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevi',
    'BCC': 'Basal Cell Carcinoma',
    'AK': 'Actinic Keratosis',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular Lesions',
    'SCC': 'Squamous Cell Carcinoma'
}

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "skin_classifier_v2.pth"
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("[Classifier] Loading skin disease classifier v2...")
_model = models.efficientnet_b0(weights=None)
_model.classifier[1] = nn.Linear(_model.classifier[1].in_features, len(CLASSES))
checkpoint = torch.load(MODEL_PATH, map_location=device)
_model.load_state_dict(checkpoint['model_state_dict'])
_model.eval()
print("[Classifier] Classifier v2 ready!")


def classify_skin_image(image_path: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = _model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        results = []
        for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
            results.append({
                "code": cls,
                "name": CLASS_NAMES[cls],
                "confidence": round(float(prob) * 100, 2)
            })

        results.sort(key=lambda x: x["confidence"], reverse=True)
        top = results[0]

        return {
            "predicted_class": top["code"],
            "predicted_name": top["name"],
            "confidence": top["confidence"],
            "all_predictions": results
        }

    except Exception as e:
        print(f"[Classifier] Error: {e}")
        return {
            "predicted_class": "UNKNOWN",
            "predicted_name": "Unknown",
            "confidence": 0.0,
            "all_predictions": [],
            "error": str(e)
        }
