import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps
from pathlib import Path

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevi",
    "BCC": "Basal Cell Carcinoma",
    "AK": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesions",
    "SCC": "Squamous Cell Carcinoma",
}

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "skin_classifier_v2.pth"
device = torch.device("cpu")

# Resize + ImageNet normalize; apply to EXIF-corrected RGB PIL images (same for classifier + GradCAM).
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

_model: nn.Module | None = None
_unavailable_reason: str | None = None


def load_image_rgb_exif(image_path: str) -> Image.Image:
    """Open RGB, apply EXIF orientation, then resize/normalize via `transform`."""
    img = Image.open(image_path).convert("RGB")
    return ImageOps.exif_transpose(img)


def _uncertainty_fields() -> dict:
    return {
        "confidence_tier": "low",
        "margin": 0.0,
        "ambiguous": True,
        "top2_name": None,
        "top2_confidence": 0.0,
    }


def get_skin_classifier_model() -> tuple[nn.Module | None, str | None]:
    """Load EfficientNet-B0 once; return (model, None) or (None, error message)."""
    global _model, _unavailable_reason
    if _unavailable_reason is not None:
        return None, _unavailable_reason
    if _model is not None:
        return _model, None
    if not MODEL_PATH.is_file():
        _unavailable_reason = (
            f"Missing weights file: {MODEL_PATH}. "
            "Add skin_classifier_v2.pth under the project models/ folder, or ask your teammate for the checkpoint."
        )
        print(f"[Classifier] {_unavailable_reason}")
        return None, _unavailable_reason
    try:
        print("[Classifier] Loading skin disease classifier v2...")
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        _model = m
        print("[Classifier] Classifier v2 ready.")
        return _model, None
    except Exception as e:
        _unavailable_reason = str(e)
        print(f"[Classifier] Load failed: {e}")
        return None, _unavailable_reason


def _confidence_tier(top_pct: float) -> str:
    if top_pct > 70:
        return "high"
    if top_pct >= 40:
        return "medium"
    return "low"


def classify_skin_image(image_path: str) -> dict:
    model, err = get_skin_classifier_model()
    if model is None:
        return {
            "predicted_class": "UNAVAILABLE",
            "predicted_name": "Classifier weights not loaded",
            "confidence": 0.0,
            "all_predictions": [],
            "error": err,
            **_uncertainty_fields(),
        }
    try:
        pil = load_image_rgb_exif(image_path)
        chw = transform(pil)
        chw_flip = TF.hflip(chw)
        batch = torch.stack([chw, chw_flip], dim=0).to(device)

        with torch.inference_mode():
            logits = model(batch)
            logits_mean = logits.mean(dim=0, keepdim=True)
            probabilities = torch.softmax(logits_mean, dim=1)[0]

        results = []
        for i, (cls, prob) in enumerate(zip(CLASSES, probabilities)):
            results.append(
                {
                    "code": cls,
                    "name": CLASS_NAMES[cls],
                    "confidence": round(float(prob) * 100, 2),
                }
            )

        results.sort(key=lambda x: x["confidence"], reverse=True)
        top = results[0]
        top_pct = top["confidence"]
        second = results[1] if len(results) > 1 else {"name": None, "confidence": 0.0}
        margin = round(top_pct - second["confidence"], 2)

        extra = {
            "confidence_tier": _confidence_tier(top_pct),
            "margin": margin,
            "ambiguous": margin < 15.0,
            "top2_name": second["name"],
            "top2_confidence": second["confidence"] if second["name"] is not None else 0.0,
        }

        return {
            "predicted_class": top["code"],
            "predicted_name": top["name"],
            "confidence": top["confidence"],
            "all_predictions": results,
            **extra,
        }

    except Exception as e:
        print(f"[Classifier] Error: {e}")
        return {
            "predicted_class": "UNKNOWN",
            "predicted_name": "Unknown",
            "confidence": 0.0,
            "all_predictions": [],
            "error": str(e),
            **_uncertainty_fields(),
        }
