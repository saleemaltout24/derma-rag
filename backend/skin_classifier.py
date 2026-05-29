import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
from pathlib import Path

from backend.config import CLASSIFIER_TTA_VIEWS, CONFUSION_MARGIN_THRESHOLD

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

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_V3 = _MODELS_DIR / "skin_classifier_v3.pth"
_V2 = _MODELS_DIR / "skin_classifier_v2.pth"
_DEFAULT_WEIGHTS = _V3 if _V3.is_file() else _V2
MODEL_PATH = Path(os.getenv("SKIN_CLASSIFIER_PATH", str(_DEFAULT_WEIGHTS)))
device = torch.device("cpu")

# Classes that often look alike on dermoscopy (model confuses them).
CONFUSION_PAIRS = {
    frozenset({"AK", "SCC"}),
    frozenset({"AK", "BKL"}),
    frozenset({"BCC", "SCC"}),
    frozenset({"VASC", "BKL"}),
    frozenset({"MEL", "NV"}),
}

# ImageNet normalize after letterbox (same for classifier + GradCAM).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ]
)

_model: nn.Module | None = None
_unavailable_reason: str | None = None


def load_image_rgb_exif(image_path: str) -> Image.Image:
    """Open RGB and apply EXIF orientation."""
    img = Image.open(image_path).convert("RGB")
    return ImageOps.exif_transpose(img)


def letterbox_rgb(pil: Image.Image, size: int = INPUT_SIZE) -> Image.Image:
    """Resize to fit inside size×size, pad to square (keeps aspect ratio — good for dermoscopy)."""
    w, h = pil.size
    scale = size / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def pil_to_model_tensor(pil: Image.Image) -> torch.Tensor:
    """Letterbox + ImageNet normalize → CHW tensor."""
    return transform(letterbox_rgb(pil))


def load_model_tensor(image_path: str) -> torch.Tensor:
    """Full path: file → EXIF fix → letterbox → tensor."""
    return pil_to_model_tensor(load_image_rgb_exif(image_path))


def _tta_pil_variants(pil: Image.Image) -> list[Image.Image]:
    """TTA views: 1 = original only; 4 = original + flips (slower, slightly more accurate)."""
    variants = [
        pil,
        pil.transpose(Image.FLIP_LEFT_RIGHT),
        pil.transpose(Image.FLIP_TOP_BOTTOM),
        pil.transpose(Image.ROTATE_180),
    ]
    return variants[:CLASSIFIER_TTA_VIEWS]


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
        label = MODEL_PATH.name
        print(f"[Classifier] Loading weights: {MODEL_PATH}")
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        _model = m
        val_acc = checkpoint.get("val_accuracy") if isinstance(checkpoint, dict) else None
        extra = f" (checkpoint val acc {val_acc:.2f}%)" if val_acc is not None else ""
        print(f"[Classifier] Ready: {label}{extra}")
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


def _score_by_code(results: list[dict]) -> dict[str, float]:
    return {r["code"]: float(r["confidence"]) for r in results}


def _confusion_pair_fields(results: list[dict], top: dict, second: dict, margin: float) -> dict:
    """Mark clinically similar classes when scores are close (e.g. AK mistaken for SCC)."""
    top_code = top["code"]
    second_code = second["code"]
    scores = _score_by_code(results)
    extra: dict = {}

    ak = scores.get("AK", 0.0)
    scc = scores.get("SCC", 0.0)
    ak_scc_gap = abs(ak - scc)
    if ak > 0 and scc > 0 and ak_scc_gap < CONFUSION_MARGIN_THRESHOLD:
        extra["confusion_pair"] = ["AK", "SCC"]
        extra["confusion_clinical_note"] = (
            "Actinic keratosis (AK) and squamous cell carcinoma (SCC) often look very similar. "
            "The AI frequently confuses them. If the lesion may be AK, do not treat SCC as certain — "
            "a dermatologist can tell them apart; biopsy is sometimes needed."
        )
        extra["ambiguous"] = True
        if top_code == "SCC" and ak >= 10.0:
            extra["ak_alternative_likely"] = True
        elif top_code == "AK" and scc >= 10.0:
            extra["scc_alternative_likely"] = True
        return extra

    # Model very confident on SCC but AK score tiny — still flag for demos / clinical AK photos.
    if top_code == "SCC" and scc >= 55.0 and ak < 15.0:
        extra["confusion_clinical_note"] = (
            f"The model is confident about squamous cell carcinoma ({scc:.1f}%) but gives actinic keratosis "
            f"only {ak:.1f}%. AK and SCC are related (AK is sun damage that can sometimes progress to SCC) "
            f"and often look alike — the AI often calls AK lesions SCC. If this lesion is clinically actinic "
            f"keratosis, trust examination or biopsy over this score."
        )
        extra["ak_scc_overcall_warning"] = True
        if extra.get("confidence_tier") == "high":
            extra["confidence_tier"] = "medium"
        return extra

    pair = frozenset({top_code, second_code})
    if pair in CONFUSION_PAIRS and margin < CONFUSION_MARGIN_THRESHOLD:
        other_name = CLASS_NAMES.get(second_code, second_code)
        extra["confusion_pair"] = [top_code, second_code]
        extra["confusion_clinical_note"] = (
            f"{top['name']} and {other_name} can look similar on dermoscopy. "
            f"The margin between them is only {margin:.1f}% — consider both."
        )
        extra["ambiguous"] = True
    return extra


def classify_skin_image(image_path: str) -> dict:
    model, err = get_skin_classifier_model()
    if model is None:
        return {
            "predicted_class": "UNAVAILABLE",
            "predicted_class_index": -1,
            "predicted_name": "Classifier weights not loaded",
            "confidence": 0.0,
            "all_predictions": [],
            "error": err,
            **_uncertainty_fields(),
        }
    try:
        pil = letterbox_rgb(load_image_rgb_exif(image_path))
        tensors = [transform(v) for v in _tta_pil_variants(pil)]
        batch = torch.stack(tensors, dim=0).to(device)

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
        extra.update(_confusion_pair_fields(results, top, second, margin))
        if extra.get("ambiguous"):
            extra["confidence_tier"] = "low" if margin < 15.0 else extra["confidence_tier"]
            if extra.get("confusion_clinical_note") and extra["confidence_tier"] == "high":
                extra["confidence_tier"] = "medium"

        return {
            "predicted_class": top["code"],
            "predicted_class_index": CLASSES.index(top["code"]),
            "predicted_name": top["name"],
            "confidence": top["confidence"],
            "all_predictions": results,
            **extra,
        }

    except Exception as e:
        print(f"[Classifier] Error: {e}")
        return {
            "predicted_class": "UNKNOWN",
            "predicted_class_index": -1,
            "predicted_name": "Unknown",
            "confidence": 0.0,
            "all_predictions": [],
            "error": str(e),
            **_uncertainty_fields(),
        }
