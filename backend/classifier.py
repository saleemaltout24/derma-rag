from pathlib import Path

import numpy as np
from PIL import Image

from backend.config import CLASSIFIER_LABELS, CLASSIFIER_MODEL_PATH, ENABLE_CLASSIFIER

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional runtime dependency
    ort = None

_session = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _load_session(model_path: Path):
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(model_path))
    return _session


def _preprocess_image(image_path: str, input_shape) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    # Normalize roughly similar to ImageNet preprocessing.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    if len(input_shape) == 4 and input_shape[-1] == 3:
        return np.expand_dims(arr, axis=0).astype(np.float32)  # NHWC
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0).astype(np.float32)  # NCHW


def run_optional_classifier(image_path: str) -> dict:
    if not ENABLE_CLASSIFIER:
        return {
            "enabled": False,
            "label": None,
            "confidence": None,
            "uncertainty": 1.0,
            "note": "Classifier disabled in this deployment.",
        }

    model_path = Path(CLASSIFIER_MODEL_PATH) if CLASSIFIER_MODEL_PATH else None
    if not model_path or not model_path.exists():
        return {
            "enabled": True,
            "label": None,
            "confidence": None,
            "uncertainty": 1.0,
            "note": "Classifier enabled but model file not found.",
        }

    if ort is None:
        return {
            "enabled": True,
            "label": None,
            "confidence": None,
            "uncertainty": 1.0,
            "note": "onnxruntime is not installed.",
        }

    try:
        session = _load_session(model_path)
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape
        tensor = _preprocess_image(image_path, input_shape)
        output = session.run(None, {input_name: tensor})[0]
        logits = np.array(output[0], dtype=np.float32)
        probs = _softmax(logits)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = CLASSIFIER_LABELS[idx] if idx < len(CLASSIFIER_LABELS) else f"class_{idx}"
        return {
            "enabled": True,
            "label": label,
            "confidence": confidence,
            "uncertainty": float(1.0 - confidence),
            "note": "Classifier prediction computed from ONNX model.",
        }
    except Exception as exc:
        return {
            "enabled": True,
            "label": None,
            "confidence": None,
            "uncertainty": 1.0,
            "note": f"Classifier inference failed: {exc}",
        }
