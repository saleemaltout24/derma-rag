import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from pathlib import Path
import base64
import io

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "skin_classifier_v2.pth"
device = torch.device("cpu")

print("[GradCAM] Loading model v2...")
_model = models.efficientnet_b0(weights=None)
_model.classifier[1] = nn.Linear(_model.classifier[1].in_features, len(CLASSES))
checkpoint = torch.load(MODEL_PATH, map_location=device)
_model.load_state_dict(checkpoint['model_state_dict'])
_model.eval()
print("[GradCAM] Model v2 ready.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def generate_gradcam(image_path: str, class_idx: int = None) -> str:
    try:
        original = Image.open(image_path).convert("RGB")
        input_tensor = transform(original).unsqueeze(0)

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        target_layer = _model.features[-1]
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        output = _model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        _model.zero_grad()
        output[0, class_idx].backward()

        fh.remove()
        bh.remove()

        grads = gradients[0].squeeze(0)
        acts = activations[0].squeeze(0)
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cam.detach().numpy()

        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize(original.size, Image.BILINEAR)
        cam_np = np.array(cam_img)

        heatmap = np.zeros((*cam_np.shape, 3), dtype=np.uint8)
        heatmap[:, :, 0] = cam_np
        heatmap[:, :, 1] = (255 - cam_np) // 2
        heatmap[:, :, 2] = 0

        original_np = np.array(original)
        overlay = (0.6 * original_np + 0.4 * heatmap).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)

        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print(f"[GradCAM] Error: {e}")
        return None
