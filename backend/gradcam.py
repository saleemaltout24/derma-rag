import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
import io

from backend.skin_classifier import (
    INPUT_SIZE,
    get_skin_classifier_model,
    letterbox_rgb,
    load_image_rgb_exif,
    transform,
)


def generate_gradcam(image_path: str, class_idx: int = None) -> str | None:
    model, err = get_skin_classifier_model()
    if model is None:
        print(f"[GradCAM] Skipped: {err}")
        return None
    try:
        original = load_image_rgb_exif(image_path)
        boxed = letterbox_rgb(original)
        input_tensor = transform(boxed).unsqueeze(0)

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        target_layer = model.features[-1]
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        output = model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        model.zero_grad()
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
        cam_img = cam_img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        cam_np = np.array(cam_img)

        heatmap = np.zeros((*cam_np.shape, 3), dtype=np.uint8)
        heatmap[:, :, 0] = cam_np
        heatmap[:, :, 1] = (255 - cam_np) // 2
        heatmap[:, :, 2] = 0

        # Overlay on letterboxed view (matches classifier input geometry)
        display_np = np.array(boxed)
        overlay = (0.6 * display_np + 0.4 * heatmap).astype(np.uint8)
        overlay_img = Image.fromarray(overlay)

        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        print(f"[GradCAM] Error: {e}")
        return None
