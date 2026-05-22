import os

import ollama

from backend.config import ENABLE_VISION_LLM, VISION_MODEL

_PLACEHOLDER = (
    "User uploaded a skin image for analysis. "
    "Visual description is disabled (set ENABLE_VISION_LLM=true and run Ollama with "
    f"{VISION_MODEL}). Similar textbook images and the skin classifier provide context."
)

_VISION_PROMPT = (
    "You are assisting a dermatology reference app. Describe only what you see in this "
    "skin image: lesion type, color, borders, surface, approximate size if inferable, "
    "and distribution. Do not give a definitive diagnosis. Keep it concise (under 120 words)."
)


def analyze_skin_image(image_path: str, timeout: int = 120) -> str:
    """
    Optional vision-LLM description for multimodal prompts.

    Default: fast placeholder (CLIP + classifier carry visual evidence).
    Set ENABLE_VISION_LLM=true to call Ollama vision (slower; needs VISION_MODEL pulled).
    """
    if not os.path.exists(image_path):
        return "Image file not found."

    if not ENABLE_VISION_LLM:
        return _PLACEHOLDER

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _VISION_PROMPT,
                    "images": [image_path],
                }
            ],
            options={"temperature": 0.2},
        )
        text = (response.get("message") or {}).get("content", "").strip()
        return text if text else _PLACEHOLDER
    except Exception as e:
        return f"{_PLACEHOLDER} (Vision LLM failed: {e})"
