import os
from ollama import chat
from backend.config import VISION_MODEL


def analyze_skin_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        return "Vision model error: image file not found."

    prompt = """
Describe this dermatology image only from visible findings.
Do not diagnose yet.

Focus on:
- color
- shape
- borders
- texture
- scale or crust
- distribution if visible
- severity if visible
- image quality limitations

Return a short clinical-style visual description.
"""

    try:
        response = chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path],
                }
            ],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Vision model error: {str(e)}"