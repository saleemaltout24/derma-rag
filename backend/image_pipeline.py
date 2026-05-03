import os

def analyze_skin_image(image_path: str, timeout: int = 120) -> str:
    """
    Skip slow vision model - use CLIP similarity search instead.
    Returns a placeholder description; actual analysis comes from image_vector_store matches.
    """
    if not os.path.exists(image_path):
        return "Image file not found."
    return "User uploaded a skin image for analysis. Please refer to the similar textbook images below for context."
