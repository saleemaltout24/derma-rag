from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TEXTBOOK_DIR = DATA_DIR / "textbooks"
TEXTBOOK_IMAGE_DIR = DATA_DIR / "textbook_images"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"

VECTORSTORE_DIR = Path("C:/derma_rag_vectorstore")
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
IMAGE_METADATA_PATH = PROCESSED_DIR / "image_metadata.json"

TEXT_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"
IMAGE_INDEX_PATH = VECTORSTORE_DIR / "image_faiss_index"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHAT_MODEL = "mistral:7b"
VISION_MODEL = "llama3.2-vision"
CLIP_MODEL = "openai/clip-vit-base-patch32"

DATA_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)