import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "processed"

TEXTBOOK_DIR = DATA_DIR / "textbooks"
TEXTBOOK_IMAGE_DIR = DATA_DIR / "textbook_images"
UPLOAD_DIR = DATA_DIR / "uploads"

DEFAULT_VECTORSTORE_DIR = DATA_DIR / "vectorstore"
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(DEFAULT_VECTORSTORE_DIR)))

CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", str(PROCESSED_DIR / "chunks.json")))
IMAGE_METADATA_PATH = Path(
    os.getenv("IMAGE_METADATA_PATH", str(PROCESSED_DIR / "image_metadata.json"))
)

TEXT_INDEX_PATH = VECTORSTORE_DIR / "faiss_index"
IMAGE_INDEX_PATH = VECTORSTORE_DIR / "image_faiss_index"
TEXT_INDEX_MANIFEST_PATH = VECTORSTORE_DIR / "faiss_index_manifest.json"

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral:7b")
VISION_MODEL = os.getenv("VISION_MODEL", "llama3.2-vision")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
HARDWARE_PROFILE = os.getenv("HARDWARE_PROFILE", "cpu").lower()
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "6"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
ENABLE_CLASSIFIER = os.getenv("ENABLE_CLASSIFIER", "false").lower() == "true"
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "")
CLASSIFIER_LABELS = [label.strip() for label in os.getenv("CLASSIFIER_LABELS", "").split(",") if label.strip()]
USE_SQLITE_SESSIONS = os.getenv("USE_SQLITE_SESSIONS", "false").lower() == "true"
SESSION_DB_PATH = Path(os.getenv("SESSION_DB_PATH", str(DATA_DIR / "sessions.db")))
DEBUG_PAYLOADS = os.getenv("DEBUG_PAYLOADS", "true").lower() == "true"

EMBEDDING_MODEL = EMBED_MODEL

DATA_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
