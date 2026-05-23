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
# FAISS: wide recall; cross-encoder rerank: narrow precision. If k is too small, the
# right passage may never enter the rerank pool (no amount of reranking fixes that).
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
# Multimodal (image upload) uses smaller retrieval by default for speed.
MULTIMODAL_RETRIEVE_TOP_K = int(os.getenv("MULTIMODAL_RETRIEVE_TOP_K", "12"))
MULTIMODAL_RERANK_TOP_K = int(os.getenv("MULTIMODAL_RERANK_TOP_K", "3"))
MULTIMODAL_MAX_HISTORY_TURNS = int(os.getenv("MULTIMODAL_MAX_HISTORY_TURNS", "4"))
MAX_CONTEXT_CHARS_PER_DOC = int(os.getenv("MAX_CONTEXT_CHARS_PER_DOC", "600"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("MAX_TOTAL_CONTEXT_CHARS", "3500"))
MAX_NEARBY_TEXT_CHARS = int(os.getenv("MAX_NEARBY_TEXT_CHARS", "400"))
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
# Skip extra LLM call for structured state on image uploads (big speed win).
SKIP_STATE_LLM_ON_IMAGE = os.getenv("SKIP_STATE_LLM_ON_IMAGE", "true").lower() == "true"
# Classifier TTA views: 4 = accurate, 1 = faster inference.
CLASSIFIER_TTA_VIEWS = max(1, min(4, int(os.getenv("CLASSIFIER_TTA_VIEWS", "4"))))
# Flag similar-looking classes (e.g. AK vs SCC) when scores are close.
CONFUSION_MARGIN_THRESHOLD = float(os.getenv("CONFUSION_MARGIN_THRESHOLD", "25"))
# Run Grad-CAM while the main LLM generates sections 2–4.
PARALLEL_GRADCAM = os.getenv("PARALLEL_GRADCAM", "true").lower() == "true"
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "450"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))
ENABLE_CLASSIFIER = os.getenv("ENABLE_CLASSIFIER", "false").lower() == "true"
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "")
CLASSIFIER_LABELS = [label.strip() for label in os.getenv("CLASSIFIER_LABELS", "").split(",") if label.strip()]
USE_SQLITE_SESSIONS = os.getenv("USE_SQLITE_SESSIONS", "false").lower() == "true"
SESSION_DB_PATH = Path(os.getenv("SESSION_DB_PATH", str(DATA_DIR / "sessions.db")))
DEBUG_PAYLOADS = os.getenv("DEBUG_PAYLOADS", "false").lower() == "true"
ENABLE_VISION_LLM = os.getenv("ENABLE_VISION_LLM", "false").lower() == "true"
DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

EMBEDDING_MODEL = EMBED_MODEL

DATA_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)
TEXTBOOK_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
