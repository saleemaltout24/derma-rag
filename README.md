# Derma-RAG

Local-first dermatology assistant with:
- Textbook-grounded RAG over dermatology PDFs
- Optional image similarity against textbook figures
- Bilingual English/Turkish conversation
- Ollama-backed chat and optional vision support

This is an educational support tool and not a substitute for clinical care.

## Tech stack

- Backend: FastAPI + Uvicorn
- Retrieval: sentence-transformers + FAISS
- Image pipeline: CLIP + FAISS + PyMuPDF/OpenCV extraction
- LLM serving: Ollama
- Frontend: React + Vite + Tailwind

## Prerequisites

- Python 3.10+
- Node.js 20+
- Ollama running locally

## Quickstart

### 1) Python environment

Windows PowerShell:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

```bash
copy .env.example .env
```

Update `.env` if needed (especially model tags and paths).
You can tune retrieval via `ENABLE_RERANK`, `RETRIEVE_TOP_K`, and `RERANK_TOP_K`.
Optional persistence is available with `USE_SQLITE_SESSIONS=true`.
Use `DEBUG_PAYLOADS=false` to hide retrieval internals from API responses.

### 3) Pull Ollama models

```bash
ollama pull mistral:7b
ollama pull llama3.2-vision
```

### 4) Build local data artifacts

Place textbooks under `data/textbooks/*.pdf`, then run:

```bash
python scripts/process_books.py
python scripts/create_embeddings.py
python scripts/extract_textbook_images.py
python scripts/create_image_embeddings.py
```

### 5) Start backend

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 6) Start frontend

```bash
cd frontend
npm install
npm run dev
```

UI default: `http://127.0.0.1:5173`  
API default: `http://127.0.0.1:8000`

## API endpoints

- `POST /chat` (multipart form: `session_id`, `question`, optional `file`)
- `POST /reset` (`session_id`)
- `GET /ask` (`question`, `session_id`)
- `GET /health`

`/chat` and `/ask` responses include `retrieval_debug` for retrieval/rerank traceability.

## Important notes

- FAISS indexes depend on the embedding model. If `EMBEDDING_MODEL` changes, rebuild indexes.
- Chunks now include `page_start/page_end`; rerun ingestion before rebuilding embeddings after schema changes.
- Uploads are stored temporarily and removed after each request.
- Optional System B classifier contract is exposed in API responses as `classifier_result`.
- Optional ONNX classifier runtime:
  - set `ENABLE_CLASSIFIER=true`
  - set `CLASSIFIER_MODEL_PATH=/path/to/model.onnx`
  - optional class names with `CLASSIFIER_LABELS=eczema,psoriasis,acne,...`
- Retrieval eval scaffold:
  - `python scripts/evaluate_retrieval.py`
  - optional input file: `python scripts/evaluate_retrieval.py data/eval_queries.sample.json`
- End-to-end text eval scaffold:
  - `python scripts/evaluate_end_to_end.py`
  - custom API/cases: `python scripts/evaluate_end_to_end.py http://127.0.0.1:8000 data/eval_end_to_end.sample.json`
