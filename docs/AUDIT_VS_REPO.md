# External audit vs this repository (May 2026)

This table compares claims from a third-party “Full Code Audit” against the **current** `derma-rag-remote-check` codebase after teammate updates on `main`.

| Audit claim | Accurate today? | Notes |
|-------------|-----------------|-------|
| ~80% blueprint complete | **Mostly** | Core RAG + multimodal + classifier UI are real; page-level citations and retrieval eval scripts are still thin. |
| `requirements.txt` fully pinned | **No** | Package names only, no version pins. |
| `.env.example` exists | **Yes** (added in fix pass) | Was missing before; see repo root `.env.example`. |
| `VECTORSTORE_DIR` env-driven | **Yes** | `backend/config.py` |
| Safe uploads (uuid + allowlist) | **Yes** | `app.py` `build_safe_upload_path` |
| Lazy FAISS + clear errors | **Yes** | `backend/vector_store.py` |
| Ollama via `ollama.chat()` | **Yes** | `backend/llm.py` |
| Cross-encoder reranker | **Yes** | `backend/rag_pipeline.py` `rerank_docs` |
| `page_start` / `page_end` on every chunk | **No** | UI supports pages; `chunks.json` / `process_books.py` do not populate them; `create_embeddings.py` has no manifest. |
| Page-scoped CLIP wired | **Partial** | `collect_source_pages` → `search_similar_images`; only useful when chunks carry page fields. |
| ONNX classifier in production | **No** | `backend/classifier.py` exists but unused; live path is **`skin_classifier_v2.pth`** (PyTorch). |
| `evaluate_retrieval.py` / `evaluate_end_to_end.py` | **No** | Only `scripts/eval_classifier.py` + `data/eval/*/.gitkeep`. |
| Multimodal “System A / System B” blocks | **No** | Prompts use **CLASSIFIER** + **supporting** textbook blocks. |
| Root `README.md` | **Yes** | Setup + classifier + eval_classifier. |
| Root `.gitignore` | **Yes** | Ignores `.env`, `processed/`, `vectorstore/`, uploads, etc. |
| Empty frontend subtitle | **No** | Subtitle present in `App.jsx`. |

## Bugs from audit — status after fix pass

| Issue | Status |
|-------|--------|
| CORS hardcoded `*` | **Fixed** — `CORS_ORIGINS` from env, default localhost:5173 |
| Vision LLM stub only | **Documented + optional** — `ENABLE_VISION_LLM=false` by default; set `true` + pull `VISION_MODEL` to enable |
| `print()` dumps on every request | **Fixed** — gated on `DEBUG_PAYLOADS` |
| `run_llm()` returns `"LLM error: …"` as answer | **Fixed** — raises `LLMError` → HTTP **503** in `app.py` |
| References in chunks | **Addressed earlier** — `strip_trailing_reference_suffix` in `process_books.py` |

## What to highlight in a demo (still valid)

1. Text sources with book name (pages when metadata exists).
2. English / Turkish intent + answers.
3. Image upload: CLIP textbook matches + HAM10000-style classifier (8 classes) + optional Grad-CAM.
4. Structured session state across turns.
5. `DEBUG_PAYLOADS=true` for retrieval debug in API responses.

## Known limitations (honest for report)

- Vision description is a **placeholder** unless `ENABLE_VISION_LLM=true`.
- Classifier is **8 HAM10000 lesion types**, not full dermatology taxonomy.
- Chunk quality depends on re-running `process_books.py` after preprocessing changes.
- Reranker uses **definition-priority** tie-break — strong for “what is X?”, weaker for some treatment queries.
