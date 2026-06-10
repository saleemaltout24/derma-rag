"""
Shared evaluation utilities for DermaRAG.
================================================
Runs the full RAG pipeline once over the eval set and caches the generations
to ``processed/eval_predictions.json``. Both evaluators reuse this cache so the
LLM-as-Judge (``evaluate_rag_faithfulness.py``) and RAGAS
(``evaluate_rag_ragas.py``) score the *same* answers without re-running the
(slow) pipeline twice.

Each prediction record has the shape:
    {
        "id": "en_01",
        "language": "en",
        "question": "...",
        "ground_truth": "...",          # reference answer (may be "")
        "expected_keywords": [...],
        "answer": "...",                # generated answer
        "contexts": ["...", "..."],     # retrieved chunks, one string each
        "context_str": "...",           # chunks joined (for the LLM judge)
        "response_time_s": 1.23,
        "no_answer": false              # pipeline returned a refusal/clarification
    }
"""

import json
import sys
import time
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.rag_pipeline import answer_medical_question
from backend.state_manager import create_empty_state

EVAL_PATH = ROOT / "data" / "eval_rag_quality.json"
PREDICTIONS_PATH = ROOT / "processed" / "eval_predictions.json"

# Per-chunk truncation used when joining context for the LLM judge (keeps the
# judge prompt small and matches the original faithfulness behaviour).
_JUDGE_CHUNK_CHARS = 400

# Phrases that mean "pipeline found nothing" — used to flag no-answer cases.
# LEAD phrases: must appear in the first 200 chars OR the answer must be short
# (<250 chars) to count as a true no-answer, so a trailing disclaimer on a real
# answer does not wrongly flag it.
_NO_ANSWER_PHRASES = (
    "i don't have enough information",
    "yeterli bilgi",
    "tıbbi kaynaklarda",
    "to help properly i need",
    "where on your body",
    "hangi bölgesinde",
)
# Phrases that are no-answer regardless of position (hard refusals).
_HARD_REFUSAL_PHRASES = (
    "to help properly i need",
    "where on your body",
    "hangi bölgesinde",
)


def is_no_answer(text: str) -> bool:
    """True when the pipeline refused or asked for clarification instead of answering."""
    t = (text or "").lower().strip()
    if any(phrase in t for phrase in _HARD_REFUSAL_PHRASES):
        return True
    soft_phrases = [p for p in _NO_ANSWER_PHRASES if p not in _HARD_REFUSAL_PHRASES]
    for phrase in soft_phrases:
        if phrase in t:
            if len(t) < 250 or t.index(phrase) < 200:
                return True
    return False


def extract_contexts(debug: dict) -> list[str]:
    """Return retrieved chunks as a list of strings (one per source) for RAGAS."""
    sources = debug.get("selected_sources") or []
    contexts = []
    for s in sources:
        text = (s.get("text") or "").strip()
        section = (s.get("section_title") or "").strip()
        if text:
            contexts.append(f"[{section}] {text}" if section else text)
    return contexts


def build_context_str(contexts: list[str]) -> str:
    """Join retrieved chunks into one block (truncated) for the LLM judge prompt."""
    return "\n\n".join(c[:_JUDGE_CHUNK_CHARS] for c in contexts)


def load_eval_cases() -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Eval file not found at {EVAL_PATH}")
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_prediction(case: dict, verbose: bool) -> dict:
    qid = case["id"]
    question = case["question"]
    lang = case.get("language", "en")

    if verbose:
        print(f"  running pipeline for {qid}: {question}")

    t_start = time.perf_counter()
    try:
        answer, _state, debug = answer_medical_question(
            question=question,
            history=[],
            current_state=create_empty_state(),
            forced_language=lang,
        )
    except Exception as e:  # noqa: BLE001 — record failure, keep evaluating others
        print(f"  [pipeline error] {qid}: {e}")
        answer, debug = "", {}
    response_time_s = round(time.perf_counter() - t_start, 2)

    contexts = extract_contexts(debug)
    return {
        "id": qid,
        "language": lang,
        "question": question,
        "ground_truth": case.get("ground_truth", ""),
        "expected_keywords": case.get("expected_keywords", []),
        "answer": answer,
        "contexts": contexts,
        "context_str": build_context_str(contexts),
        "response_time_s": response_time_s,
        "no_answer": is_no_answer(answer),
    }


def _cache_is_valid(cached: dict, cases: list[dict]) -> bool:
    if not isinstance(cached, dict):
        return False
    want_ids = [c["id"] for c in cases]
    have_ids = [p.get("id") for p in cached.get("predictions", [])]
    return want_ids == have_ids


def generate_predictions(use_cache: bool = True, verbose: bool = True) -> list[dict]:
    """
    Run the RAG pipeline over every eval case and return prediction records.

    When ``use_cache`` is True and a valid cache exists (same set of case ids),
    the cached generations are reused instead of re-running the pipeline.
    """
    cases = load_eval_cases()

    if use_cache and PREDICTIONS_PATH.exists():
        try:
            with open(PREDICTIONS_PATH, encoding="utf-8") as f:
                cached = json.load(f)
            if _cache_is_valid(cached, cases):
                if verbose:
                    print(f"Using cached predictions → {PREDICTIONS_PATH} "
                          f"({len(cached['predictions'])} cases)")
                    print("  (delete this file or pass --refresh to regenerate)")
                return cached["predictions"]
            if verbose:
                print("Cached predictions are stale (eval set changed) — regenerating.")
        except (json.JSONDecodeError, OSError):
            if verbose:
                print("Could not read prediction cache — regenerating.")

    if verbose:
        print("=" * 60)
        print(f"  Generating predictions for {len(cases)} eval cases")
        print("  (this runs the full RAG pipeline — Ollama must be running)")
        print("=" * 60)

    predictions = [_build_prediction(case, verbose) for case in cases]

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\nPredictions cached → {PREDICTIONS_PATH}")

    return predictions


def parse_refresh_flag(argv: list[str]) -> bool:
    """Return True (i.e. ignore cache) if --refresh / --no-cache is passed."""
    return any(arg in ("--refresh", "--no-cache") for arg in argv)
