"""
DermaRAG — Full RAG Evaluation Runner
================================================
Runs BOTH evaluators on the same generated answers:
  1. LLM-as-Judge   (evaluate_rag_faithfulness.py)
  2. RAGAS library  (evaluate_rag_ragas.py)

The RAG pipeline is executed only once; its generations are cached in
``processed/eval_predictions.json`` and reused by both evaluators.

Usage (from project root, venv activated, Ollama running):
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --refresh        # regenerate answers first
    python scripts/evaluate_rag.py --only ragas     # run only RAGAS
    python scripts/evaluate_rag.py --only judge     # run only the LLM-as-Judge
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _eval_common import generate_predictions, parse_refresh_flag  # noqa: E402


def _selected() -> str:
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1].lower()
    return "both"


def main() -> None:
    refresh = parse_refresh_flag(sys.argv)
    which = _selected()

    # Generate (or load cached) predictions once, up front, so both evaluators
    # score identical answers. After this, both run with use_cache=True.
    print("Step 1/2 — preparing predictions")
    generate_predictions(use_cache=not refresh)

    # Strip --refresh so the sub-evaluators reuse the freshly written cache.
    sys.argv = [a for a in sys.argv if a not in ("--refresh", "--no-cache")]

    if which in ("both", "judge"):
        print("\n" + "#" * 60)
        print("#  Running LLM-as-Judge evaluation")
        print("#" * 60)
        import evaluate_rag_faithfulness
        evaluate_rag_faithfulness.run_evaluation()

    if which in ("both", "ragas"):
        print("\n" + "#" * 60)
        print("#  Running RAGAS evaluation")
        print("#" * 60)
        import evaluate_rag_ragas
        evaluate_rag_ragas.run_evaluation()

    print("\nDone. Reports written to the processed/ folder:")
    print("  LLM-as-Judge: processed/rag_eval_report.json + rag_eval_summary.txt")
    print("  RAGAS:        processed/ragas_report.json + ragas_summary.txt + ragas_scores.csv")


if __name__ == "__main__":
    main()
