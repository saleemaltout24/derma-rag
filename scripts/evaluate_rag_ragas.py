"""
RAGAS Evaluation — industry-standard RAG metrics
================================================
Scores the DermaRAG pipeline using the RAGAS library, driven entirely by your
LOCAL stack: the Ollama chat model (as the judge LLM) and the same HuggingFace
sentence-transformer used for retrieval (as the embedding model). No data
leaves the machine and no API keys are needed.

Metrics (all 0.0–1.0, higher is better):
    Faithfulness          — answer claims are supported by retrieved context
    AnswerRelevancy       — answer is on-topic for the question (needs embeddings)
    ContextPrecision      — retrieved chunks are relevant, ranked well (uses reference)
    ContextRecall         — retrieved chunks cover the reference answer
    FactualCorrectness    — answer agrees with the reference (ground_truth)
    SemanticSimilarity    — answer is semantically close to the reference (embeddings)

Usage (from project root, venv activated, Ollama running):
    python scripts/evaluate_rag_ragas.py
    python scripts/evaluate_rag_ragas.py --refresh     # ignore the prediction cache

Output:
    processed/ragas_report.json    — per-question scores + averages
    processed/ragas_summary.txt    — printable table for your report
    processed/ragas_scores.csv     — raw per-row scores from RAGAS
"""

import json
import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _eval_common import generate_predictions, parse_refresh_flag  # noqa: E402
from backend.config import CHAT_MODEL, EMBED_MODEL, LLM_NUM_CTX  # noqa: E402

OUT_JSON = ROOT / "processed" / "ragas_report.json"
OUT_TXT = ROOT / "processed" / "ragas_summary.txt"
OUT_CSV = ROOT / "processed" / "ragas_scores.csv"

# Placeholder used when the pipeline retrieved nothing, so RAGAS context metrics
# can still run instead of crashing on an empty context list.
_EMPTY_CONTEXT = "(no relevant context was retrieved)"


def _patch_langchain_vertexai_stub() -> None:
    """
    Work around a hard import in some RAGAS builds.

    ``ragas.llms.base`` does ``from langchain_community.chat_models.vertexai
    import ChatVertexAI``, but newer ``langchain-community`` removed that module
    path (VertexAI moved to ``langchain-google-vertexai``). We never use
    VertexAI (only ChatOllama), so register a lightweight stub for that path
    before importing ragas, instead of pinning the whole langchain stack.
    """
    import types

    mod_path = "langchain_community.chat_models.vertexai"
    if mod_path in sys.modules:
        return
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return  # real module exists; no stub needed
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(mod_path)

    class ChatVertexAI:  # placeholder; never instantiated in this project
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ChatVertexAI stub: VertexAI is not used by DermaRAG.")

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[mod_path] = stub


def _import_ragas():
    """Import RAGAS lazily so the rest of the toolchain works without it installed."""
    _patch_langchain_vertexai_stub()
    try:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            FactualCorrectness,
            Faithfulness,
            ResponseRelevancy,
            SemanticSimilarity,
        )
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("ragas"):
            print("ERROR: RAGAS is not installed.")
            print("Install it with:")
            print("    pip install ragas langchain-ollama langchain-huggingface datasets")
        else:
            print(f"ERROR: RAGAS failed to import due to a missing dependency: {e.name}")
            print("This is usually a langchain version mismatch. Try:")
            print("    pip install -U ragas langchain langchain-community "
                  "langchain-ollama langchain-huggingface")
        print(f"\n(import error: {e})")
        sys.exit(1)
    except ImportError as e:
        print(f"ERROR: RAGAS failed to import: {e}")
        print("Likely a langchain version mismatch. Try reinstalling:")
        print("    pip install -U ragas langchain langchain-community "
              "langchain-ollama langchain-huggingface")
        sys.exit(1)

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        print("ERROR: langchain-ollama not installed.")
        print("    pip install langchain-ollama")
        sys.exit(1)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        print("ERROR: langchain-huggingface not installed.")
        print("    pip install langchain-huggingface")
        sys.exit(1)

    return {
        "EvaluationDataset": EvaluationDataset,
        "RunConfig": RunConfig,
        "evaluate": evaluate,
        "LangchainEmbeddingsWrapper": LangchainEmbeddingsWrapper,
        "LangchainLLMWrapper": LangchainLLMWrapper,
        "ChatOllama": ChatOllama,
        "HuggingFaceEmbeddings": HuggingFaceEmbeddings,
        "metrics": {
            "ContextPrecision": ContextPrecision,
            "ContextRecall": ContextRecall,
            "FactualCorrectness": FactualCorrectness,
            "Faithfulness": Faithfulness,
            "ResponseRelevancy": ResponseRelevancy,
            "SemanticSimilarity": SemanticSimilarity,
        },
    }


def _build_dataset(predictions: list[dict], EvaluationDataset):
    rows = []
    for p in predictions:
        contexts = p.get("contexts") or [_EMPTY_CONTEXT]
        rows.append(
            {
                "user_input": p["question"],
                "retrieved_contexts": contexts,
                "response": p["answer"] or "(no answer produced)",
                "reference": p.get("ground_truth") or "",
            }
        )
    return EvaluationDataset.from_list(rows), rows


def _safe_avg(values: list) -> float:
    nums = [v for v in values if isinstance(v, (int, float)) and v == v]  # drop NaN
    return round(sum(nums) / len(nums), 3) if nums else float("nan")


def run_evaluation() -> None:
    refresh = parse_refresh_flag(sys.argv)

    print("=" * 60)
    print("  DermaRAG — RAGAS Evaluation (local Ollama + HF embeddings)")
    print("=" * 60)
    print(f"  Judge LLM:   {CHAT_MODEL}")
    print(f"  Embeddings:  {EMBED_MODEL}")

    r = _import_ragas()

    # ── 1. predictions (shared cache with the LLM-as-Judge evaluator) ──────────
    predictions = generate_predictions(use_cache=not refresh)
    eval_dataset, rows = _build_dataset(predictions, r["EvaluationDataset"])

    # ── 2. wrap local models for RAGAS ─────────────────────────────────────────
    judge_llm = r["LangchainLLMWrapper"](
        r["ChatOllama"](model=CHAT_MODEL, temperature=0.0, num_ctx=LLM_NUM_CTX)
    )
    judge_embeddings = r["LangchainEmbeddingsWrapper"](
        r["HuggingFaceEmbeddings"](model_name=EMBED_MODEL)
    )

    m = r["metrics"]
    metrics = [
        m["Faithfulness"](),
        m["ResponseRelevancy"](),
        m["ContextPrecision"](),
        m["ContextRecall"](),
        m["FactualCorrectness"](),
        m["SemanticSimilarity"](),
    ]

    # Local models are slow; give each LLM call generous time and avoid hammering.
    run_config = r["RunConfig"](timeout=600, max_workers=1, max_retries=2)

    print("\nRunning RAGAS metrics (this can take a while on a local LLM)...\n")
    result = r["evaluate"](
        dataset=eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    # ── 3. collect per-row + average scores ────────────────────────────────────
    df = result.to_pandas()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    metric_cols = [c for c in df.columns if c not in
                   ("user_input", "retrieved_contexts", "response", "reference")]

    per_case = []
    for pred, (_, drow) in zip(predictions, df.iterrows()):
        scores = {col: (float(drow[col]) if drow[col] == drow[col] else None)
                  for col in metric_cols}
        per_case.append({
            "id": pred["id"],
            "language": pred["language"],
            "question": pred["question"],
            "no_answer": pred["no_answer"],
            "scores": scores,
        })

    averages = {col: _safe_avg([row["scores"].get(col) for row in per_case])
                for col in metric_cols}

    report = {
        "judge_llm": CHAT_MODEL,
        "embeddings": EMBED_MODEL,
        "n": len(per_case),
        "metrics": metric_cols,
        "averages": averages,
        "cases": per_case,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved → {OUT_JSON}")
    print(f"Raw scores saved  → {OUT_CSV}")

    _write_summary(per_case, averages, metric_cols)


def _write_summary(per_case: list[dict], averages: dict, metric_cols: list[str]) -> None:
    # short header labels keep the table readable
    short = {
        "faithfulness": "Faithful",
        "answer_relevancy": "AnsRelev",
        "context_precision": "CtxPrec",
        "context_recall": "CtxRecall",
        "factual_correctness": "FactCorr",
        "factual_correctness(mode=f1)": "FactCorr",
        "semantic_similarity": "SemSim",
    }
    headers = [short.get(c, c[:9]) for c in metric_cols]

    lines = []
    head = f"| {'ID':<6} | " + " | ".join(f"{h:^9}" for h in headers) + " |"
    sep = "+" + "-" * (len(head) - 2) + "+"
    lines.append(sep)
    lines.append(head)
    lines.append(sep)
    for row in per_case:
        flag = "!" if row["no_answer"] else " "
        cells = []
        for col in metric_cols:
            v = row["scores"].get(col)
            cells.append(f"{v:^9.2f}" if isinstance(v, (int, float)) else f"{'-':^9}")
        lines.append(f"|{flag}{row['id']:<5} | " + " | ".join(cells) + " |")
    lines.append(sep)
    avg_cells = []
    for col in metric_cols:
        v = averages.get(col)
        avg_cells.append(f"{v:^9.2f}" if isinstance(v, (int, float)) and v == v else f"{'-':^9}")
    lines.append(f"| {'AVG':<5} | " + " | ".join(avg_cells) + " |")
    lines.append(sep)
    lines.append("  ! = pipeline returned a refusal / clarification (no real answer)")

    table = "\n".join(lines)
    print("\n" + table)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("DermaRAG — RAGAS Evaluation Results\n")
        f.write("Evaluation method: RAGAS library (local Ollama judge + HF embeddings)\n")
        n = len(per_case)
        en = sum(1 for r in per_case if r["language"] == "en")
        tr = sum(1 for r in per_case if r["language"] == "tr")
        f.write(f"Questions evaluated: {n} ({en} EN, {tr} TR)\n")
        f.write(f"No-answer cases:     {sum(1 for r in per_case if r['no_answer'])}\n\n")
        f.write(table + "\n\n")
        f.write("Metric definitions (RAGAS):\n")
        f.write("  Faithfulness        — claims in the answer are grounded in retrieved context\n")
        f.write("  AnswerRelevancy     — answer is relevant to the question (embedding-based)\n")
        f.write("  ContextPrecision    — relevant chunks are ranked above irrelevant ones\n")
        f.write("  ContextRecall       — retrieved context covers the reference answer\n")
        f.write("  FactualCorrectness  — answer agrees with the reference (ground_truth)\n")
        f.write("  SemanticSimilarity  — answer is semantically close to the reference\n")

    print(f"\nSummary saved → {OUT_TXT}")
    print(f"\n{'=' * 60}")
    print(f"  RAGAS AVERAGES (n={len(per_case)})")
    for col in metric_cols:
        v = averages.get(col)
        shown = f"{v:.2f}" if isinstance(v, (int, float)) and v == v else "n/a"
        print(f"  {col:<28} {shown}")
    print(f"{'=' * 60}")
    print("\n  Target benchmarks:  >= 0.70 acceptable   >= 0.80 good   >= 0.85 strong")


if __name__ == "__main__":
    run_evaluation()
