import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from backend.config import RERANK_TOP_K, RETRIEVE_TOP_K  # noqa: E402
from backend.rag_pipeline import rerank_docs  # noqa: E402
from backend.vector_store import search  # noqa: E402


def normalize_source(value: str | None) -> str:
    return Path(value or "").name.lower()


def evaluate_case(case: dict) -> dict:
    query = case.get("query", "")
    expected_sources = {normalize_source(s) for s in case.get("expected_sources", [])}
    expected_pages = {(normalize_source(s), int(p)) for s, p in case.get("expected_source_pages", [])}

    retrieved = search(query, k=RETRIEVE_TOP_K)
    selected = rerank_docs(query, retrieved, k=RERANK_TOP_K)

    selected_sources = [normalize_source(doc.get("source")) for doc in selected]
    selected_source_pages = set()
    for doc in selected:
        source = normalize_source(doc.get("source"))
        start = doc.get("page_start")
        end = doc.get("page_end")
        if start is None and end is None:
            continue
        if start is None:
            start = end
        if end is None:
            end = start
        for page in range(int(start), int(end) + 1):
            selected_source_pages.add((source, page))

    source_hit = bool(expected_sources & set(selected_sources)) if expected_sources else None
    page_hit = bool(expected_pages & selected_source_pages) if expected_pages else None

    return {
        "query": query,
        "source_hit": source_hit,
        "page_hit": page_hit,
        "selected_sources": selected_sources,
        "selected_source_pages_count": len(selected_source_pages),
    }


def main() -> None:
    default_input = ROOT_DIR / "data" / "eval_queries.sample.json"
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_input

    if not input_path.exists():
        raise FileNotFoundError(f"Eval query file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = [evaluate_case(case) for case in cases]
    source_hits = [r["source_hit"] for r in results if r["source_hit"] is not None]
    page_hits = [r["page_hit"] for r in results if r["page_hit"] is not None]

    summary = {
        "input_file": str(input_path),
        "num_cases": len(results),
        "source_hit_rate": (sum(source_hits) / len(source_hits)) if source_hits else None,
        "page_hit_rate": (sum(page_hits) / len(page_hits)) if page_hits else None,
        "results": results,
    }

    output_path = ROOT_DIR / "processed" / "retrieval_eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluated {len(results)} queries.")
    print(f"Report saved to: {output_path}")
    print(json.dumps({k: summary[k] for k in ['source_hit_rate', 'page_hit_rate']}, indent=2))


if __name__ == "__main__":
    main()
