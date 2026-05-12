import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT_DIR / "data" / "eval_end_to_end.sample.json"
DEFAULT_API = "http://127.0.0.1:8000"


def call_ask(api_base: str, question: str, session_id: str) -> tuple[dict, float]:
    params = urllib.parse.urlencode({"question": question, "session_id": session_id})
    url = f"{api_base}/ask?{params}"
    started = time.perf_counter()
    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return payload, elapsed_ms


def main() -> None:
    api_base = DEFAULT_API
    case_path = DEFAULT_CASES

    if len(__import__("sys").argv) > 1:
        api_base = __import__("sys").argv[1]
    if len(__import__("sys").argv) > 2:
        case_path = Path(__import__("sys").argv[2])

    if not case_path.exists():
        raise FileNotFoundError(f"Case file not found: {case_path}")

    with open(case_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    latencies = []
    for idx, case in enumerate(cases, start=1):
        question = case.get("question", "")
        session_id = case.get("session_id", f"eval-{idx}")
        payload, latency_ms = call_ask(api_base, question, session_id)
        latencies.append(latency_ms)
        results.append(
            {
                "index": idx,
                "question": question,
                "session_id": session_id,
                "latency_ms": round(latency_ms, 2),
                "intent": payload.get("intent"),
                "answer_preview": (payload.get("answer") or "")[:240],
            }
        )

    summary = {
        "api_base": api_base,
        "num_cases": len(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2) if latencies else None,
        "results": results,
    }

    output_path = ROOT_DIR / "processed" / "end_to_end_eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluated {len(results)} end-to-end cases.")
    print(f"Saved report to: {output_path}")
    print(json.dumps({"avg_latency_ms": summary["avg_latency_ms"], "p95_latency_ms": summary["p95_latency_ms"]}, indent=2))


if __name__ == "__main__":
    main()
