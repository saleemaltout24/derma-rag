import json
import subprocess
from typing import Any

from backend.config import CHAT_MODEL


def run_llm(prompt: str, model: str | None = None) -> str:
    chosen_model = model or CHAT_MODEL

    result = subprocess.run(
        ["ollama", "run", chosen_model],
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )

    if result.returncode != 0:
        return f"LLM error: {result.stderr}"

    return result.stdout.strip()


def run_json_llm(prompt: str) -> dict[str, Any]:
    raw = run_llm(prompt)

    if raw.lower().startswith("llm error"):
        return {"error": raw}

    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass

    return {"error": "Invalid JSON from LLM", "raw": raw}