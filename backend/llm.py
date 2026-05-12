import json
from typing import Any
from backend.config import CHAT_MODEL
import ollama

def run_llm(prompt: str, model: str | None = None) -> str:
    chosen_model = model or CHAT_MODEL

    # 🔴 DISABLE MODE
    if chosen_model in [None, "", "none"]:
        return "LLM disabled — pipeline working"

    try:
        response = ollama.chat(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
            },
        )
        return response["message"]["content"].strip()

    except Exception as e:
        return f"LLM error: {str(e)}"


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
