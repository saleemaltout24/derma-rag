import json
from typing import Any

import ollama

from backend.config import CHAT_MODEL, LLM_NUM_CTX, LLM_NUM_PREDICT


class LLMError(RuntimeError):
    """Raised when Ollama is unreachable or the model fails; map to HTTP 503 in app.py."""


def run_llm(prompt: str, model: str | None = None) -> str:
    chosen_model = model or CHAT_MODEL

    if chosen_model in (None, "", "none"):
        raise LLMError("LLM is disabled (CHAT_MODEL is none). Set CHAT_MODEL in .env.")

    try:
        response = ollama.chat(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.2,
                "num_predict": LLM_NUM_PREDICT,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        return response["message"]["content"].strip()

    except LLMError:
        raise
    except Exception as e:
        raise LLMError(
            f"Ollama request failed for model '{chosen_model}'. "
            f"Is Ollama running and is the model pulled? ({e})"
        ) from e


def run_json_llm(prompt: str) -> dict[str, Any]:
    try:
        raw = run_llm(prompt)
    except LLMError as e:
        return {"error": str(e)}
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"error": "Invalid JSON from LLM", "raw": raw}
