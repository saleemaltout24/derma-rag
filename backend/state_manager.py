import re
from typing import Any


def looks_like_definition_question(message: str) -> bool:
    """Heuristic: user wants a concept definition (improves embedding query)."""
    t = message.strip().lower()
    if not t:
        return False
    if re.search(
        r"\b(what\s+is|what\'?s|what\s+are|define|definition(\s+of)?|meaning(\s+of)?)\b",
        t,
    ):
        return True
    if re.search(
        r"\b(nedir|tanımı|tanım(\s+nedir)?|tanımla|anlamı|ne\s+demek)\b",
        t,
    ):
        return True
    return False


def create_empty_state() -> dict[str, Any]:
    return {
        "body_site": None,
        "symptoms": [],
        "triggers": [],
        "duration": None,
        "severity": None,
        "question_goal": None,
        "ruled_out": [],
        "notes": [],
    }


def _merge_unique_list(old: list[str], new: list[str]) -> list[str]:
    combined = list(old)
    for item in new:
        if item and item not in combined:
            combined.append(item)
    return combined


def merge_state(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)

    if update.get("body_site"):
        merged["body_site"] = update["body_site"]

    if update.get("duration"):
        merged["duration"] = update["duration"]

    if update.get("severity"):
        merged["severity"] = update["severity"]

    if update.get("question_goal"):
        merged["question_goal"] = update["question_goal"]

    merged["symptoms"] = _merge_unique_list(current.get("symptoms", []), update.get("symptoms", []))
    merged["triggers"] = _merge_unique_list(current.get("triggers", []), update.get("triggers", []))
    merged["ruled_out"] = _merge_unique_list(current.get("ruled_out", []), update.get("ruled_out", []))
    merged["notes"] = _merge_unique_list(current.get("notes", []), update.get("notes", []))

    return merged


def build_search_query(user_message: str, state: dict[str, Any], language: str) -> str:
    q = user_message.strip()
    if looks_like_definition_question(q):
        if language == "tr":
            q_for_search = f"tanım dermatoloji {q}"
        else:
            q_for_search = f"definition dermatology {q}"
    else:
        q_for_search = q

    lines = [f"Current question: {q_for_search}"]

    if state.get("body_site"):
        lines.append(f"Body site: {state['body_site']}")

    if state.get("symptoms"):
        lines.append("Symptoms: " + ", ".join(state["symptoms"]))

    if state.get("triggers"):
        lines.append("Triggers: " + ", ".join(state["triggers"]))

    if state.get("duration"):
        lines.append(f"Duration: {state['duration']}")

    if state.get("severity"):
        lines.append(f"Severity: {state['severity']}")

    if state.get("ruled_out"):
        lines.append("Ruled out: " + ", ".join(state["ruled_out"]))

    if state.get("question_goal"):
        lines.append(f"Goal: {state['question_goal']}")

    if language == "tr":
        lines.append("Language: Turkish")
    else:
        lines.append("Language: English")

    return "\n".join(lines)


def format_state_for_prompt(state: dict[str, Any]) -> str:
    parts = []

    for key in [
        "body_site",
        "symptoms",
        "triggers",
        "duration",
        "severity",
        "question_goal",
        "ruled_out",
        "notes",
    ]:
        value = state.get(key)
        if value:
            parts.append(f"{key}: {value}")

    return "\n".join(parts) if parts else "No structured state available."