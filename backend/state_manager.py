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

    # Scalar slots: allow extractor to clear with null (truthy-only updates used to
    # leave stale question_goal across turns, e.g. vague symptom → "what is acne").
    if "body_site" in update:
        merged["body_site"] = update.get("body_site") or None

    if "duration" in update:
        merged["duration"] = update.get("duration") or None

    if "severity" in update:
        merged["severity"] = update.get("severity") or None

    if "question_goal" in update:
        merged["question_goal"] = update.get("question_goal")

    merged["symptoms"] = _merge_unique_list(current.get("symptoms", []), update.get("symptoms", []))
    merged["triggers"] = _merge_unique_list(current.get("triggers", []), update.get("triggers", []))
    merged["ruled_out"] = _merge_unique_list(current.get("ruled_out", []), update.get("ruled_out", []))
    merged["notes"] = _merge_unique_list(current.get("notes", []), update.get("notes", []))

    return merged


def build_search_query(user_message: str, state: dict[str, Any], language: str) -> str:
    """
    One-line semantic query for embedding / FAISS. Avoids labeled lines like
    "Current question:" / "Goal:" — those add little semantic signal and dilute the vector.
    """
    parts: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        t = (s or "").strip()
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        parts.append(t)

    q_raw = user_message.strip()
    q = q_raw.lower() if q_raw else ""

    if looks_like_definition_question(q_raw):
        if language == "tr":
            add("tanım dermatoloji")
        else:
            add("definition dermatology")

    if q:
        add(q)
    else:
        add("dermatology")

    if state.get("body_site"):
        add(str(state["body_site"]))

    for x in (state.get("symptoms") or [])[:10]:
        add(str(x))
    for x in (state.get("triggers") or [])[:8]:
        add(str(x))
    for x in (state.get("ruled_out") or [])[:5]:
        add(str(x))

    if state.get("duration"):
        add(str(state["duration"]))
    if state.get("severity"):
        add(str(state["severity"]))

    goal = state.get("question_goal")
    if goal == "treatment_advice":
        add("treatment")
    elif goal == "cause_assessment":
        add("causes")
    elif goal == "symptom_assessment":
        add("symptoms")
    elif goal == "product_advice":
        add("skincare product")
    elif goal == "diagnosis_question":
        add("diagnosis")
    elif goal == "prevention_advice":
        add("prevention")

    return " ".join(parts)


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
