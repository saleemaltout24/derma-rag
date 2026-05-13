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


_DEF_FOCUS_EN = re.compile(
    r"\b(?:what\s+is|what\'?s|define|definition\s+of|meaning\s+of)\s+(?:the\s+)?(.+?)\s*\??\s*$",
    re.I,
)
_DEF_FOCUS_TR = re.compile(r"^(.+?)\s+nedir\s*\??\s*$", re.I)

_REF_PRON_EN = re.compile(
    r"\b(it|this|that|them|they|these|those)\b",
    re.I,
)
_REF_PRON_TR = re.compile(r"\b(bu|şu|bunlar|şunlar|onlar)\b", re.I)


def extract_definition_focus(user_question: str) -> str | None:
    """Trailing noun phrase from definitional questions, e.g. 'What is acne?' -> 'acne'."""
    q = (user_question or "").strip().lower()
    if not q:
        return None
    m = _DEF_FOCUS_EN.search(q)
    if m:
        focus = m.group(1).strip()
    else:
        m = _DEF_FOCUS_TR.match(q)
        if not m:
            return None
        focus = m.group(1).strip()
    focus = re.sub(r"^the\s+", "", focus).rstrip("?.! ")
    if len(focus) < 2 or focus in {"skin", "this", "that", "it", "bu"}:
        return None
    if any(x in focus for x in (" and ", " vs ", " versus ", " ve ", " ile ")):
        return None
    return focus


def _message_has_referential_pronoun(message: str) -> bool:
    if not (message or "").strip():
        return False
    t = (message or "").strip()
    if _REF_PRON_EN.search(t):
        return True
    return bool(_REF_PRON_TR.search(t.lower()))


def infer_topic_for_referential(
    history: list[dict],
    current_state: dict[str, Any],
) -> str | None:
    """Infer entity / site+symptom blob for pronouns like 'it' / 'this' from history and state."""
    for msg in reversed(history or []):
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        focus = extract_definition_focus(content)
        if focus:
            return focus
        if looks_like_definition_question(content):
            m = re.search(
                r"\b(?:what|who)\s+is\s+(?:the\s+)?(.+?)\s*\??\s*$",
                content.strip(),
                re.I,
            )
            if m:
                cand = m.group(1).strip().rstrip("?.! ")
                if len(cand) >= 2 and cand.lower() not in {"skin", "this", "that", "it"}:
                    return cand
    site = current_state.get("body_site")
    syms = current_state.get("symptoms") or []
    parts: list[str] = []
    if site:
        parts.append(str(site))
    for s in syms[:4]:
        if s:
            parts.append(str(s))
    blob = " ".join(parts).strip()
    return blob or None


def maybe_expand_referential_question(
    question: str,
    history: list[dict],
    current_state: dict[str, Any],
) -> str:
    """
    Replace bare pronouns with an inferred topic so extraction / retrieval see
    e.g. 'What causes acne?' instead of 'What causes it?'.
    """
    q0 = (question or "").strip()
    if not q0 or not history:
        return q0
    if not _message_has_referential_pronoun(q0):
        return q0
    topic = infer_topic_for_referential(history, current_state)
    if not topic:
        return q0
    ql = q0.lower()
    if re.search(
        r"\b(my|i\s|i'|i’|our)\s+(rash|skin|face|hand|hands|leg|legs|spots|bumps|itch|itching)\b",
        ql,
    ):
        return q0
    out = q0
    for pat in (r"\bit\b", r"\bthem\b", r"\bthey\b", r"\bthese\b", r"\bthose\b", r"\bthis\b", r"\bthat\b"):
        out = re.sub(pat, topic, out, flags=re.I)
    return out if out.strip() else q0


def is_referential_followup(
    message: str,
    history: list[dict],
    current_state: dict[str, Any],
) -> bool:
    """
    Short follow-up that refers to the prior topic via pronouns (not slot-fillers like 'on my face').
    Used to route intent=FOLLOW_UP and to expand the text before RAG.
    """
    if not _message_has_referential_pronoun(message):
        return False
    if not history:
        return False
    if current_state.get("question_goal") == "general_dermatology":
        return True
    if infer_topic_for_referential(history, current_state):
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
