import logging
from typing import Any

from backend.llm import run_json_llm

logger = logging.getLogger(__name__)


EXTRACTION_SCHEMA_EXAMPLE = {
    "body_site": None,
    "symptoms": [],
    "triggers": [],
    "duration": None,
    "severity": None,
    "question_goal": None,
    "ruled_out": [],
    "notes": [],
}


ALLOWED_GOALS = {
    "symptom_assessment",
    "cause_assessment",
    "treatment_advice",
    "product_advice",
    "prevention_advice",
    "diagnosis_question",
    "general_dermatology",
}


def _as_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return None


def _list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def extract_structured_state(message: str, history_text: str = "") -> dict[str, Any]:
    prompt = f"""
You are a structured medical information extractor for a dermatology chatbot.

Extract only information explicitly stated or strongly implied by the user's latest message.
Use the earlier conversation when relevant to understand short follow-ups.
Do not invent facts.

Output language (important):
- Put every human-readable string value in ENGLISH (plain clinical or everyday terms).
- Translate from Turkish or other languages when needed.
- The field "question_goal" must be exactly one of the allowed English snake_case literals below, or null.

Return valid JSON only (one object, no markdown fences).

Schema:
{{
  "body_site": string or null,
  "symptoms": list of strings,
  "triggers": list of strings,
  "duration": string or null,
  "severity": string or null,
  "question_goal": one of ["symptom_assessment", "cause_assessment", "treatment_advice", "product_advice", "prevention_advice", "diagnosis_question", "general_dermatology"] or null,
  "ruled_out": list of strings,
  "notes": list of strings
}}

Examples:
- "elim çok kaşınıyor" => body_site hand, symptoms itching, question_goal symptom_assessment
- "alkol içmedim dezenfektan kullandım" => triggers alcohol-based disinfectant, ruled_out drinking alcohol, question_goal cause_assessment
- "hangi kremi kullanayım" => question_goal product_advice
- "My scalp has been flaking for a month" => body_site scalp, symptoms flaking, duration one month, question_goal symptom_assessment

Previous conversation:
{history_text}

Latest user message:
{message}
"""
    data = run_json_llm(prompt)

    if "error" in data:
        logger.warning("state_extractor: extraction failed: %s", data)
        return dict(EXTRACTION_SCHEMA_EXAMPLE)

    cleaned = dict(EXTRACTION_SCHEMA_EXAMPLE)
    cleaned["body_site"] = _as_optional_str(data.get("body_site"))
    cleaned["duration"] = _as_optional_str(data.get("duration"))
    cleaned["severity"] = _as_optional_str(data.get("severity"))

    cleaned["symptoms"] = _list_of_strings(data.get("symptoms", []))
    cleaned["triggers"] = _list_of_strings(data.get("triggers", []))
    cleaned["ruled_out"] = _list_of_strings(data.get("ruled_out", []))
    cleaned["notes"] = _list_of_strings(data.get("notes", []))

    question_goal = data.get("question_goal")
    if question_goal in ALLOWED_GOALS:
        cleaned["question_goal"] = question_goal
    else:
        cleaned["question_goal"] = None

    return cleaned
