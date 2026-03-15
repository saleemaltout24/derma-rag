from typing import Any

from backend.llm import run_json_llm


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


def extract_structured_state(message: str, history_text: str = "") -> dict[str, Any]:
    prompt = f"""
You are a structured medical information extractor for a dermatology chatbot.

Extract only information explicitly stated or strongly implied by the user's latest message.
Use the earlier conversation only when needed to understand short follow-ups.
Do not invent facts.

Return valid JSON only.

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

Previous conversation:
{history_text}

Latest user message:
{message}
"""
    data = run_json_llm(prompt)

    if "error" in data:
        return dict(EXTRACTION_SCHEMA_EXAMPLE)

    cleaned = dict(EXTRACTION_SCHEMA_EXAMPLE)
    cleaned["body_site"] = data.get("body_site")
    cleaned["duration"] = data.get("duration")
    cleaned["severity"] = data.get("severity")

    symptoms = data.get("symptoms", [])
    triggers = data.get("triggers", [])
    ruled_out = data.get("ruled_out", [])
    notes = data.get("notes", [])
    question_goal = data.get("question_goal")

    cleaned["symptoms"] = symptoms if isinstance(symptoms, list) else []
    cleaned["triggers"] = triggers if isinstance(triggers, list) else []
    cleaned["ruled_out"] = ruled_out if isinstance(ruled_out, list) else []
    cleaned["notes"] = notes if isinstance(notes, list) else []

    if question_goal in ALLOWED_GOALS:
        cleaned["question_goal"] = question_goal
    else:
        cleaned["question_goal"] = None

    return cleaned