from lingua import Language, LanguageDetectorBuilder

from backend.intent_router import general_help_response
from backend.llm import run_llm
from backend.prompt_template import (
    build_medical_prompt,
    build_product_prompt,
    build_treatment_prompt,
)
from backend.state_extractor import extract_structured_state
from backend.state_manager import build_search_query, format_state_for_prompt, merge_state
from backend.vector_store import search

languages = [Language.ENGLISH, Language.TURKISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()



def detect_language(text: str) -> str:
    language = detector.detect_language_of(text)

    if language == Language.TURKISH:
        return "tr"
    if language == Language.ENGLISH:
        return "en"
    return "en"



def format_user_history(history: list, max_turns: int = 8) -> str:
    recent = history[-max_turns:]
    lines = []

    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role and content:
            lines.append(f"{role.title()}: {content}")

    return "\n".join(lines)



def build_context_from_docs(docs: list) -> str:
    parts = []

    for doc in docs:
        source = doc.get("source", "Unknown source")
        section = doc.get("section_title", "Unknown section")
        text = doc.get("text", "").strip()

        if text:
            parts.append(f"Source: {source}\nSection: {section}\n{text}")

    return "\n\n".join(parts)



def answer_general_help(language: str) -> str:
    return general_help_response(language)



def answer_medical_question(
    question: str,
    history: list,
    current_state: dict,
    forced_language: str | None = None,
) -> tuple[str, dict]:
    language = forced_language if forced_language else detect_language(question)
    history_text = format_user_history(history)

    extracted = extract_structured_state(question, history_text)
    updated_state = merge_state(current_state, extracted)

    search_query = build_search_query(question, updated_state, language)
    docs = search(search_query, k=4)
    context = build_context_from_docs(docs)
    structured_state_text = format_state_for_prompt(updated_state)

    goal = updated_state.get("question_goal")
    if goal == "product_advice":
        prompt = build_product_prompt(
            context=context,
            question=question,
            history=history_text,
            structured_state=structured_state_text,
            language=language,
        )
    elif goal == "treatment_advice":
        prompt = build_treatment_prompt(
            context=context,
            question=question,
            history=history_text,
            structured_state=structured_state_text,
            language=language,
        )
    else:
        prompt = build_medical_prompt(
            context=context,
            question=question,
            history=history_text,
            structured_state=structured_state_text,
            language=language,
    )

    answer = run_llm(prompt)
    return answer, updated_state