from lingua import Language, LanguageDetectorBuilder
from sentence_transformers import CrossEncoder

from backend.config import ENABLE_RERANK, RERANK_MODEL, RERANK_TOP_K, RETRIEVE_TOP_K
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
reranker = None



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



def definition_priority(text: str) -> int:
    """Integer tie-breaker for definitional chunks. Never mixed into FAISS L2 distance."""
    t = text.lower()
    padded = f" {t} "
    score = 0
    if " is a " in padded or " is an " in padded:
        score += 2
    if "defined as" in t:
        score += 2
    if "refers to" in t:
        score += 2
    if "definition" in t:
        score += 2
    if " nedir" in padded or " tanımlanır" in t or " olarak bilinir" in t:
        score += 2
    return score


def clean_docs_for_retrieval(docs: list[dict]) -> list[dict]:
    """Conservative junk filter before rerank (word count, boilerplate, heading-only)."""
    cleaned: list[dict] = []
    for d in docs:
        t = (d.get("text") or "").strip()
        if not t:
            continue
        tl = t.lower()
        if "intentionally left blank" in tl:
            continue
        words = t.split()
        if len(words) < 20:
            continue
        n = len(t)
        uppercase_ratio = sum(1 for c in t if c.isupper()) / max(n, 1)
        if uppercase_ratio > 0.6 and len(words) < 40:
            continue
        cleaned.append(d)
    return cleaned


def build_context_from_docs(docs: list) -> str:
    MAX_CHARS = 3000
    parts = []

    for doc in docs:
        source = doc.get("source", "Unknown source")
        section = doc.get("section_title", "Unknown section")
        text = doc.get("text", "").strip()
        if not text:
            continue
        page_start = doc.get("page_start")
        page_end = doc.get("page_end")
        page_info = ""
        if page_start is not None and page_end is not None:
            page_info = f"\nPages: {page_start}-{page_end}"
        elif page_start is not None:
            page_info = f"\nPage: {page_start}"

        if text:
            parts.append(f"Source: {source}{page_info}\nSection: {section}\n{text}")

    context = "\n\n".join(parts[:5])
    return context[:MAX_CHARS]



def answer_general_help(language: str) -> str:
    return general_help_response(language)



def answer_medical_question(
    question: str,
    history: list,
    current_state: dict,
    forced_language: str | None = None,
) -> tuple[str, dict, dict]:
    language = forced_language if forced_language else detect_language(question)
    history_text = format_user_history(history)

    extracted = extract_structured_state(question, history_text)
    updated_state = merge_state(current_state, extracted)

    search_query = build_search_query(question, updated_state, language)
    retrieved_raw = search(search_query, k=RETRIEVE_TOP_K)
    retrieved_docs = clean_docs_for_retrieval(retrieved_raw)
    if not retrieved_docs:
        retrieved_docs = list(retrieved_raw)

    print("\n===== RETRIEVED DOCS =====")
    for doc in retrieved_docs[:3]:
        print(doc.get("text", "")[:300])
        print("-----")

    docs = rerank_docs(search_query, retrieved_docs, k=RERANK_TOP_K)

    print("\n===== RERANKED DOC SCORES (top 5) =====")
    for doc in docs[:5]:
        print(doc.get("score"), doc.get("rerank_score"), doc.get("definition_priority"))

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

    print("\n===== FINAL PROMPT =====\n")
    print(prompt[:2000])

    answer = run_llm(prompt)
    debug = build_retrieval_debug(search_query, retrieved_docs, docs)
    return answer, updated_state, debug


def rerank_docs(query: str, docs: list[dict], k: int) -> list[dict]:
    if not docs:
        return docs

    def _faiss_distance(d: dict) -> float:
        return float(d.get("distance", d.get("score", 0.0)))

    if not ENABLE_RERANK:
        boosted = [dict(doc) for doc in docs]
        for item in boosted:
            item["definition_priority"] = definition_priority(item.get("text", ""))
        boosted.sort(key=lambda d: (-d["definition_priority"], _faiss_distance(d)))
        return boosted[:k]

    global reranker
    if reranker is None:
        try:
            reranker = CrossEncoder(RERANK_MODEL)
        except Exception:
            boosted = [dict(doc) for doc in docs]
            for item in boosted:
                item["definition_priority"] = definition_priority(item.get("text", ""))
            boosted.sort(key=lambda d: (-d["definition_priority"], _faiss_distance(d)))
            return boosted[:k]

    pairs = [(query, doc.get("text", "")) for doc in docs]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        boosted = [dict(doc) for doc in docs]
        for item in boosted:
            item["definition_priority"] = definition_priority(item.get("text", ""))
        boosted.sort(key=lambda d: (-d["definition_priority"], _faiss_distance(d)))
        return boosted[:k]

    reranked = []
    for doc, score in zip(docs, scores):
        item = dict(doc)
        item["rerank_score"] = float(score)
        item["definition_priority"] = definition_priority(item.get("text", ""))
        reranked.append(item)

    reranked.sort(key=lambda d: (-d["definition_priority"], -d["rerank_score"]))
    return reranked[:k]


def build_retrieval_debug(
    query: str,
    retrieved_docs: list[dict],
    selected_docs: list[dict],
) -> dict:
    return {
        "query": query,
        "retrieve_top_k": RETRIEVE_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "enable_rerank": ENABLE_RERANK,
        "retrieved_count": len(retrieved_docs),
        "selected_count": len(selected_docs),
        "selected_sources": [
            {
                "source": doc.get("source"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
                "definition_priority": doc.get("definition_priority"),
                "distance": doc.get("distance", doc.get("score")),
            }
            for doc in selected_docs
        ],
    }