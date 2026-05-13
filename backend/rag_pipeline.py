import re

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
from backend.state_manager import (
    build_search_query,
    create_empty_state,
    format_state_for_prompt,
    looks_like_definition_question,
    merge_state,
)
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


_DEF_FOCUS_EN = re.compile(
    r"\b(?:what\s+is|what\'?s|define|definition\s+of|meaning\s+of)\s+(?:the\s+)?(.+?)\s*\??\s*$",
    re.I,
)
_DEF_FOCUS_TR = re.compile(r"^(.+?)\s+nedir\s*\??\s*$", re.I)


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


def chunk_matches_definition_focus(focus: str, text: str) -> bool:
    """True if every meaningful token in *focus* appears as a whole word in *text*."""
    if not focus or not text:
        return False
    t = (text or "").lower()
    tokens = [p for p in re.split(r"\s+", focus.strip().lower()) if len(p) >= 2]
    if not tokens:
        tokens = [focus.strip().lower()]
    for tok in tokens:
        if len(tok) == 2 and not tok.isalpha():
            continue
        if not re.search(rf"\b{re.escape(tok)}\b", t, re.I):
            return False
    return True


def refine_docs_for_definition_retrieval(
    question: str,
    retrieved_docs: list[dict],
    retrieve_k: int,
) -> list[dict]:
    """
    Hybrid recall: dense retrieval often misses the right entity; for definitional
    questions, prefer chunks that literally contain the asked concept, and optionally
    run a second dense query focused on that concept.
    """
    if not retrieved_docs or not looks_like_definition_question(question):
        return retrieved_docs
    focus = extract_definition_focus(question)
    if not focus:
        return retrieved_docs

    def _sort_by_distance(ds: list[dict]) -> list[dict]:
        out = [dict(d) for d in ds]
        out.sort(key=lambda d: float(d.get("distance", d.get("score", 0.0))))
        return out

    def _matches(ds: list[dict]) -> list[dict]:
        return [d for d in ds if chunk_matches_definition_focus(focus, d.get("text", ""))]

    hits = _matches(retrieved_docs)
    if len(hits) >= 3:
        return _sort_by_distance(hits)[:retrieve_k]

    booster = f"{focus} dermatology clinical pathogenesis diagnosis"
    try:
        raw_extra = search(booster, k=retrieve_k)
        extra = clean_docs_for_retrieval(raw_extra)
        if not extra:
            extra = list(raw_extra)
    except Exception:
        extra = []

    seen: set[tuple] = set()
    merged: list[dict] = []
    for pool in (retrieved_docs, extra):
        for d in pool:
            key = (
                d.get("source"),
                d.get("page_start"),
                d.get("page_end"),
                (d.get("text") or "")[:240],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(d))

    hits2 = _matches(merged)
    if hits2:
        return _sort_by_distance(hits2)[:retrieve_k]
    if hits:
        return _sort_by_distance(hits)[:retrieve_k]
    return retrieved_docs


def topic_alignment_adjustment(user_question: str, chunk_text: str) -> int:
    """
    Downrank passages whose *primary* topic conflicts with a definitional focus
    (e.g. rosacea chapter hits 'acne rosacea' + 'definition' for 'What is acne?').
    """
    focus = extract_definition_focus(user_question)
    if not focus:
        return 0
    q_lower = user_question.lower()
    t = (chunk_text or "").lower()
    focus_l = focus.lower()

    acne_focus = focus_l in (
        "acne",
        "akne",
        "akne vulgarisi",
        "akne vulgaris",
    ) or focus_l.startswith(("acne ", "akne "))
    if acne_focus and "rosacea" not in q_lower:
        has_vulgaris = bool(re.search(r"\bacne\s+vulgaris\b", t))
        if has_vulgaris:
            return 6
        ros_hits = len(re.findall(r"\brosacea\b", t))
        has_acne_rosacea = "acne rosacea" in t
        acne_primary = bool(
            re.search(r"\bcomedo", t)
            or "pilosebaceous" in t
            or "sebaceous gland" in t
            or "p. acnes" in t
            or "propionibacterium" in t
        )
        if ros_hits >= 2 and (has_acne_rosacea or ros_hits >= 4) and not has_vulgaris:
            if not acne_primary:
                return -20
        if ros_hits >= 4 and not has_vulgaris and not acne_primary:
            return -12

    return 0


def _definition_priority_lexical(text: str) -> int:
    """Lexical tie-breaker for definitional prose (not query-aware)."""
    t = (text or "").lower()
    padded = f" {t} "
    score = 0

    if " is a " in padded:
        score += 3
    if " is an " in padded:
        score += 3
    if "refers to" in t:
        score += 3
    if "defined as" in t:
        score += 3
    if "definition" in t:
        score += 2

    if " nedir" in padded or " tanımlanır" in t or " olarak bilinir" in t:
        score += 2

    domain_hits = sum(
        1 for w in ("chronic", "inflammatory", "condition", "disease") if w in t
    )
    score += min(domain_hits, 2)

    if " figure " in padded or t.lstrip().startswith("figure "):
        score -= 2
    if t.count(",") > 25 and "chapter" in t:
        score -= 3

    return score


def definition_priority(text: str, user_question: str | None = None) -> int:
    """Definitional chunk score: lexical cues plus query–passage topic alignment."""
    score = _definition_priority_lexical(text)
    if user_question:
        score += topic_alignment_adjustment(user_question, text)
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


# Pre-retrieval slot gates: only when question_goal is set (None => skip gate, e.g. definitional / general questions).
_GOALS_NEEDING_BODY_AND_SYMPTOMS = frozenset(
    {"symptom_assessment", "diagnosis_question", "cause_assessment"}
)
_GOALS_NEEDING_BODY_OR_SYMPTOMS = frozenset({"treatment_advice"})


def _missing_slots_for_gate(state: dict) -> list[str]:
    """Return ordered list of missing slot names for debug / messaging."""
    missing: list[str] = []
    if not state.get("body_site"):
        missing.append("body_site")
    symptoms = state.get("symptoms") or []
    if not symptoms:
        missing.append("symptoms")
    return missing


def _clarification_for_missing_slots(
    goal: str | None,
    missing: list[str],
    language: str,
) -> str:
    need_loc = "body_site" in missing
    need_sym = "symptoms" in missing
    if language == "tr":
        if need_loc and need_sym:
            return (
                "Daha iyi yardımcı olabilmem için iki şeye ihtiyacım var: "
                "Şikayetin vücudun hangi bölgesinde (ör. el, yüz, saçlı deri)? "
                "Ve ne hissediyorsun (ör. kaşıntı, yanma, pullanma, kızarıklık)?"
            )
        if need_loc:
            return "Şikayetin vücudun hangi bölgesinde olduğunu yazar mısın? (örnek: dirsek, yüz, ayak tabanı)"
        return "Hangi belirtileri yaşıyorsun? (örnek: kaşıntı, kızarıklık, pullanma, ağrı)"
    if need_loc and need_sym:
        return (
            "To help properly I need two things: where on your body is the problem "
            "(e.g. hand, face, scalp)? And what are you feeling (e.g. itch, burn, scale, redness)?"
        )
    if need_loc:
        return "Where on your body is this happening? (e.g. elbow, face, sole of the foot)"
    return "What symptoms are you having? (e.g. itch, redness, flaking, pain)"


def _should_skip_retrieval_for_slots(state: dict) -> tuple[bool, list[str]]:
    """
    If True, skip FAISS + answer LLM and return a single clarification message instead.
    """
    goal = state.get("question_goal")
    if goal is None:
        return False, []

    body_ok = bool(state.get("body_site"))
    sym_ok = bool(state.get("symptoms"))

    if goal in _GOALS_NEEDING_BODY_AND_SYMPTOMS:
        if body_ok and sym_ok:
            return False, []
        return True, _missing_slots_for_gate(state)

    if goal in _GOALS_NEEDING_BODY_OR_SYMPTOMS:
        if body_ok or sym_ok:
            return False, []
        return True, _missing_slots_for_gate(state)

    return False, []


def _skipped_retrieval_debug(
    question_goal: str | None,
    missing: list[str],
) -> dict:
    return {
        "skipped_retrieval": True,
        "reason": "missing_structured_slots",
        "question_goal": question_goal,
        "missing": missing,
        "query": None,
        "retrieve_top_k": RETRIEVE_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "enable_rerank": ENABLE_RERANK,
        "retrieved_count": 0,
        "selected_count": 0,
        "selected_sources": [],
    }


def answer_medical_question(
    question: str,
    history: list,
    current_state: dict,
    forced_language: str | None = None,
) -> tuple[str, dict, dict]:
    language = forced_language if forced_language else detect_language(question)
    history_text = format_user_history(history)

    extracted = extract_structured_state(question, history_text)
    if looks_like_definition_question(question):
        # New definitional turn should not inherit prior symptom slots / question_goal.
        updated_state = merge_state(create_empty_state(), extracted)
    else:
        updated_state = merge_state(current_state, extracted)

    skip, missing = (False, [])
    if not looks_like_definition_question(question):
        skip, missing = _should_skip_retrieval_for_slots(updated_state)
    if skip:
        msg = _clarification_for_missing_slots(
            updated_state.get("question_goal"),
            missing,
            language,
        )
        debug = _skipped_retrieval_debug(updated_state.get("question_goal"), missing)
        return msg, updated_state, debug

    search_query = build_search_query(question, updated_state, language)
    retrieved_raw = search(search_query, k=RETRIEVE_TOP_K)
    retrieved_docs = clean_docs_for_retrieval(retrieved_raw)
    if not retrieved_docs:
        retrieved_docs = list(retrieved_raw)

    retrieved_docs = refine_docs_for_definition_retrieval(
        question,
        retrieved_docs,
        RETRIEVE_TOP_K,
    )

    print("\n===== RETRIEVED DOCS =====")
    for doc in retrieved_docs[:3]:
        print(doc.get("text", "")[:300])
        print("-----")

    docs = rerank_docs(
        search_query,
        retrieved_docs,
        k=RERANK_TOP_K,
        user_question=question,
    )

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


def rerank_docs(
    query: str,
    docs: list[dict],
    k: int,
    user_question: str | None = None,
) -> list[dict]:
    if not docs:
        return docs

    uq = (user_question or "").strip()
    uq_for_priority = uq if uq else None
    ce_query = uq if uq and looks_like_definition_question(uq) else query

    def _faiss_distance(d: dict) -> float:
        return float(d.get("distance", d.get("score", 0.0)))

    if not ENABLE_RERANK:
        boosted = [dict(doc) for doc in docs]
        for item in boosted:
            item["definition_priority"] = definition_priority(
                item.get("text", ""),
                uq_for_priority,
            )
        boosted.sort(key=lambda d: (-d["definition_priority"], _faiss_distance(d)))
        return boosted[:k]

    global reranker
    if reranker is None:
        try:
            reranker = CrossEncoder(RERANK_MODEL)
        except Exception:
            boosted = [dict(doc) for doc in docs]
            for item in boosted:
                item["definition_priority"] = definition_priority(
                    item.get("text", ""),
                    uq_for_priority,
                )
            boosted.sort(key=lambda d: (-d["definition_priority"], _faiss_distance(d)))
            return boosted[:k]

    pairs = [(ce_query, doc.get("text", "")) for doc in docs]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        boosted = [dict(doc) for doc in docs]
        for item in boosted:
            item["definition_priority"] = definition_priority(
                item.get("text", ""),
                uq_for_priority,
            )
        boosted.sort(key=lambda d: (-d["definition_priority"], _faiss_distance(d)))
        return boosted[:k]

    reranked = []
    for doc, score in zip(docs, scores):
        item = dict(doc)
        item["rerank_score"] = float(score)
        item["definition_priority"] = definition_priority(
            item.get("text", ""),
            uq_for_priority,
        )
        reranked.append(item)

    reranked.sort(
        key=lambda d: (
            -d["definition_priority"],
            -d.get("rerank_score", 0.0),
            _faiss_distance(d),
        )
    )
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
