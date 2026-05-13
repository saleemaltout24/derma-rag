from backend.config import RERANK_TOP_K, RETRIEVE_TOP_K
from backend.classifier import run_optional_classifier
from backend.image_pipeline import analyze_skin_image
from backend.image_vector_store import search_similar_images
from backend.rag_pipeline import detect_language, format_user_history, rerank_docs
from backend.state_extractor import extract_structured_state
from backend.state_manager import merge_state, build_search_query, format_state_for_prompt, maybe_expand_referential_question
from backend.vector_store import search as search_text
from backend.llm import run_llm


def build_text_context(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.get("source", "Unknown source")
        section = doc.get("section_title", "Unknown section")
        text = doc.get("text", "").strip()
        page_start = doc.get("page_start")
        page_end = doc.get("page_end")
        if page_start is not None and page_end is not None:
            source = f"{source} (pages {page_start}-{page_end})"
        elif page_start is not None:
            source = f"{source} (page {page_start})"
        if text:
            parts.append(f"Source: {source}\nSection: {section}\n{text}")
    return "\n\n".join(parts)


def build_image_context(image_matches: list) -> str:
    if not image_matches:
        return "No similar textbook images found."

    parts = []

    for i, item in enumerate(image_matches, start=1):

        disease = item.get("disease", "unknown")
        body_site = item.get("body_site", "unknown")
        caption = item.get("caption", "")
        nearby = item.get("nearby_text", "")
        source = item.get("source") or item.get("source_pdf", "unknown")
        page = item.get("page", "unknown")
        score = item.get("score")
        score_text = f"{score:.4f}" if isinstance(score, float) else "n/a"

        parts.append(
            f"""
Match {i}
Source: {source}
Page: {page}
Similarity score: {score_text}
Possible condition: {disease}
Body site: {body_site}
Caption: {caption}

Medical description:
{nearby}
"""
        )

    return "\n".join(parts)


def build_multimodal_prompt(
    question: str,
    history_text: str,
    structured_state: str,
    image_description: str,
    image_context: str,
    text_context: str,
    classifier_context: str,
    language: str,
) -> str:
    if language == "tr":
        return f"""
Sen hasta dostu bir dermatoloji asistanısın.

Kurallar:
- Cevabı sadece Türkçe ver.
- Kesin tanı koyma.
- Kanıt kaynaklarını karıştırma:
  - Sistem A (ders kitabı metni + ders kitabı görselleri) ana kaynaktır.
  - Sistem B (opsiyonel sınıflandırıcı) varsa sadece zayıf ön bilgi olarak kullan.
- En olası yaygın durumları önce düşün.
- Kısa, anlaşılır ve pratik yaz.
- Kullanıcıyı gereksiz korkutma.
- Kanıtlar çelişirse bunu açıkça söyle ve System A'ya öncelik ver.

Konuşma geçmişi:
{history_text}

Yapısal durum özeti:
{structured_state}

Kullanıcının yazdığı mesaj:
{question}

Kanıt Bölümü 1 - Kullanıcı görsel açıklaması:
{image_description}

Kanıt Bölümü 2 - Sistem A (ders kitabı benzer görseller):
{image_context}

Kanıt Bölümü 3 - Sistem A (erişilen ders kitabı metni):
{text_context}

Kanıt Bölümü 4 - Sistem B (opsiyonel sınıflandırıcı):
{classifier_context}

Cevap formatı:
1. En olası durum
2. Neden buna benziyor
3. Ne yapabilirsiniz
4. Ne zaman doktora görünmeli
"""
    return f"""
You are a patient-friendly dermatology assistant.

Rules:
- Answer only in English.
- Do not give a confirmed diagnosis.
- Keep evidence channels separate:
  - System A (textbook text + textbook images) is the primary grounded evidence.
  - System B (optional classifier) is only a weak prior if present.
- Focus on the most likely common conditions first.
- Keep the answer practical and easy to understand.
- Do not scare the user unnecessarily.
- If channels disagree, state the disagreement and prioritize System A.

Conversation history:
{history_text}

Structured state:
{structured_state}

User message:
{question}

Evidence block 1 - Uploaded image description:
{image_description}

Evidence block 2 - System A (similar textbook images):
{image_context}

Evidence block 3 - System A (retrieved textbook text):
{text_context}

Evidence block 4 - System B (optional classifier):
{classifier_context}

Answer format:
1. Most likely condition
2. Why it may match
3. What you can do
4. When to see a doctor
"""


def format_classifier_context(classifier_result: dict) -> str:
    label = classifier_result.get("label")
    confidence = classifier_result.get("confidence")
    uncertainty = classifier_result.get("uncertainty")
    note = classifier_result.get("note", "")
    enabled = classifier_result.get("enabled", False)
    return (
        f"enabled: {enabled}\n"
        f"label: {label}\n"
        f"confidence: {confidence}\n"
        f"uncertainty: {uncertainty}\n"
        f"note: {note}"
    )


def answer_multimodal_question(
    question: str,
    image_path: str,
    history: list,
    current_state: dict,
    forced_language: str | None = None,
):
    language = forced_language if forced_language else detect_language(question or "")
    history_text = format_user_history(history)

    working_question = maybe_expand_referential_question(
        question or "",
        history,
        current_state,
    )

    extracted = extract_structured_state(working_question, history_text)
    updated_state = merge_state(current_state, extracted)

    image_description = analyze_skin_image(image_path)
    text_query = build_search_query(working_question or image_description, updated_state, language)
    retrieved_text_docs = search_text(text_query, k=RETRIEVE_TOP_K)
    text_docs = rerank_docs(
        text_query,
        retrieved_text_docs,
        k=RERANK_TOP_K,
        user_question=working_question or "",
    )

    allowed_source_pages = collect_source_pages(text_docs)
    image_matches = search_similar_images(
        image_path,
        k=3,
        body_site=updated_state.get("body_site"),
        allowed_source_pages=allowed_source_pages,
    )

    image_context = build_image_context(image_matches)
    text_context = build_text_context(text_docs)
    structured_state_text = format_state_for_prompt(updated_state)
    classifier_result = run_optional_classifier(image_path)
    classifier_context = format_classifier_context(classifier_result)

    prompt = build_multimodal_prompt(
        question=working_question or "",
        history_text=history_text,
        structured_state=structured_state_text,
        image_description=image_description,
        image_context=image_context,
        text_context=text_context,
        classifier_context=classifier_context,
        language=language,
    )

    answer = run_llm(prompt)
    retrieval_debug = {
        "query": text_query,
        "retrieve_top_k": RETRIEVE_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "retrieved_count": len(retrieved_text_docs),
        "selected_count": len(text_docs),
        "selected_sources": [
            {
                "source": doc.get("source"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
            }
            for doc in text_docs
        ],
    }

    return (
        answer,
        updated_state,
        image_description,
        image_matches,
        text_docs,
        classifier_result,
        retrieval_debug,
    )


def collect_source_pages(docs: list[dict]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for doc in docs:
        source = doc.get("source")
        if not source:
            continue
        start = doc.get("page_start")
        end = doc.get("page_end")
        if start is None and end is None:
            continue
        if start is None:
            start = end
        if end is None:
            end = start
        for page in range(int(start), int(end) + 1):
            pairs.add((source, page))
    return pairs
