from backend.config import RERANK_TOP_K, RETRIEVE_TOP_K
from backend.skin_classifier import classify_skin_image
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

ÖNEMLİ KURALLAR:
- Cevabı sadece Türkçe ver.
- Kesin tanı koyma, ama sınıflandırıcı sonucunu daima başta belirt.
- Sınıflandırıcı (aşağıda) en güvenilir kanıttır. Cevabına mutlaka sınıflandırıcının en yüksek tahminini yazarak başla.
- Ders kitabı metni ve görselleri destekleyici kanıttır; sınıflandırıcıyla uyumluysa birlikte sun.
- Ders kitabı sınıflandırıcıyla çelişirse, sınıflandırıcı önceliklidir — farkı kısaca belirt.
- Sınıflandırıcı listesinde OLMAYAN hastalık adlarını KULLANMA.
- Kısa, anlaşılır ve pratik yaz.
- Kullanıcıyı gereksiz korkutma.

=== SINIFLANDIRICI SONUCU (birincil kanıt) ===
{classifier_context}

=== Konuşma geçmişi ===
{history_text}

=== Yapısal durum özeti ===
{structured_state}

=== Kullanıcının yazdığı mesaj ===
{question}

=== Kullanıcı görsel açıklaması ===
{image_description}

=== Ders kitabı benzer görseller (destekleyici) ===
{image_context}

=== Ders kitabı metni (destekleyici) ===
{text_context}

Cevap formatı (bu sırayı takip et):
1. En olası durum — sınıflandırıcının en yüksek tahminini ve güven oranını yaz
2. Neden buna benziyor — görselden ve ders kitabından destekleyici bilgi
3. Ne yapabilirsiniz
4. Ne zaman doktora görünmeli
"""
    return f"""
You are a patient-friendly dermatology assistant.

CRITICAL RULES:
- Answer only in English.
- Do not give a confirmed diagnosis, but always lead with the classifier's top prediction.
- The classifier result below is the primary evidence. You MUST start your answer by stating the classifier's top predicted condition and its confidence.
- Textbook text and images are supporting evidence; present them alongside the classifier when they agree.
- If textbook evidence contradicts the classifier, the classifier takes priority — briefly note the disagreement.
- NEVER use disease names that are NOT in the classifier's prediction list.
- Keep the answer practical and easy to understand.
- Do not scare the user unnecessarily.

=== CLASSIFIER RESULT (primary evidence) ===
{classifier_context}

=== Conversation history ===
{history_text}

=== Structured state ===
{structured_state}

=== User message ===
{question}

=== Uploaded image description ===
{image_description}

=== Similar textbook images (supporting) ===
{image_context}

=== Retrieved textbook text (supporting) ===
{text_context}

Answer format (follow this order):
1. Most likely condition — state the classifier's top prediction and confidence percentage
2. Why it may match — supporting details from the image and textbook
3. What you can do
4. When to see a doctor
"""


def format_classifier_context(classifier_result: dict) -> str:
    top_name = classifier_result.get("predicted_name", "")
    top_confidence = classifier_result.get("confidence", 0.0)
    all_preds = classifier_result.get("all_predictions", [])

    if not top_name or top_confidence == 0.0:
        return "Classifier unavailable for this image."

    lines = [
        f"TOP PREDICTION: {top_name} ({top_confidence:.1f}% confidence)",
        "",
        "Full ranking:",
    ]
    for pred in all_preds:
        lines.append(f"  - {pred['name']}: {pred['confidence']:.1f}%")
    return "\n".join(lines)


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
    classifier_result = classify_skin_image(image_path)
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
