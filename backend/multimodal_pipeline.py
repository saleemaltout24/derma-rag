import re

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


def is_classifier_available(classifier_result: dict) -> bool:
    code = classifier_result.get("predicted_class", "")
    confidence = classifier_result.get("confidence", 0.0) or 0.0
    if code in ("UNAVAILABLE", "UNKNOWN", ""):
        return False
    return confidence > 0


def build_classifier_answer_lead(classifier_result: dict, language: str) -> str:
    if not is_classifier_available(classifier_result):
        return ""

    name = classifier_result.get("predicted_name", "")
    code = classifier_result.get("predicted_class", "")
    confidence = float(classifier_result.get("confidence", 0.0))
    top2_name = classifier_result.get("top2_name")
    top2_confidence = classifier_result.get("top2_confidence")
    ambiguous = classifier_result.get("ambiguous") is True
    all_predictions = classifier_result.get("all_predictions", [])

    ranking_lines = [
        f"  - {pred.get('name', '?')}: {float(pred.get('confidence', 0)):.1f}%"
        for pred in all_predictions[:8]
    ]
    ranking = "\n".join(ranking_lines)

    if language == "tr":
        lines = [
            "### 1. En Olası Durum",
            f"Bu yükleme için görüntü sınıflandırıcısının en yüksek tahmini: **{name}** ({code}) — **%{confidence:.1f}** güven.",
        ]
        if top2_name:
            runner = f"İkinci sıra: {top2_name} (%{float(top2_confidence or 0):.1f})."
            lines.append(
                f"{runner} Model tam kesin değil — klinik değerlendirme önemli."
                if ambiguous
                else runner
            )
        lines.extend(
            [
                "",
                "Bu görsel için sınıflandırıcı sıralaması:",
                ranking,
                "",
                "*Yalnızca yapay zekâ taramasıdır; kesin tanı değildir. Şüpheli lezyonlar için dermatoloğa başvurun.*",
            ]
        )
        return "\n".join(lines)

    lines = [
        "### 1. Most Likely Condition",
        f"The skin image classifier's top prediction for **this upload** is **{name}** ({code}) at **{confidence:.1f}%** confidence.",
    ]
    if top2_name:
        runner = f"Runner-up: {top2_name} ({float(top2_confidence or 0):.1f}%)."
        lines.append(
            f"{runner} The model is not fully certain — clinical review is important."
            if ambiguous
            else runner
        )
    lines.extend(
        [
            "",
            "Classifier ranking for this image:",
            ranking,
            "",
            "*AI screening estimate only — not a confirmed diagnosis. See a dermatologist for any concerning lesion.*",
        ]
    )
    return "\n".join(lines)


def strip_duplicate_section_one(text: str) -> str:
    """Remove a model-written section 1 so it cannot override the classifier lead."""
    if not text or not text.strip():
        return text
    pattern = (
        r"^\s*(?:#{1,3}\s*)?1\.\s*.+?"
        r"(?=\n\s*(?:#{1,3}\s*)?2\.|\Z)"
    )
    return re.sub(pattern, "", text, count=1, flags=re.DOTALL | re.IGNORECASE).lstrip()


def merge_multimodal_answer(lead: str, body: str) -> str:
    body = strip_duplicate_section_one((body or "").strip())
    if not lead:
        return body
    if not body:
        return lead
    return f"{lead}\n\n{body}"


def build_multimodal_prompt(
    question: str,
    history_text: str,
    structured_state: str,
    image_description: str,
    image_context: str,
    text_context: str,
    classifier_context: str,
    language: str,
    classifier_available: bool = False,
) -> str:
    if language == "tr":
        section_rules = (
            """
ÖNEMLİ — Bölüm 1 zaten yazıldı. Bölüm 1 YAZMA. Bu yükleme için yalnızca yukarıdaki SINIFLANDIRICI SONUCU geçerlidir.
Konuşma geçmişindeki eski tahminleri yok say; başka hastalık adı kullanma.
Cevabına tam olarak şu başlıkla başla: ### 2. Neden buna benziyor
Ardından: ### 3. Ne yapabilirsiniz  ve  ### 4. Ne zaman doktora görünmeli
"""
            if classifier_available
            else """
Cevap formatı (bu sırayı takip et):
1. En olası durum — sınıflandırıcı veya ders kitabına göre
2. Neden buna benziyor
3. Ne yapabilirsiniz
4. Ne zaman doktora görünmeli
"""
        )
        return f"""
Sen hasta dostu bir dermatoloji asistanısın.

ÖNEMLİ KURALLAR:
- Cevabı sadece Türkçe ver.
- Kesin tanı koyma.
- Sınıflandırıcı varsa yalnızca bu yüklemedeki sonuca uy; geçmiş mesajlardaki eski tahminleri tekrarlama.
- Ders kitabı metni ve görselleri destekleyici kanıttır.
- Sınıflandırıcı listesinde OLMAYAN hastalık adlarını KULLANMA (sınıflandırıcı varsa).
- Kısa, anlaşılır ve pratik yaz.
- Kullanıcıyı gereksiz korkutma.
{section_rules}
=== SINIFLANDIRICI SONUCU (birincil kanıt — bu yükleme) ===
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
"""
    section_rules = (
        """
IMPORTANT — Section 1 is already written for you. Do NOT write section 1.
For this upload, ONLY the CLASSIFIER RESULT block above is valid for the disease name and confidence.
Ignore disease names from conversation history (they may be from earlier images).
Start your reply with exactly: ### 2. Why it may match
Then: ### 3. What you can do  and  ### 4. When to see a doctor
"""
        if classifier_available
        else """
Answer format (follow this order):
1. Most likely condition — from classifier or textbook if classifier unavailable
2. Why it may match
3. What you can do
4. When to see a doctor
"""
    )
    return f"""
You are a patient-friendly dermatology assistant.

CRITICAL RULES:
- Answer only in English.
- Do not give a confirmed diagnosis.
- When the classifier is available, sections 2–4 must align with the CURRENT classifier result only.
- Never repeat a disease name from an earlier message if it differs from the classifier block above.
- Textbook text and images are supporting evidence.
- When the classifier is available, never use disease names outside its prediction list.
- Keep the answer practical and easy to understand.
- Do not scare the user unnecessarily.
{section_rules}
=== CLASSIFIER RESULT (primary evidence — this upload only) ===
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
"""


def format_classifier_context(classifier_result: dict) -> str:
    top_name = classifier_result.get("predicted_name", "")
    top_confidence = classifier_result.get("confidence", 0.0)
    all_preds = classifier_result.get("all_predictions", [])

    if not top_name or top_confidence == 0.0:
        return "CLASSIFIER_UNAVAILABLE: weights missing or inference failed. Use textbook text and images as primary evidence."

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
    classifier_available = is_classifier_available(classifier_result)
    classifier_lead = build_classifier_answer_lead(classifier_result, language)

    prompt = build_multimodal_prompt(
        question=working_question or "",
        history_text=history_text,
        structured_state=structured_state_text,
        image_description=image_description,
        image_context=image_context,
        text_context=text_context,
        classifier_context=classifier_context,
        language=language,
        classifier_available=classifier_available,
    )

    llm_body = run_llm(prompt)
    answer = merge_multimodal_answer(classifier_lead, llm_body)
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
