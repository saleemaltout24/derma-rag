from backend.image_pipeline import analyze_skin_image
from backend.image_vector_store import search_similar_images
from backend.rag_pipeline import detect_language, format_user_history
from backend.state_extractor import extract_structured_state
from backend.state_manager import merge_state, build_search_query, format_state_for_prompt
from backend.vector_store import search as search_text
from backend.llm import run_llm


def build_text_context(docs: list) -> str:
    parts = []
    for doc in docs:
        source = doc.get("source", "Unknown source")
        section = doc.get("section_title", "Unknown section")
        text = doc.get("text", "").strip()
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

        parts.append(
            f"""
Match {i}
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
    language: str,
) -> str:
    if language == "tr":
        return f"""
Sen hasta dostu bir dermatoloji asistanısın.

Kurallar:
- Cevabı sadece Türkçe ver.
- Kesin tanı koyma.
- Görsel benzerlik ve kitap bilgisini birlikte değerlendir.
- En olası yaygın durumları önce düşün.
- Kısa, anlaşılır ve pratik yaz.
- Kullanıcıyı gereksiz korkutma.

Konuşma geçmişi:
{history_text}

Yapısal durum özeti:
{structured_state}

Kullanıcının yazdığı mesaj:
{question}

Yüklenen görselin açıklaması:
{image_description}

Benzer ders kitabı görselleri:
{image_context}

İlgili ders kitabı metinleri:
{text_context}

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
- Use both visual similarity and textbook text.
- Focus on the most likely common conditions first.
- Keep the answer practical and easy to understand.
- Do not scare the user unnecessarily.

Conversation history:
{history_text}

Structured state:
{structured_state}

User message:
{question}

Uploaded image description:
{image_description}

Similar textbook images:
{image_context}

Relevant textbook text:
{text_context}

Answer format:
1. Most likely condition
2. Why it may match
3. What you can do
4. When to see a doctor
"""


def answer_multimodal_question(
    question: str,
    image_path: str,
    history: list,
    current_state: dict,
    forced_language: str | None = None,
):
    language = forced_language if forced_language else detect_language(question or "")
    history_text = format_user_history(history)

    extracted = extract_structured_state(question or "", history_text)
    updated_state = merge_state(current_state, extracted)

    image_description = analyze_skin_image(image_path)
    image_matches = search_similar_images(image_path, k=3)

    text_query = build_search_query(question or image_description, updated_state, language)
    text_docs = search_text(text_query, k=4)

    image_context = build_image_context(image_matches)
    text_context = build_text_context(text_docs)
    structured_state_text = format_state_for_prompt(updated_state)

    prompt = build_multimodal_prompt(
        question=question or "",
        history_text=history_text,
        structured_state=structured_state_text,
        image_description=image_description,
        image_context=image_context,
        text_context=text_context,
        language=language,
    )

    answer = run_llm(prompt)
    return answer, updated_state, image_description, image_matches, text_docs