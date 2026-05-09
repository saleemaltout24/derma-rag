from backend.image_pipeline import analyze_skin_image
from backend.image_vector_store import search_similar_images
from backend.rag_pipeline import detect_language, format_user_history
from backend.state_extractor import extract_structured_state
from backend.state_manager import merge_state, build_search_query, format_state_for_prompt
from backend.vector_store import search as search_text
from backend.llm import run_llm
from backend.skin_classifier import classify_skin_image


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
        parts.append(f"Match {i}\nPossible condition: {disease}\nBody site: {body_site}\nCaption: {caption}\nMedical description:\n{nearby}")
    return "\n".join(parts)


def build_multimodal_prompt(
    question: str,
    history_text: str,
    structured_state: str,
    image_context: str,
    text_context: str,
    classification: dict,
    language: str,
) -> str:
    predicted = classification.get("predicted_name", "Unknown")
    confidence = classification.get("confidence", 0)
    all_preds = classification.get("all_predictions", [])
    top3 = all_preds[:3]
    top3_text = "\n".join([f"- {p['name']}: {p['confidence']:.1f}%" for p in top3])
    confidence_note = "HIGH confidence" if confidence >= 70 else "MODERATE confidence" if confidence >= 50 else "LOW confidence — treat as approximate"

    if language == "tr":
        return f"""
Sen hasta dostu bir dermatoloji asistanısın.

GÖREVIN: Aşağıdaki derin öğrenme tahminini esas alarak hastaya açıklama yap.
TAHMİN: {predicted} (%{confidence:.1f} - {confidence_note})

KURALLAR:
- Cevabını MUTLAKA "{predicted}" ile başlat.
- Sadece {predicted} hakkında konuş — başka hastalık önerme.
- Kesin tanı koyma.
- Sadece Türkçe cevap ver.
- Aşağıdaki ders kitabı bilgisini yalnızca {predicted} ile ilgiliyse kullan.
- İlgisiz bilgi ekleme.

Derin öğrenme modeli sonuçları:
{top3_text}

Kullanıcının sorusu: {question}

{predicted} hakkında ders kitabı bilgisi:
{text_context}

Cevap formatı:
1. En olası durum: {predicted} (%{confidence:.1f} güven)
2. Bu hastalık nedir ve neden bu tanı
3. Ne yapabilirsiniz
4. Ne zaman doktora görünmeli
"""
    return f"""You are a dermatology assistant. Answer in 4 short sections.

PREDICTION: {predicted} ({confidence:.1f}% confidence)
{top3_text}

User question: {question}

Relevant textbook info:
{text_context[:500]}

Rules:
- Start with "Most likely condition: **{predicted}**"
- Do NOT describe the image visually
- Keep each section to 2-3 sentences max

Answer format — follow this EXACTLY, do not skip any section:
1. Most likely condition: **{predicted}** ({confidence:.1f}% confidence) — write this line exactly
2. What is {predicted}: explain in 2-3 sentences what this disease is
3. What to do: 2-3 bullet points of practical advice
4. When to see a doctor: 1-2 sentences
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

    # Run classifier first
    classification = classify_skin_image(image_path)
    predicted_name = classification['predicted_name']
    print(f"[Classifier] {predicted_name} ({classification['confidence']}%)")

    image_description = analyze_skin_image(image_path)
    image_matches = search_similar_images(image_path, k=3)

    # Search RAG using predicted disease name — not just the question
    text_query = f"{predicted_name} {question}".strip()
    text_docs = search_text(text_query, k=4)

    image_context = build_image_context(image_matches)
    text_context = build_text_context(text_docs)
    structured_state_text = format_state_for_prompt(updated_state)

    prompt = build_multimodal_prompt(
        question=question or "",
        history_text=history_text,
        structured_state=structured_state_text,
        image_context=image_context,
        text_context=text_context,
        classification=classification,
        language=language,
    )

    answer = run_llm(prompt)
    return answer, updated_state, image_description, image_matches, text_docs, classification
