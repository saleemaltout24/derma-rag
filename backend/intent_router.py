from backend.llm import run_llm


ALLOWED_INTENTS = {
    "GENERAL_HELP",
    "LANGUAGE_CHANGE_TR",
    "LANGUAGE_CHANGE_EN",
    "MEDICAL_QUESTION",
    "OUT_OF_SCOPE",
}


def classify_user_intent(message: str) -> str:
    prompt = f"""
You are an intent router for a dermatology assistant.

Classify the user's message into EXACTLY ONE label:

- GENERAL_HELP → asking what the assistant does, capabilities, who you are
- LANGUAGE_CHANGE_TR → user wants answers in Turkish from now on (e.g. Türkçe, Turkish)
- LANGUAGE_CHANGE_EN → user wants answers in English from now on (e.g. English, in English)
- MEDICAL_QUESTION → about skin, scalp, nails, hair on scalp, rash, itch, lesions, acne, eczema, etc.
- OUT_OF_SCOPE → NOT dermatology (e.g. brain, heart, bones, internal medicine, unrelated topics)

IMPORTANT RULES:
- If the message is NOT clearly about skin, scalp, or nails → OUT_OF_SCOPE
- If unsure whether it is dermatology → OUT_OF_SCOPE
- Be strict. Do NOT guess.

Return ONLY the label. No explanation.

User message:
{message}

Label:
"""
    label = run_llm(prompt).strip().upper()

    if label in ALLOWED_INTENTS:
        return label

    return "OUT_OF_SCOPE"


def general_help_response(language: str) -> str:
    if language == "tr":
        return (
            "Ben bir dermatoloji asistanıyım. Cilt, saçlı deri ve tırnaklarla ilgili sorular sorabilirsiniz. "
            "Kaşıntı, kızarıklık, kepek, akne, egzama, sedef, mantar, döküntü ve temel cilt bakımı hakkında yardımcı olabilirim. "
            "İsterseniz cilt fotoğrafı da yükleyebilirsiniz."
        )

    return (
        "I am a dermatology assistant. You can ask about skin, scalp, and nail problems. "
        "I can help with itching, redness, dandruff, acne, eczema, psoriasis, fungal infections, rashes, "
        "and basic skin care. You can also optionally upload a skin image to help me better understand your condition."
    )


def out_of_scope_response(language: str) -> str:
    if language == "tr":
        return (
            "Bu asistan yalnızca cilt, saçlı deri ve tırnaklarla ilgili sorular için tasarlanmıştır. "
            "Lütfen bu konulara yönelik bir soru sorun veya genel bilgi için \"Ne yapabilirsin?\" yazın."
        )
    return (
        "This assistant only answers questions about skin, scalp, and nails. "
        "Please ask a question in that area, or type what you can ask for general help."
    )
