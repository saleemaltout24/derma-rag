from backend.llm import run_llm


ALLOWED_INTENTS = {
    "GENERAL_HELP",
    "LANGUAGE_CHANGE_TR",
    "LANGUAGE_CHANGE_EN",
    "MEDICAL_QUESTION",
    "FOLLOW_UP",
    "CORRECTION",
}


def classify_user_intent(message: str) -> str:
    prompt = f"""
You are an intent router for a dermatology chatbot.

Classify the user's latest message into exactly one label:
- GENERAL_HELP
- LANGUAGE_CHANGE_TR
- LANGUAGE_CHANGE_EN
- MEDICAL_QUESTION
- FOLLOW_UP
- CORRECTION

Meaning:
- GENERAL_HELP: asks what the chatbot can do, what they can ask, who you are, how it works
- LANGUAGE_CHANGE_TR: user wants future answers in Turkish
- LANGUAGE_CHANGE_EN: user wants future answers in English
- MEDICAL_QUESTION: a fresh dermatology question about skin, scalp, nails, lesion, itch, rash, acne, dandruff, eczema, fungus, psoriasis, treatment, diagnosis, or product advice
- FOLLOW_UP: a dermatology follow-up that depends on earlier context
- CORRECTION: user is correcting something previously misunderstood

Return only one label.
No explanation.

User message:
{message}

Label:
"""
    label = run_llm(prompt).strip().upper()

    if label in ALLOWED_INTENTS:
        return label

    return "MEDICAL_QUESTION"


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
        "and basic skin care. You can also upload a skin image."
    )