import random
import re
from typing import Any

from backend.llm import run_llm


GREETING_TOKENS = frozenset(
    {
        "hello",
        "hi",
        "hey",
        "hiya",
        "yo",
        "howdy",
        "sup",
        "selam",
        "merhaba",
        "meraba",
        "günaydın",
        "gunaydin",
        "iyigünler",
        "iyigunler",
        "sa",
        "slm",
    }
)

GREETING_FILLERS = frozenset(
    {
        "there",
        "again",
        "you",
        "u",
        "everyone",
        "guys",
        "folks",
        "team",
        "mate",
        "dostum",
        "abi",
        "all",
    }
)

# If the user is clearly asking for capabilities, use help text not a short greeting.
_CAPABILITY_SUBSTRINGS = (
    "what can",
    "who are you",
    "how do you",
    "how does this",
    "tell me about",
    "capabilities",
    "what do you",
    "ne yapabil",
    "neler yap",
    "kim",
    "nasıl çalış",
    "nasil calis",
)


ALLOWED_INTENTS = {
    "GENERAL_HELP",
    "LANGUAGE_CHANGE_TR",
    "LANGUAGE_CHANGE_EN",
    "MEDICAL_QUESTION",
    "OUT_OF_SCOPE",
}


def is_probable_slot_followup(
    message: str,
    history: list[dict],
    current_state: dict[str, Any],
) -> bool:
    """
    Short replies that fill missing body_site / symptoms after a slot clarification,
    without calling the LLM intent router (which has no history and mislabels e.g. "on my face").
    """
    msg = (message or "").strip().lower()
    if not msg:
        return False
    words = msg.split()
    if len(words) > 8:
        return False
    if not history or history[-1].get("role") != "assistant":
        return False

    goal = current_state.get("question_goal")
    if goal not in {
        "symptom_assessment",
        "diagnosis_question",
        "cause_assessment",
        "treatment_advice",
    }:
        return False

    has_body = bool(current_state.get("body_site"))
    has_sym = bool(current_state.get("symptoms"))

    if goal in ("symptom_assessment", "diagnosis_question", "cause_assessment"):
        if has_body and has_sym:
            return False
    elif goal == "treatment_advice":
        if has_body and has_sym:
            return False

    locationish = (
        msg.startswith("on ")
        or msg.startswith("my ")
        or msg.startswith("in ")
        or msg.startswith("the ")
        or " on my " in msg
        or msg.startswith("on my")
        or any(
            w in msg
            for w in (
                "face",
                "scalp",
                "arm",
                "arms",
                "hand",
                "hands",
                "leg",
                "legs",
                "foot",
                "feet",
                "back",
                "chest",
                "neck",
                "elbow",
                "knee",
                "yüz",
                "saç",
                "dirsek",
                "bacak",
                "sırt",
                "göğüs",
                "gogus",
                "ayak",
                "diz",
            )
        )
    )
    symptomish = any(
        w in msg
        for w in (
            "itch",
            "itchy",
            "hurt",
            "hurts",
            "burn",
            "pain",
            "red",
            "rash",
            "dry",
            "scale",
            "flake",
            "bump",
            "spots",
            "kaşıntı",
            "kasinti",
            "yanma",
            "ağrı",
            "agri",
            "kızarık",
            "kizarik",
            "döküntü",
            "dokuntu",
        )
    )

    if goal in ("symptom_assessment", "diagnosis_question", "cause_assessment"):
        if not has_body and (locationish or symptomish):
            return True
        if not has_sym and symptomish:
            return True
        return False

    # treatment_advice: gate needs body OR symptom; user may add the missing piece
    if goal == "treatment_advice":
        if not has_body and locationish:
            return True
        if not has_sym and symptomish:
            return True
    return False


def classify_user_intent(message: str) -> str:
    prompt = f"""
You are an intent router for a dermatology assistant.

Classify the user's message into EXACTLY ONE label:

- GENERAL_HELP → short greetings (hi, hello) OR asking what the assistant does / capabilities / who you are
- LANGUAGE_CHANGE_TR → user wants answers in Turkish from now on (e.g. Türkçe, Turkish)
- LANGUAGE_CHANGE_EN → user wants answers in English from now on (e.g. English, in English)
- MEDICAL_QUESTION → about skin, scalp, nails, hair on scalp, rash, itch, lesions, acne, eczema, etc.
- OUT_OF_SCOPE → NOT dermatology (e.g. brain, heart, bones, internal medicine, unrelated topics)

The system should allow ALL dermatology-related concepts, including:

- skin conditions (acne, eczema, psoriasis)
- lesions, moles, tumors (melanoma, BCC)
- symptoms (itching, redness, rash)
- infections (fungal, bacterial, viral on skin)
- cosmetic skin concerns
- treatments and products

The system should block:
- non-dermatology organs
- unrelated diseases 

IMPORTANT RULES:
- If the message is NOT clearly about the listed concepts → OUT_OF_SCOPE
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


def is_greeting(message: str) -> bool:
    """
    True for short, greeting-only messages (no extra LLM).
    Uses whole-word tokens so substrings like "hi" inside "this" do not match.
    """
    t = (message or "").lower().strip()
    if not t or len(t) > 120:
        return False
    if any(s in t for s in _CAPABILITY_SUBSTRINGS):
        return False
    tokens = re.findall(r"[a-zçğıöşü]+", t)
    if not tokens or len(tokens) > 5:
        return False
    if not any(tok in GREETING_TOKENS for tok in tokens):
        return False
    return all(tok in GREETING_TOKENS or tok in GREETING_FILLERS for tok in tokens)


def greeting_response(language: str) -> str:
    if language == "tr":
        choices = (
            "Merhaba! Bugün cildiniz, saçlı deriniz veya tırnaklarınızla ilgili nasıl yardımcı olabilirim?",
            "Selam! Hangi cilt sorununu konuşmak istersiniz?",
            "Merhaba, buyurun — kaşıntı, döküntü veya lezyon gibi konularda sorabilirsiniz.",
        )
    else:
        choices = (
            "Hi! How can I help with your skin today?",
            "Hey — what skin concern would you like to talk about?",
            "Hello! Feel free to ask about rashes, itching, acne, or other skin, scalp, or nail questions.",
        )
    return random.choice(choices)


def general_help_response(language: str) -> str:
    if language == "tr":
        return (
            "Cilt, saçlı deri ve tırnak konularında yardımcı olabilirim: akne, egzama, kaşıntı, döküntü, mantar enfeksiyonları gibi. "
            "İsterseniz bir fotoğraf da yükleyebilirsiniz."
        )

    return (
        "I can help with skin, scalp, and nail concerns — for example acne, rashes, irritation, or infections on the skin. "
        "You can also upload a photo if that helps."
    )


def out_of_scope_response(language: str) -> str:
    if language == "tr":
        return (
            "Yalnızca cilt, saçlı deri ve tırnaklarla ilgili sorulara yanıt verebilirim. "
            "Lütfen bu alanda bir soru sorun, böylece size daha iyi rehberlik edebilirim."
        )
    return (
        "I can only answer questions about skin, scalp, and nails. "
        "Please ask a question in that area so I can guide you further."
    )
