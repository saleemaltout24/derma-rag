from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import os
import shutil
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from backend.config import UPLOAD_DIR
from backend.intent_router import classify_user_intent, general_help_response
from backend.rag_pipeline import answer_medical_question, detect_language
from backend.multimodal_pipeline import answer_multimodal_question
from backend.state_manager import create_empty_state

app = FastAPI(title="Dermatology RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions: Dict[str, List[dict]] = {}
session_languages: Dict[str, str] = {}
session_state: Dict[str, dict] = {}


def get_or_create_history(session_id: str) -> List[dict]:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    return chat_sessions[session_id]


def get_or_create_state(session_id: str) -> dict:
    if session_id not in session_state:
        session_state[session_id] = create_empty_state()
    return session_state[session_id]


@app.post("/chat")
async def chat(
    session_id: str = Form("default"),
    question: str = Form(""),
    file: Optional[UploadFile] = File(None),
):
    temp_path = None
    try:
        history = get_or_create_history(session_id)
        state = get_or_create_state(session_id)

        if question:
            intent = classify_user_intent(question)
        else:
            intent = "MEDICAL_QUESTION"

        if intent == "LANGUAGE_CHANGE_TR":
            session_languages[session_id] = "tr"
            answer = "Tabii, bundan sonra Türkçe cevap vereceğim."
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            return {"session_id": session_id, "intent": intent, "answer": answer,
                    "structured_state": state, "history": history}

        if intent == "LANGUAGE_CHANGE_EN":
            session_languages[session_id] = "en"
            answer = "Sure, I will answer in English from now on."
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            return {"session_id": session_id, "intent": intent, "answer": answer,
                    "structured_state": state, "history": history}

        if intent == "GENERAL_HELP" and not file:
            current_language = session_languages.get(session_id) or detect_language(question)
            answer = general_help_response(current_language)
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            return {"session_id": session_id, "intent": intent, "answer": answer,
                    "structured_state": state, "history": history}

        current_language = session_languages.get(session_id)
        if current_language is None:
            current_language = detect_language(question) if question else "en"

        if file:
            temp_path = str(UPLOAD_DIR / file.filename)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            answer, updated_state, image_description, image_matches, text_docs, classification = answer_multimodal_question(
                question=question,
                image_path=temp_path,
                history=history,
                current_state=state,
                forced_language=current_language,
            )
            session_state[session_id] = updated_state
            history.append({"role": "user", "content": question if question else "[uploaded image]"})
            history.append({"role": "assistant", "content": answer})

            return {
                "session_id": session_id,
                "intent": "MULTIMODAL_QUESTION",
                "answer": answer,
                "classification": classification,
                "image_description": image_description,
                "image_matches": image_matches,
                "text_matches": text_docs,
                "structured_state": session_state[session_id],
                "history": history,
            }

        answer, updated_state = answer_medical_question(
            question=question,
            history=history,
            current_state=state,
            forced_language=current_language,
        )
        session_state[session_id] = updated_state
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return {
            "session_id": session_id,
            "intent": "MEDICAL_QUESTION",
            "answer": answer,
            "structured_state": session_state[session_id],
            "history": history,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/reset")
def reset_chat(session_id: str = "default"):
    chat_sessions[session_id] = []
    session_state[session_id] = create_empty_state()
    session_languages.pop(session_id, None)
    return {"message": f"Session '{session_id}' reset."}


@app.get("/ask")
def ask(question: str, session_id: str = "default"):
    try:
        history = get_or_create_history(session_id)
        state = get_or_create_state(session_id)
        intent = classify_user_intent(question)

        if intent == "LANGUAGE_CHANGE_TR":
            session_languages[session_id] = "tr"
            answer = "Tabii, bundan sonra Türkçe cevap vereceğim."
        elif intent == "LANGUAGE_CHANGE_EN":
            session_languages[session_id] = "en"
            answer = "Sure, I will answer in English from now on."
        elif intent == "GENERAL_HELP":
            current_language = session_languages.get(session_id) or detect_language(question)
            answer = general_help_response(current_language)
        else:
            current_language = session_languages.get(session_id) or detect_language(question)
            answer, updated_state = answer_medical_question(
                question=question,
                history=history,
                current_state=state,
                forced_language=current_language,
            )
            session_state[session_id] = updated_state

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        return {
            "session_id": session_id,
            "intent": intent,
            "answer": answer,
            "structured_state": session_state.get(session_id, state),
            "history": history,
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/gradcam")
async def gradcam_endpoint(
    file: UploadFile = File(...),
):
    from backend.gradcam import generate_gradcam
    from backend.skin_classifier import classify_skin_image
    import tempfile

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        classification = classify_skin_image(tmp_path)
        heatmap_b64 = generate_gradcam(tmp_path)
        return {
            "classification": classification,
            "heatmap": heatmap_b64,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
