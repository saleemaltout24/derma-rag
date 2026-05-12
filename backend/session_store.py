import json
import sqlite3
from typing import Any

from backend.config import SESSION_DB_PATH, USE_SQLITE_SESSIONS


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SESSION_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            history_json TEXT NOT NULL,
            state_json TEXT NOT NULL,
            language TEXT
        )
        """
    )
    return conn


def load_session_data(
    session_id: str,
    chat_sessions: dict[str, list[dict]],
    session_state: dict[str, dict[str, Any]],
    session_languages: dict[str, str],
) -> None:
    if not USE_SQLITE_SESSIONS:
        return
    if session_id in chat_sessions and session_id in session_state:
        return

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT history_json, state_json, language FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return

    history_json, state_json, language = row
    chat_sessions[session_id] = json.loads(history_json)
    session_state[session_id] = json.loads(state_json)
    if language:
        session_languages[session_id] = language


def persist_session_data(
    session_id: str,
    chat_sessions: dict[str, list[dict]],
    session_state: dict[str, dict[str, Any]],
    session_languages: dict[str, str],
) -> None:
    if not USE_SQLITE_SESSIONS:
        return

    history = chat_sessions.get(session_id, [])
    state = session_state.get(session_id, {})
    language = session_languages.get(session_id)

    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions(session_id, history_json, state_json, language)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              history_json=excluded.history_json,
              state_json=excluded.state_json,
              language=excluded.language
            """,
            (
                session_id,
                json.dumps(history, ensure_ascii=False),
                json.dumps(state, ensure_ascii=False),
                language,
            ),
        )


def reset_session_data(
    session_id: str,
    chat_sessions: dict[str, list[dict]],
    session_state: dict[str, dict[str, Any]],
    session_languages: dict[str, str],
) -> None:
    if USE_SQLITE_SESSIONS:
        with _get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    chat_sessions[session_id] = []
    session_languages.pop(session_id, None)
