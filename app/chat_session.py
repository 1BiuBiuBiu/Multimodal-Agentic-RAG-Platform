"""
app/chat_session.py
───────────────────
DeepSeek-style persistent session management.
Stores chat sessions on disk (data/chat_sessions/) with automatic timeline grouping:
- 今天 (Today)
- 昨天 (Yesterday)
- 7 天内 (Past 7 days)
- 更早 (Earlier)
"""

from __future__ import annotations
import json
import time
import uuid
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

CHAT_DIR = Path("data/chat_sessions")
INDEX_FILE = CHAT_DIR / "sessions_index.json"


def ensure_storage() -> None:
    """Ensure session directory and index file exist."""
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        try:
            INDEX_FILE.write_text("[]", encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to create sessions index: {e}")


def _session_sort_key(s: Dict[str, Any]) -> float:
    """Stable sort timestamp: prioritize the latest actual message dialogue time, fallback to creation time."""
    return float(s.get("last_message_at") or s.get("created_at") or s.get("updated_at") or 0)


def list_sessions(include_empty: bool = False) -> List[Dict[str, Any]]:
    """
    List saved chat sessions ordered stably by dialogue/creation timestamp descending.
    Clicking or viewing sessions will never reorder the list.
    """
    ensure_storage()
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            cleaned = []
            for s in data:
                m_cnt = s.get("message_count", 0)
                if m_cnt == 0 and not include_empty:
                    s_file = CHAT_DIR / f"{s.get('id', '')}.json"
                    if s_file.is_file():
                        try:
                            disk_msgs = json.loads(s_file.read_text(encoding="utf-8"))
                            if isinstance(disk_msgs, list) and len(disk_msgs) > 0:
                                s["message_count"] = len(disk_msgs)
                                m_cnt = len(disk_msgs)
                        except Exception:
                            pass
                if include_empty or m_cnt > 0:
                    cleaned.append(s)
            cleaned.sort(key=_session_sort_key, reverse=True)
            return cleaned
    except Exception as e:
        logger.warning(f"Failed to read sessions index: {e}")
    return []


def _save_index(sessions: List[Dict[str, Any]]) -> None:
    ensure_storage()
    try:
        # Filter out empty sessions before persisting index
        filtered = [s for s in sessions if s.get("message_count", 0) > 0]
        INDEX_FILE.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save sessions index: {e}")


def get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Load all messages for a specific session."""
    ensure_storage()
    session_file = CHAT_DIR / f"{session_id}.json"
    if not session_file.exists():
        return []
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception as e:
        logger.warning(f"Failed to read session file {session_file}: {e}")
    return []


def _derive_title(messages: List[Dict[str, Any]], default: str = "新对话") -> str:
    """Derive a concise, meaningful title from the first user question."""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            raw = str(m["content"]).strip().replace("\n", " ")
            if len(raw) > 22:
                return raw[:20] + "…"
            return raw
    return default


def save_session(
    session_id: str,
    messages: List[Dict[str, Any]],
    title: Optional[str] = None,
    is_new_message: bool = True,
) -> Dict[str, Any]:
    """
    Persist session messages to disk and update index entry only if messages exist.
    Timestamp and ordering only update when actual new dialogue messages are recorded.
    """
    ensure_storage()
    session_file = CHAT_DIR / f"{session_id}.json"
    
    # If no messages, do not pollute index or storage with empty skeletons
    if not messages:
        if session_file.is_file():
            try:
                session_file.unlink()
            except Exception:
                pass
        index = [s for s in list_sessions(include_empty=True) if s.get("id") != session_id]
        _save_index(index)
        return {"id": session_id, "title": "新对话", "created_at": time.time(), "updated_at": time.time(), "message_count": 0}

    try:
        session_file.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to write session {session_id}: {e}")

    index = list_sessions(include_empty=True)
    now = time.time()
    found = False

    target_title = title or _derive_title(messages)

    for item in index:
        if item.get("id") == session_id:
            old_count = item.get("message_count", 0)
            # Only update time when actual message count changes or explicitly requested
            if is_new_message and len(messages) != old_count:
                item["updated_at"] = now
                item["last_message_at"] = now
            item["message_count"] = len(messages)
            current_title = item.get("title", "")
            if title:
                item["title"] = title
            elif current_title in ("新对话", "", None) or current_title.startswith("新对话"):
                item["title"] = target_title
            found = True
            break

    if not found:
        new_entry = {
            "id": session_id,
            "title": target_title,
            "created_at": now,
            "updated_at": now,
            "last_message_at": now,
            "message_count": len(messages),
        }
        index.insert(0, new_entry)

    _save_index(index)
    for item in index:
        if item.get("id") == session_id:
            return item
    return {"id": session_id, "title": target_title, "created_at": now, "updated_at": now, "message_count": len(messages)}


def create_session(title: str = "新对话") -> Dict[str, Any]:
    """Create a temporary new chat session object in memory without polluting index or disk."""
    session_id = f"sess_{uuid.uuid4().hex[:10]}"
    now = time.time()
    return {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def delete_session(session_id: str) -> None:
    """Delete a session from disk and remove it from index."""
    ensure_storage()
    session_file = CHAT_DIR / f"{session_id}.json"
    if session_file.exists():
        try:
            session_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to unlink session file {session_file}: {e}")

    index = [s for s in list_sessions() if s.get("id") != session_id]
    _save_index(index)


def group_sessions_by_date(sessions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize sessions like DeepSeek:
    - 今天 (Today)
    - 昨天 (Yesterday)
    - 7 天内 (Past 7 days)
    - 更早 (Earlier)
    """
    groups: Dict[str, List[Dict[str, Any]]] = {
        "今天": [],
        "昨天": [],
        "7 天内": [],
        "更早": [],
    }
    today = date.today()
    yesterday = today - timedelta(days=1)
    seven_days_ago = today - timedelta(days=7)

    for s in sessions:
        ts = _session_sort_key(s) or time.time()
        try:
            s_date = datetime.fromtimestamp(ts).date()
        except Exception:
            s_date = today

        if s_date == today:
            groups["今天"].append(s)
        elif s_date == yesterday:
            groups["昨天"].append(s)
        elif s_date >= seven_days_ago:
            groups["7 天内"].append(s)
        else:
            groups["更早"].append(s)

    return {k: v for k, v in groups.items() if v}
