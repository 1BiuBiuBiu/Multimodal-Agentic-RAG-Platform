"""
app/chat_worker.py
──────────────────
Background async generation worker for chat sessions.
Decouples query generation from the Streamlit UI lifecycle:
- Survives browser refreshes, tab switching, and session toggling without interruption.
- Background worker continuously receives SSE tokens and decisions, persisting completed results to disk.
- Streamlit UI re-attaches seamlessly to ongoing or completed background tasks.
"""

from __future__ import annotations
import json
import threading
import time
from typing import Dict, List, Any, Optional
from loguru import logger
import httpx

from app.chat_session import get_session_messages, save_session


class SessionTask:
    def __init__(self, session_id: str, prompt: str, chat_config: dict, history: list, api_base: str):
        self.session_id = session_id
        self.prompt = prompt
        self.chat_config = chat_config
        self.history = history
        self.api_base = api_base
        self.status = "running"  # running, done, error
        self.current_status = "正在分析问题意图，准备启动多模态检索..."
        self.live_steps: List[Dict[str, Any]] = [
            {"step": "🟢 意图路由", "detail": "系统正在分析用户问题，准备启动多模态知识库检索..."}
        ]
        self.live_cycles: List[Dict[str, Any]] = []
        self.fragments: List[str] = []
        self.data: Dict[str, Any] = {}
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None


# Process-level active task registry
_REGISTRY: Dict[str, SessionTask] = {}
_REGISTRY_LOCK = threading.Lock()


def get_active_task(session_id: str) -> Optional[SessionTask]:
    """Retrieve the current active or recent background task for a given session."""
    with _REGISTRY_LOCK:
        return _REGISTRY.get(session_id)


def remove_task(session_id: str) -> None:
    """Remove a finished or dismissed task from registry."""
    with _REGISTRY_LOCK:
        _REGISTRY.pop(session_id, None)


def _worker_run(task: SessionTask) -> None:
    """Worker execution loop running in an isolated daemon thread."""
    try:
        with httpx.stream(
            "POST",
            f"{task.api_base}/query/stream",
            json={
                "question": task.prompt,
                "config": task.chat_config,
                "history": [
                    {"role": m["role"], "content": m["content"]}
                    for m in task.history
                    if not m.get("incomplete")
                ],
            },
            timeout=httpx.Timeout(600, connect=15, write=30, pool=10),
            trust_env=False,
        ) as response:
            response.raise_for_status()
            response.encoding = "utf-8"
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                etype = event.get("type")

                if etype == "decision_step":
                    with task.lock:
                        task.live_steps.append(event)
                        if event.get("data") and isinstance(event["data"], dict):
                            cyc = event["data"].get("current_cycle")
                            if cyc and cyc not in task.live_cycles:
                                task.live_cycles.append(cyc)
                elif etype == "status":
                    status_text = event.get("content", "")
                    with task.lock:
                        task.current_status = status_text
                        if task.live_steps:
                            task.live_steps[-1]["detail"] = status_text
                elif etype == "token":
                    with task.lock:
                        task.fragments.append(event["content"])
                elif etype == "done":
                    with task.lock:
                        task.data.update(event)
                elif etype == "error":
                    raise ValueError(event.get("content", "未知生成异常"))

        if not task.data:
            raise ValueError("连接提前中断，未收到完成信号。")

        # Persist full completed response to disk safely
        final_answer = task.data.get("answer", "".join(task.fragments))
        final_trace = task.data.get("decision_trace", task.live_steps)
        final_cycles = task.data.get("query_doc_history", task.live_cycles)

        assistant_msg = {
            "role": "assistant",
            "content": final_answer,
            "sources": task.data.get("sources", []),
            "degradation_warnings": task.data.get("degradation_warnings", []),
            "retrieval_trace": task.data.get("retrieval_trace", []),
            "decision_trace": final_trace,
            "query_doc_history": final_cycles,
            "original_query": task.prompt,
            "token_usage": task.data.get("token_usage", {}),
            "token_breakdown": task.data.get("token_breakdown", []),
            "pipeline_config": task.data.get("pipeline_config", {}),
            "meta": {
                "latency_ms": task.data.get("latency_ms", 0),
                "rewrite_count": task.data.get("rewrite_count", 0),
                "used_web": task.data.get("used_web", False),
                "total_tokens": task.data.get("total_tokens", 0),
            },
        }

        # Read current session messages and append assistant response if not present
        disk_msgs = get_session_messages(task.session_id)
        # Avoid duplicate appending
        if not (disk_msgs and disk_msgs[-1].get("role") == "assistant" and disk_msgs[-1].get("content") == final_answer):
            disk_msgs.append(assistant_msg)
            save_session(task.session_id, disk_msgs, is_new_message=True)

        with task.lock:
            task.status = "done"

    except Exception as exc:
        logger.exception(f"Background generation worker error for session {task.session_id}: {exc}")
        err_text = str(exc)
        with task.lock:
            task.status = "error"
            task.error = err_text

        # If interrupted with partial fragments, persist partial state instead of leaving blank
        partial_content = "".join(task.fragments) if task.fragments else f"⚠️ 回答生成异常：{err_text}"
        disk_msgs = get_session_messages(task.session_id)
        if not (disk_msgs and disk_msgs[-1].get("role") == "assistant"):
            disk_msgs.append({
                "role": "assistant",
                "content": partial_content,
                "incomplete": True,
            })
            save_session(task.session_id, disk_msgs, is_new_message=True)


def start_background_task(
    session_id: str,
    prompt: str,
    chat_config: dict,
    history: list,
    api_base: str = "http://127.0.0.1:8000"
) -> SessionTask:
    """Start or replace a persistent background worker task for a session."""
    with _REGISTRY_LOCK:
        # If previous task is still running, let it continue or replace cleanly
        old = _REGISTRY.get(session_id)
        if old and old.status == "running":
            return old

        task = SessionTask(session_id, prompt, chat_config, history, api_base)
        t = threading.Thread(target=_worker_run, args=(task,), daemon=True, name=f"Worker-{session_id}")
        task.thread = t
        _REGISTRY[session_id] = task
        t.start()
        return task
