"""
app/main.py
───────────
Streamlit UI — calls the FastAPI backend at localhost:8000.
Run separately:  streamlit run app/main.py
(Start the API first: uvicorn app.api:app --port 8000)
"""

import os
import json
import logging

import httpx
import streamlit as st
from app.chunk_browser import render_chunk_browser

logger = logging.getLogger(__name__)

# When running via docker-compose the UI container sets API_BASE=http://api:8000
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Agentic RAG Document Assistant")
st.caption("Hybrid retrieval · Cross-encoder reranking · Corrective RAG · Web fallback")

# ── Sidebar — upload ──────────────────────────────────────────────────────────
with st.sidebar:
    view = st.radio("界面", ["对话问答", "PDF 分块"], key="view")
    st.divider()
    st.header("Upload Documents")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Index Document"):
        with st.spinner("Ingesting and indexing …"):
            resp = httpx.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=600,
                trust_env=False,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Indexed {data['chunks']} chunks from {data['filename']}")
        else:
            st.error(f"Error: {resp.text}")

    st.divider()
    st.header("RAGAs Scores")
    if st.button("Load Latest Scores"):
        resp = httpx.get(f"{API_BASE}/metrics", timeout=10, trust_env=False)
        scores = resp.json()
        if "detail" not in scores:
            for k, v in scores.items():
                st.metric(k.replace("_", " ").title(), f"{v:.3f}")
        else:
            st.info(scores["detail"])

if view == "PDF 分块":
    render_chunk_browser(API_BASE)
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("incomplete"):
            st.caption("回答生成中断，请重新提问。")
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** — Page {s['page']}")
        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("Latency", f"{m['latency_ms']} ms")
            cols[1].metric("Rewrites", m['rewrite_count'])
            cols[2].metric("Web used", "Yes" if m['used_web'] else "No")

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        fragments = []
        data = {}

        def answer_chunks():
            with httpx.stream(
                "POST",
                f"{API_BASE}/query/stream",
                json={
                    "question": prompt,
                    "history": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if not m.get("incomplete")
                    ],
                },
                timeout=httpx.Timeout(60, connect=10, write=30, pool=10),
                # Connect directly to our backend; Windows/system proxies may
                # return 502 for local streaming requests.
                trust_env=False,
            ) as response:
                response.raise_for_status()
                response.encoding = "utf-8"
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "status":
                        status.info(event["content"])
                    elif event["type"] == "token":
                        status.empty()
                        fragments.append(event["content"])
                        yield event["content"]
                    elif event["type"] == "done":
                        data.update(event)
                    elif event["type"] == "error":
                        raise ValueError(event["content"])
            if not data:
                raise ValueError("连接提前中断，回答尚未完成，请重试。")
            if "".join(fragments) != data["answer"]:
                raise ValueError("回答接收不完整，请重试。")

        try:
            st.write_stream(answer_chunks())
            status.empty()
            if data["sources"]:
                with st.expander("📚 Sources"):
                    for source in data["sources"]:
                        st.markdown(f"- **{source['source']}** — Page {source['page']}")
            cols = st.columns(3)
            cols[0].metric("Latency", f"{data['latency_ms']} ms")
            cols[1].metric("Rewrites", data["rewrite_count"])
            cols[2].metric("Web used", "Yes" if data["used_web"] else "No")
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "meta": {key: data[key] for key in ("latency_ms", "rewrite_count", "used_web")},
            })
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("UI streaming request failed")
            status.empty()
            if fragments:
                st.session_state.messages.append({
                    "role": "assistant", "content": "".join(fragments), "incomplete": True,
                })
            if isinstance(exc, httpx.TimeoutException):
                st.error("等待响应超时，请稍后重试。已收到的内容已保留。")
            elif isinstance(exc, httpx.ConnectError):
                st.error("无法连接后端，请确认 API 服务已启动。")
            elif isinstance(exc, httpx.HTTPStatusError):
                st.error(f"后端接口返回 HTTP {exc.response.status_code}，请检查 API 服务日志。")
            elif isinstance(exc, httpx.HTTPError):
                st.error(f"流式连接中断（{type(exc).__name__}），已收到的内容已保留，请重试。")
            else:
                st.error(str(exc))
