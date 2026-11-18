"""
app/main.py
───────────
Streamlit UI — calls the FastAPI backend at localhost:8000.
Run separately:  streamlit run app/main.py
(Start the API first: uvicorn app.api:app --port 8000)
"""

import base64
import html
import os
import json
import logging
import re

import httpx
import streamlit as st
from app.chunk_browser import render_chunk_browser
from app.experiments_ui import render_experiments
from app.config_ui import config_controls
from app.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)

# When running via docker-compose the UI container sets API_BASE=http://api:8000
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Global Styles for Citations and Source Jumps ─────────────────────────────
st.markdown("""
<style>
html {
    scroll-behavior: smooth;
}
.citation-link {
    text-decoration: none !important;
    color: #1a73e8 !important;
    background: #e8f0fe !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    border: 1px solid #c2e7ff !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
    display: inline-block !important;
    margin: 0 2px !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
}
.citation-link:hover {
    background: #d2e3fc !important;
    border-color: #aecbfa !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.18) !important;
    transform: translateY(-1px);
}
.source-card {
    scroll-margin-top: 80px;
    border: 1px solid #dadce0;
    border-left: 5px solid #1a73e8;
    border-radius: 6px;
    padding: 12px 16px;
    margin-top: 12px;
    margin-bottom: 12px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(60,64,67,0.08);
    transition: all 0.25s ease-in-out;
}
.source-card:target {
    border-left-color: #ea4335 !important;
    background: #fef7f6 !important;
    box-shadow: 0 0 0 3px rgba(234, 67, 53, 0.25) !important;
    transform: scale(1.008);
}
.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.source-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.82em;
    font-weight: 500;
}
.badge-text { background: #e8f0fe; color: #1967d2; }
.badge-table { background: #e6f4ea; color: #137333; }
.badge-image { background: #fef7e0; color: #b06000; }
.source-meta {
    font-size: 0.84em;
    color: #5f6368;
    margin-bottom: 8px;
}
.source-snippet {
    background: #f8f9fa;
    border: 1px solid #e8eaed;
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 0.92em;
    line-height: 1.6;
    color: #202124;
    white-space: pre-wrap;
    max-height: 380px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 Agentic RAG Document Assistant")
st.caption("Hybrid retrieval · Cross-encoder reranking · Corrective RAG · Web fallback")

# ── Sidebar — upload ──────────────────────────────────────────────────────────
with st.sidebar:
    view = st.radio("界面", ["对话问答", "PDF 分块", "消融评估"], key="view")
    chat_config = PipelineConfig().model_dump()
    valid_config = True
    if view == "对话问答":
        with st.expander("查询与检索模块配置"):
            chat_config = config_controls("chat")
            try:
                PipelineConfig.model_validate(chat_config)
            except ValueError as exc:
                st.error(str(exc))
                valid_config = False
    st.divider()
    st.header("Upload Documents")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Index Document"):
        with st.spinner("Ingesting and indexing …"):
            resp = httpx.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=1900,
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
if view == "消融评估":
    render_experiments(API_BASE)
    st.stop()


# ── Citation Linkify and Source Rendering ─────────────────────────────────────

CITATION_REGEX = re.compile(
    r"\[(?:Source:\s*)?([a-zA-Z0-9_\.\-]+\.pdf),\s*(?:Page|page|p\.)\s*(\d+(?:[–\-]\d+)?)(?:,\s*(?:#|ID:\s*)([a-zA-Z0-9_\.\-:]+))?\]",
    re.IGNORECASE
)


def linkify_citations(text: str, sources: list, msg_id: str) -> str:
    """
    Transform citations like [2501.16566v2.pdf, Page 7, #2501.16566v2.pdf::14] into clickable anchor links
    jumping directly to the corresponding source chunk snippet.
    """
    if not sources or not text:
        return text

    def replace_citation(match):
        raw_full = match.group(0)
        pdf_name = match.group(1).strip()
        page_spec = match.group(2).strip()
        chunk_spec = match.group(3).strip() if match.group(3) else None

        target_idx = None
        # 1. First priority: Exact match by chunk_id
        if chunk_spec:
            for i, s in enumerate(sources):
                s_cid = str(s.get("chunk_id", "")).strip()
                if s_cid == chunk_spec or s_cid.endswith(f"::{chunk_spec}") or chunk_spec.endswith(s_cid):
                    target_idx = i
                    break

        # 2. Second priority: Match by PDF filename + Page
        if target_idx is None:
            for i, s in enumerate(sources):
                src_name = s.get("source", "").strip()
                if src_name.lower() == pdf_name.lower():
                    s_page = str(s.get("page", "")).strip()
                    s_page_end = str(s.get("page_end", s.get("page", ""))).strip()
                    if page_spec in (s_page, f"{s_page}-{s_page_end}") or s_page in page_spec.split("-")[0]:
                        target_idx = i
                        break

        # 3. Fallback: match by PDF filename alone
        if target_idx is None:
            for i, s in enumerate(sources):
                if s.get("source", "").strip().lower() == pdf_name.lower():
                    target_idx = i
                    break

        if target_idx is not None:
            anchor_id = f"src-{msg_id}-{target_idx}"
            matched_chunk = sources[target_idx].get("chunk_id", "")
            id_badge = f", #{matched_chunk.split('::')[-1]}" if "::" in matched_chunk else ""
            return (
                f'<a href="#{anchor_id}" class="citation-link" title="点击跳转至下方来源 {target_idx + 1} ({matched_chunk}) 的原文片段">'
                f'📎 [{pdf_name}, Page {page_spec}{id_badge}]'
                f'</a>'
            )
        return raw_full

    return CITATION_REGEX.sub(replace_citation, text)


def _render_source_cards(sources: list, msg_id: str):
    """Render full context source snippets with anchors, sections and chunk IDs."""
    if not sources:
        return
    with st.expander(f"📚 检索召回的原文片段与多模态来源（共 {len(sources)} 处，点击回答中的引用可直接跳转）", expanded=True):
        for idx, s in enumerate(sources):
            anchor_id = f"src-{msg_id}-{idx}"
            kind = s.get("type", "text")
            type_label = {"text": "正文片段", "table": "表格片段", "image": "图表 / 图片"}.get(kind, "原文片段")
            badge_class = f"badge-{kind}" if kind in ("text", "table", "image") else "badge-text"

            page_str = str(s.get("page", 1))
            if s.get("page_end") and s.get("page_end") != s.get("page"):
                page_str += f"-{s['page_end']}"

            chunk_id = s.get("chunk_id") or f"{s.get('source', 'doc')}::#{idx+1}"
            section = s.get("section") or "未标注章节"
            raw_text = s.get("text", "").strip()

            st.markdown(
                f'<div id="{anchor_id}" class="source-card">'
                f'  <div class="source-header">'
                f'    <span style="font-weight:700; color:#1a73e8; font-size:1.02em;">'
                f'      📍 来源 [{idx + 1}] · <b>{html.escape(s.get("source", ""))}</b> (第 {page_str} 页)'
                f'    </span>'
                f'    <span class="source-badge {badge_class}">{type_label}</span>'
                f'  </div>'
                f'  <div class="source-meta">'
                f'    <b>所属章节：</b>{html.escape(section)} &nbsp;|&nbsp; <b>块 ID：</b><code>{html.escape(chunk_id)}</code>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if s.get("mention_context"):
                st.caption(f"📖 **正文引用解读：** {s['mention_context']}")

            # Render specific snippet contents
            if kind == "text":
                st.markdown(f'<div class="source-snippet">{html.escape(raw_text)}</div>', unsafe_allow_html=True)
            elif kind == "table":
                tables = re.findall(r'<table\b[^>]*>.*?</table\s*>', raw_text, flags=re.I | re.S)
                if tables:
                    st.html('<div style="overflow:auto; max-height:400px; margin-bottom:8px;">' + '\n'.join(tables) + '</div>')
                else:
                    st.markdown(f'<div class="source-snippet">{html.escape(raw_text)}</div>', unsafe_allow_html=True)
            elif kind == "image":
                if s.get("caption"):
                    caps = s["caption"]
                    cap_str = " ".join(caps) if isinstance(caps, list) else str(caps)
                    st.markdown(f"**图注 Caption：** {cap_str}")
                assets = s.get("asset_images") or []
                for img_uri in assets:
                    try:
                        if "," in img_uri:
                            b64_data = img_uri.split(",", 1)[1]
                            st.image(base64.b64decode(b64_data), use_container_width=True)
                        else:
                            st.image(img_uri, use_container_width=True)
                    except Exception:
                        pass
                if not assets and raw_text and not raw_text.startswith("[图片：未识别到描述]"):
                    st.markdown(f'<div class="source-snippet">{html.escape(raw_text)}</div>', unsafe_allow_html=True)


def _render_decision_component(container, decision_trace: list, query_doc_history: list = None, original_query: str = "", is_active: bool = False):
    """
    Render live/persistent transparent Agent Decision & Reflection Timeline.
    - Default view: Shows current active step (highlighted badge/card).
    - Expander view: Click to reveal full history of steps, sub-questions, and dedicated document groups.
    """
    if not decision_trace:
        return

    latest_step = decision_trace[-1]
    step_title = latest_step.get("step", "Agent 正在执行决策")
    step_detail = latest_step.get("detail", "")

    with container.container():
        # 1. 默认展示：当前正在进行的步骤（高亮动画微标）或已完成状态
        if is_active:
            st.markdown(
                f'<div style="background: linear-gradient(90deg, #eff6ff, #f8fafc); border-left: 4px solid #2563eb; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">'
                f'  <div style="font-weight: 700; color: #1d4ed8; font-size: 0.98em; display:flex; align-items:center; gap:6px;">'
                f'    <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#2563eb; box-shadow:0 0 8px #3b82f6;"></span>'
                f'    当前执行：{html.escape(step_title)}'
                f'  </div>'
                f'  <div style="color: #4b5563; font-size: 0.91em; margin-top: 4px;">{html.escape(step_detail)}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px;">'
                f'  <span style="font-weight: 600; color: #15803d; font-size: 0.93em;">'
                f'    ✅ Agent 思考与反思全流程已完成（共 {len(decision_trace)} 个决策节点）'
                f'  </span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # 2. 点开就能看到完整历史步骤
        expander_title = f"🧭 点击展开查看完整思考决策链路与子问题检索组 ({len(decision_trace)} 步)"
        with st.expander(expander_title, expanded=False):
            if original_query:
                st.markdown(f"**🎯 原始用户问题：** `{html.escape(original_query)}`")

            for idx, item in enumerate(decision_trace):
                s_name = item.get("step", f"Step {idx + 1}")
                s_det = item.get("detail", "")
                st.markdown(f"**{idx + 1}. {s_name}**")
                st.markdown(
                    f"<div style='margin-left: 14px; margin-bottom: 8px; color: #374151; font-size: 0.92em; "
                    f"line-height: 1.5; border-left: 2px solid #93c5fd; padding-left: 8px; background: #fafafa; padding-top: 2px; padding-bottom: 2px;'>"
                    f"{html.escape(s_det)}</div>",
                    unsafe_allow_html=True
                )

            # 3. 结构化展示分轮次子问题与专属 Cross-Encoder 检索文档组（当前轮子问题1+文档组1，历史子问题1+文档组1）
            cycles = query_doc_history or []
            if cycles:
                st.divider()
                st.markdown("#### 📑 分轮次子问题与专属 Cross-Encoder 检索文档组")
                for c_idx, c_data in enumerate(cycles):
                    c_num = c_data.get("cycle", c_idx + 1)
                    c_name = c_data.get("cycle_name", f"第 {c_num} 轮检索")
                    is_current = (c_idx == len(cycles) - 1)
                    tag_text = "【当前轮次】" if is_current else "【历史轮次】"

                    st.markdown(f"**{tag_text} {c_name}** (输入查询: `{html.escape(c_data.get('input_query', ''))}`)")
                    sub_groups = c_data.get("sub_groups", [])
                    for g_idx, grp in enumerate(sub_groups):
                        sub_q = grp.get("sub_query", "")
                        docs = grp.get("docs", [])
                        st.markdown(
                            f"<div style='margin-left: 12px; margin-top: 6px; padding: 6px 10px; background: #f1f5f9; border-radius: 4px; font-size: 0.9em;'>"
                            f"<b>子问题 {g_idx + 1}：</b> <code>{html.escape(sub_q)}</code> "
                            f"(独立 Cross-Encoder 重排召回 {len(docs)} 篇专属证据)"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        for d_idx, doc in enumerate(docs[:3]):
                            doc_source = doc.get("source", "doc")
                            doc_page = doc.get("page", 1)
                            doc_id = doc.get("chunk_id", "")
                            doc_snippet = (doc.get("text", "")[:100] + "...").replace("\n", " ")
                            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 文档 [{d_idx+1}] <code>{doc_id}</code> ({doc_source} p.{doc_page}): {doc_snippet}")


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("decision_trace"):
                dec_box = st.empty()
                _render_decision_component(
                    dec_box,
                    msg["decision_trace"],
                    query_doc_history=msg.get("query_doc_history", []),
                    original_query=msg.get("original_query", ""),
                    is_active=False
                )
            rendered = linkify_citations(msg["content"], msg.get("sources", []), f"hist-{msg_idx}")
            st.markdown(rendered, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

        if msg.get("incomplete"):
            st.caption("回答生成中断，请重新提问。")

        if msg.get("degradation_warnings"):
            for warn in msg["degradation_warnings"]:
                st.warning(warn)

        if msg.get("sources"):
            _render_source_cards(msg["sources"], f"hist-{msg_idx}")

        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(4)
            cols[0].metric("耗时", f"{m['latency_ms']} ms")
            cols[1].metric("补检轮数", m['rewrite_count'])
            cols[2].metric("消耗 Token", f"{m.get('total_tokens', '-')}")
            cols[3].metric("联网回退", "是" if m['used_web'] else "否")
        if msg.get("retrieval_trace"):
            with st.expander("🔎 实际检索查询与模块配置"):
                st.json({"config": msg.get("pipeline_config", {}), "trace": msg["retrieval_trace"]}, expanded=False)

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents …", disabled=not valid_config):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        decision_box = st.empty()
        answer_box = st.empty()

        live_steps = [{"step": "🟢 意图路由", "detail": "系统正在分析用户问题，准备启动多模态知识库检索..."}]
        live_cycles = []
        _render_decision_component(decision_box, live_steps, original_query=prompt, is_active=True)

        fragments = []
        data = {}

        def answer_chunks():
            with httpx.stream(
                "POST",
                f"{API_BASE}/query/stream",
                json={
                    "question": prompt,
                    "config": chat_config,
                    "history": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                        if not m.get("incomplete")
                    ],
                },
                timeout=httpx.Timeout(300, connect=15, write=30, pool=10),
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
                    if event["type"] == "decision_step":
                        live_steps.append(event)
                        if event.get("data") and isinstance(event["data"], dict):
                            cyc = event["data"].get("current_cycle")
                            if cyc and cyc not in live_cycles:
                                live_cycles.append(cyc)
                        _render_decision_component(decision_box, live_steps, query_doc_history=live_cycles, original_query=prompt, is_active=True)
                    elif event["type"] == "token":
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
            
            # 渲染最终完成状态的决策组件
            final_trace = data.get("decision_trace", live_steps)
            final_cycles = data.get("query_doc_history", live_cycles)
            _render_decision_component(decision_box, final_trace, query_doc_history=final_cycles, original_query=prompt, is_active=False)

            if data.get("degradation_warnings"):
                for warn in data["degradation_warnings"]:
                    st.warning(warn)
            cur_id = f"live-{len(st.session_state.messages)}"
            if data.get("sources"):
                _render_source_cards(data["sources"], cur_id)
            cols = st.columns(4)
            cols[0].metric("耗时", f"{data['latency_ms']} ms")
            cols[1].metric("补检轮数", data["rewrite_count"])
            cols[2].metric("消耗 Token", f"{data.get('total_tokens', '-')}")
            cols[3].metric("联网回退", "是" if data["used_web"] else "否")
            if data.get("retrieval_trace"):
                with st.expander("🔎 实际检索查询与模块配置"):
                    st.json({"config": data.get("pipeline_config", {}), "trace": data["retrieval_trace"]}, expanded=False)
            st.session_state.messages.append({
                "role": "assistant",
                "content": data["answer"],
                "sources": data["sources"],
                "degradation_warnings": data.get("degradation_warnings", []),
                "retrieval_trace": data.get("retrieval_trace", []),
                "decision_trace": final_trace,
                "query_doc_history": final_cycles,
                "original_query": prompt,
                "token_usage": data.get("token_usage", {}),
                "pipeline_config": data.get("pipeline_config", {}),
                "meta": {
                    "latency_ms": data.get("latency_ms", 0),
                    "rewrite_count": data.get("rewrite_count", 0),
                    "used_web": data.get("used_web", False),
                    "total_tokens": data.get("total_tokens", 0),
                },
            })
            # Rerun so streamed plain text transforms into clickable citation pills
            st.rerun()
        except (httpx.HTTPError, ValueError) as exc:
            logger.exception("UI streaming request failed")
            error_msg = str(exc)
            if isinstance(exc, httpx.TimeoutException):
                error_msg = "等待后端响应超时（300秒），多轮深度反思检索较为耗时，请检查网络或稍后重试。"
            elif isinstance(exc, httpx.ConnectError):
                error_msg = "无法连接后端，请确认 FastAPI 服务已启动。"
            elif isinstance(exc, httpx.HTTPStatusError):
                error_msg = f"后端接口返回 HTTP {exc.response.status_code}，请检查服务日志。"

            content_to_save = "".join(fragments) if fragments else f"⚠️ 回答生成未完成：{error_msg}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": content_to_save,
                "incomplete": True,
            })
            st.error(error_msg)
