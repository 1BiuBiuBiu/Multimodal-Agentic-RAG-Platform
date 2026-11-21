"""
app/main.py
───────────
Streamlit UI — calls the FastAPI backend at localhost:8000.
Run separately:  streamlit run app/main.py
(Start the API first: uvicorn app.api:app --port 8000)
"""

from __future__ import annotations
import base64
import hashlib
import html
import os
import json
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from urllib.parse import quote

import httpx
import time
import streamlit as st
import streamlit.components.v1 as components
from app.chat_worker import get_active_task, start_background_task, remove_task
from app.chunk_browser import render_chunk_browser, _load_chunks
from app.experiments_ui import render_experiments
from app.config_ui import config_controls
from app.pipeline_config import PipelineConfig
from pathlib import Path

DATA_MINERU_DIR = Path("data/.mineru")

def resolve_image_path(meta: Dict[str, Any]) -> Optional[Path]:
    """Resolve physical image file from chunk metadata without PyTorch dependencies."""
    if not isinstance(meta, dict):
        return None
    for key in ("image_path", "img_full_path", "image_file"):
        if meta.get(key):
            p = Path(meta[key])
            if p.is_file():
                return p
    rel = meta.get("img_path")
    if not rel:
        return None
    artifact_dir = meta.get("artifact_dir")
    if artifact_dir:
        p = Path(artifact_dir) / rel
        if p.is_file():
            return p
    if DATA_MINERU_DIR.exists():
        for folder in DATA_MINERU_DIR.iterdir():
            if folder.is_dir():
                candidate = folder / rel
                if candidate.is_file():
                    return candidate
    candidate = Path("data") / rel
    if candidate.is_file():
        return candidate
    return None
from app.chat_session import (
    list_sessions,
    create_session,
    save_session,
    get_session_messages,
    delete_session,
    group_sessions_by_date,
    _session_sort_key,
)
from app.export_utils import export_to_markdown, export_to_html

logger = logging.getLogger(__name__)

# When running via docker-compose the UI container sets API_BASE=http://api:8000
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📄",
    layout="wide",
)

# ── Session State & Disk Persistence Initialization ─────────────────────────
if "current_session_id" not in st.session_state:
    all_saved = list_sessions()
    if all_saved:
        st.session_state.current_session_id = all_saved[0]["id"]
        st.session_state.messages = get_session_messages(st.session_state.current_session_id)
    else:
        init_sess = create_session("新对话")
        st.session_state.current_session_id = init_sess["id"]
        st.session_state.messages = []
elif "messages" not in st.session_state:
    st.session_state.messages = get_session_messages(st.session_state.current_session_id)

# ── Global Styles for Citations and Source Jumps ─────────────────────────────
st.markdown("""
<style>
html {
    scroll-behavior: smooth;
}
.citation-link {
    text-decoration: none !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    font-size: 0.86em !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
    margin: 0 2px !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
    vertical-align: middle !important;
}
/* 文本类型切片引用 */
.citation-link.cite-type-text {
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    color: #1d4ed8 !important;
}
.citation-link.cite-type-text:hover {
    background: #dbeafe !important;
    border-color: #93c5fd !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.2) !important;
    transform: translateY(-1px);
}
/* 表格类型切片引用 */
.citation-link.cite-type-table {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    color: #15803d !important;
}
.citation-link.cite-type-table:hover {
    background: #dcfce7 !important;
    border-color: #86efac !important;
    box-shadow: 0 1px 3px rgba(22,163,74,0.2) !important;
    transform: translateY(-1px);
}
/* 图片/图表类型切片引用 */
.citation-link.cite-type-image {
    background: #fff7ed !important;
    border: 1px solid #fed7aa !important;
    color: #c2410c !important;
}
.citation-link.cite-type-image:hover {
    background: #ffedd5 !important;
    border-color: #fdba74 !important;
    box-shadow: 0 1px 3px rgba(194,65,12,0.2) !important;
    transform: translateY(-1px);
}
.cite-tag {
    font-weight: 700 !important;
    font-size: 0.88em !important;
    padding: 0 2px !important;
    border-radius: 3px !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 2px !important;
}
.cite-target-anchor {
    scroll-margin-top: 100px;
}
.return-btn {
    font-size: 0.82em !important;
    color: #1a73e8 !important;
    text-decoration: none !important;
    padding: 3px 9px !important;
    border-radius: 4px !important;
    background: #e8f0fe !important;
    border: 1px solid #d2e3fc !important;
    font-weight: 600 !important;
    transition: all 0.2s ease-in-out !important;
    cursor: pointer !important;
}
.return-btn:hover {
    background: #d2e3fc !important;
    color: #174ea6 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
}
.inline-visual-card {
    margin: 12px 0 16px 0;
    padding: 12px 14px;
    background: #f8fafd;
    border: 1px solid #d2e3fc;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(66, 133, 244, 0.08);
}
.inline-visual-card img {
    max-width: 100%;
    max-height: 460px;
    border-radius: 6px;
    border: 1px solid #dadce0;
    display: inline-block;
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
/* DeepSeek-style Floating Outline TOC Navigator */
.deepseek-toc-widget {
    position: fixed;
    right: 24px;
    bottom: 24px;
    z-index: 99999;
    width: 280px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.deepseek-toc-card {
    background: rgba(255, 255, 255, 0.96);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(218, 220, 224, 0.95);
    border-radius: 12px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    overflow: hidden;
    transition: all 0.25s ease-in-out;
}
.deepseek-toc-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 12px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    cursor: pointer;
    user-select: none;
    list-style: none;
}
.deepseek-toc-header::-webkit-details-marker {
    display: none;
}
.deepseek-toc-header-title {
    font-size: 0.86em;
    font-weight: 700;
    color: #1e293b;
    display: flex;
    align-items: center;
    gap: 6px;
}
.deepseek-toc-list {
    max-height: 320px;
    overflow-y: auto;
    padding: 6px 4px;
}
.deepseek-toc-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 8px;
    margin: 2px 2px;
    border-radius: 6px;
    color: #475569;
    text-decoration: none !important;
    font-size: 0.82em;
    line-height: 1.35;
    transition: background 0.15s ease, color 0.15s ease;
}
.deepseek-toc-item:hover {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}
.deepseek-toc-item-title {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 210px;
}
.deepseek-toc-item-dash {
    color: #94a3b8;
    font-weight: bold;
    font-size: 1.05em;
    margin-left: 6px;
    transition: color 0.15s ease;
}
.deepseek-toc-item:hover .deepseek-toc-item-dash {
    color: #2563eb !important;
}
.deepseek-toc-group {
    margin-bottom: 2px;
    border-radius: 6px;
}
.deepseek-toc-group-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 8px;
    border-radius: 6px;
    cursor: pointer;
    user-select: none;
    list-style: none;
    font-size: 0.82em;
    color: #334155;
    transition: background 0.15s ease, color 0.15s ease;
}
.deepseek-toc-group-summary::-webkit-details-marker {
    display: none;
}
.deepseek-toc-group-summary:hover {
    background: #eff6ff !important;
    color: #1d4ed8 !important;
}
.deepseek-toc-sublist {
    padding-left: 14px;
    margin: 2px 0 6px 6px;
    border-left: 2px solid #e2e8f0;
}
.deepseek-toc-subitem {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 3px 6px;
    font-size: 0.78em;
    color: #64748b;
    text-decoration: none !important;
    border-radius: 4px;
    transition: background 0.15s ease, color 0.15s ease;
}
.deepseek-toc-subitem:hover {
    background: #f1f5f9 !important;
    color: #2563eb !important;
}
.deepseek-toc-subitem-title {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 190px;
}
.deepseek-toc-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 12px;
    background: #f8fafc;
    border-top: 1px solid #e2e8f0;
    font-size: 0.78em;
}
.deepseek-toc-footer a {
    color: #64748b;
    text-decoration: none !important;
    font-weight: 600;
}
.deepseek-toc-footer a:hover {
    color: #2563eb !important;
}
.toc-target-heading {
    scroll-margin-top: 90px;
}
.deepseek-group-title {
    font-size: 0.78em;
    font-weight: 700;
    color: #64748b;
    margin-top: 10px;
    margin-bottom: 2px;
    padding-left: 2px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
/* DeepSeek-style Slim Ultra-Compact Fixed-Height Session Items */
[data-testid="stSidebar"] div.stVerticalBlock > div:has([data-testid="stHorizontalBlock"]) {
    margin-bottom: -11px !important;
    padding-bottom: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    gap: 3px !important;
    align-items: center !important;
    margin-bottom: 2px !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
    height: 30px !important;
    min-height: 30px !important;
    max-height: 30px !important;
    padding: 0 8px !important;
    font-size: 0.82em !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    transition: all 0.15s ease-in-out !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    line-height: 30px !important;
    margin: 0 !important;
    font-size: 0.84em !important;
    text-align: left !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div:nth-child(1) button[kind="primary"] {
    background: #2563eb !important;
    border: 1px solid #1d4ed8 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.2) !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div:nth-child(1) button[kind="primary"] p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div:nth-child(1) button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    color: #334155 !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div:nth-child(1) button[kind="secondary"]:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
    color: #1e293b !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div:nth-child(2) button {
    justify-content: center !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #94a3b8 !important;
    width: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
    opacity: 0.65 !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] div:nth-child(2) button:hover {
    background: #fee2e2 !important;
    color: #dc2626 !important;
    opacity: 1 !important;
    border-radius: 4px !important;
}
/* Non-jumping Decision Trace Styles */
.decision-trace-details {
    margin-top: 8px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #ffffff;
    overflow: hidden;
    transition: all 0.2s ease;
}
.decision-trace-summary {
    cursor: pointer;
    padding: 8px 12px;
    background: #f8fafc;
    font-size: 0.88em;
    font-weight: 600;
    color: #334155;
    user-select: none;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.decision-trace-summary::-webkit-details-marker {
    display: none;
}
.decision-trace-summary:hover {
    background: #f1f5f9;
    color: #1e293b;
}
.decision-trace-inner {
    padding: 12px 14px;
    background: #ffffff;
    max-height: 480px;
    overflow-y: auto;
}
.decision-step-item {
    margin-bottom: 10px;
}
.decision-step-title {
    font-size: 0.9em;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 4px;
}
.decision-step-desc {
    margin-left: 8px;
    padding: 5px 10px;
    background: #f8fafc;
    border-left: 3px solid #60a5fa;
    border-radius: 0 4px 4px 0;
    font-size: 0.86em;
    color: #475569;
    line-height: 1.5;
    white-space: pre-wrap;
}
.export-print-btn {
    background: #ffffff !important;
    color: #1a73e8 !important;
    border: 1px solid #c2e7ff !important;
    padding: 5px 12px !important;
    border-radius: 6px !important;
    font-size: 0.85em !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 4px !important;
}
.export-print-btn:hover {
    background: #e8f0fe !important;
    border-color: #8ab4f8 !important;
    color: #174ea6 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
}
/* Print Stylesheet for High-Fidelity PDF Export */
@media print {
    header, [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stChatInput"],
    [data-testid="stBottom"], [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], [data-testid="stToastContainer"],
    [data-testid="stTooltipContent"], [data-baseweb="tooltip"], [data-baseweb="popover"],
    div[role="tooltip"], .deepseek-toc-widget, .return-btn, .export-print-btn,
    .print-only-hide, button, footer, iframe[height="0"] {
        display: none !important;
    }
    [data-testid="stColumn"]:has(.top-export-marker) {
        display: none !important;
    }
    [data-testid="stColumn"]:has(.top-title-marker) {
        width: 100% !important;
        flex: 1 1 100% !important;
        max-width: 100% !important;
    }
    body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], .main {
        background: #ffffff !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    .stChatMessage, [data-testid="stChatMessage"] {
        box-shadow: none !important;
        break-inside: auto !important;
        page-break-inside: auto !important;
        border-bottom: 1px solid #e2e8f0 !important;
        padding: 12px 0 !important;
    }
    .decision-trace-details, [data-testid="stExpander"], [data-testid="stExpanderDetails"] {
        break-inside: auto !important;
        page-break-inside: auto !important;
        overflow: visible !important;
        max-height: none !important;
    }
    .decision-trace-inner, .source-snippet, .source-table-scroll {
        max-height: none !important;
        overflow: visible !important;
    }
    .decision-step-item, .source-card, .inline-visual-card {
        break-inside: avoid !important;
        page-break-inside: avoid !important;
    }
    .source-card, .inline-visual-card {
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
    }
}
</style>
""", unsafe_allow_html=True)

col_title, col_export = st.columns([0.72, 0.28])
with col_title:
    st.markdown('<span class="top-title-marker" style="display:none;"></span>', unsafe_allow_html=True)
    st.title("📄 Agentic RAG Document Assistant")
    st.caption("Hybrid retrieval · Cross-encoder reranking · Corrective RAG · Web fallback")
with col_export:
    st.markdown('<div class="top-export-marker" style="margin-top: 18px;"></div>', unsafe_allow_html=True)
    if st.button("🖨️ 导出为 PDF / 打印", key="top_export_pdf_btn", use_container_width=True, help="唤起系统打印预览窗口，目标打印机选择「另存为 PDF」即可保存无损高清文档"):
        st.session_state["trigger_pdf_print"] = True

is_printing = bool(st.session_state.pop("trigger_pdf_print", False))
if is_printing:
    st.toast("💡 正在展开全部步骤与引用并调起打印窗口：请在目标打印机中选择「另存为 PDF」保存。", icon="🖨️")

# ── Sidebar — upload ──────────────────────────────────────────────────────────
with st.sidebar:
    view = st.radio("界面", ["对话问答", "PDF 分块", "消融评估"], key="view")
    chat_config = PipelineConfig().model_dump()
    valid_config = True

    if view == "对话问答":
        if st.button("➕ 开启新对话", use_container_width=True, type="primary"):
            new_sess = create_session("新对话")
            st.session_state.current_session_id = new_sess["id"]
            st.session_state.messages = []
            st.rerun()

        all_sess = list_sessions()
        if all_sess:
            grouped_sess = group_sessions_by_date(all_sess)
            for g_title, g_list in grouped_sess.items():
                st.markdown(
                    f'<div class="deepseek-group-title">📅 {g_title} '
                    f'<span style="font-size:0.85em; font-weight:normal; color:#94a3b8;">({len(g_list)})</span></div>',
                    unsafe_allow_html=True,
                )
                for s_item in g_list:
                    s_id = s_item["id"]
                    s_title = s_item.get("title") or "新对话"
                    is_active = (s_id == st.session_state.get("current_session_id"))
                    ts = _session_sort_key(s_item)
                    time_str = datetime.fromtimestamp(ts).strftime("%H:%M")
                    full_time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

                    col_btn, col_del = st.columns([0.88, 0.12])
                    disp_title = s_title if len(s_title) <= 9 else s_title[:8] + "…"
                    btn_label = f"💬 {disp_title} {time_str}" if is_active else f"{disp_title} {time_str}"
                    sess_tooltip = f"【{s_title}】\n🕒 对话时间: {full_time_str}\n📊 消息数: {s_item.get('message_count', 0)} 条"

                    if col_btn.button(
                        btn_label,
                        key=f"sess_btn_{s_id}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                        help=sess_tooltip,
                    ):
                        # Switch session is a pure read operation: NEVER call save_session or touch timestamps!
                        st.session_state.current_session_id = s_id
                        st.session_state.messages = get_session_messages(s_id)
                        st.rerun()
                    if col_del.button("🗑️", key=f"sess_del_{s_id}", help="删除此会话"):
                        delete_session(s_id)
                        if st.session_state.get("current_session_id") == s_id:
                            rem = [x for x in list_sessions() if x["id"] != s_id]
                            if rem:
                                st.session_state.current_session_id = rem[0]["id"]
                                st.session_state.messages = get_session_messages(rem[0]["id"])
                            else:
                                new_s = create_session("新对话")
                                st.session_state.current_session_id = new_s["id"]
                                st.session_state.messages = []
                        st.rerun()

        st.divider()
        with st.expander("📥 导出当前会话", expanded=False):
            curr_msgs = st.session_state.get("messages", [])
            curr_title = "当前对话"
            for s_meta in all_sess:
                if s_meta.get("id") == st.session_state.get("current_session_id"):
                    curr_title = s_meta.get("title", "当前对话")
                    break

            if st.button("🖨️ 打印 / 另存为 PDF", key="sidebar_export_pdf_btn", use_container_width=True):
                st.session_state["trigger_pdf_print"] = True
                st.rerun()

            md_content = export_to_markdown(curr_msgs, curr_title)
            st.download_button(
                label="📄 导出 Markdown (.md)",
                data=md_content,
                file_name=f"{curr_title}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_md_btn",
            )

            html_content = export_to_html(curr_msgs, curr_title)
            st.download_button(
                label="🌐 导出独立 HTML 离线报告",
                data=html_content,
                file_name=f"{curr_title}.html",
                mime="text/html",
                use_container_width=True,
                key="dl_html_btn",
                help="下载独立 HTML 单文件，支持离线全屏浏览，并可随时按 Ctrl+P 导出矢量 PDF",
            )

        with st.expander("⚙️ 查询与检索模块配置", expanded=False):
            chat_config = config_controls("chat")
            try:
                PipelineConfig.model_validate(chat_config)
            except ValueError as exc:
                st.error(str(exc))
                valid_config = False

        with st.expander("📚 知识库文档与索引管理", expanded=False):
            uploaded = st.file_uploader("选择 PDF 文件", type="pdf", key="kb_upload_pdf")
            if uploaded:
                if st.button("⚡ 上传并立即建索引", key="kb_upload_and_index", use_container_width=True, type="primary"):
                    with st.spinner("正在解析 PDF 并增量建立向量索引 …"):
                        resp = httpx.post(
                            f"{API_BASE}/ingest",
                            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                            timeout=1900,
                            trust_env=False,
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        _load_chunks.clear()
                        st.success(f"成功索引 {data['chunks']} 个切片：{data['filename']}")
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.text}")

                if st.button("📥 仅保存至 data 目录（暂不建索引）", key="kb_upload_only", use_container_width=True):
                    resp = httpx.post(
                        f"{API_BASE}/documents/upload",
                        params={"build_index_now": "false"},
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=60,
                        trust_env=False,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        _load_chunks.clear()
                        st.success(f"已保存至 data/{data['filename']}（当前未建立索引）")
                        st.rerun()
                    else:
                        st.error(f"Error: {resp.text}")

            st.divider()
            if st.button("🔄 刷新 data 目录文档状态", key="kb_refresh_docs", use_container_width=True):
                _load_chunks.clear()
                st.rerun()

            kb_docs = []
            kb_fetch_error = None
            try:
                docs_resp = httpx.get(f"{API_BASE}/documents", timeout=10, trust_env=False)
                if docs_resp.status_code == 200:
                    kb_docs = docs_resp.json().get("documents", [])
                else:
                    kb_fetch_error = f"获取文档列表失败（HTTP {docs_resp.status_code}）"
            except Exception:
                kb_fetch_error = "暂无法连接后端文档服务"

            if kb_fetch_error:
                st.caption(kb_fetch_error)
            elif not kb_docs:
                st.caption("data/ 目录暂无 PDF 文件。可直接放入 data/ 或上方上传。")
            else:
                indexed_docs_cnt = sum(1 for d in kb_docs if d.get("indexed_chunks", 0) > 0)
                unindexed_docs_cnt = sum(1 for d in kb_docs if d.get("indexed_chunks", 0) == 0 and d.get("on_disk", True))
                st.caption(f"共 {len(kb_docs)} 篇 · 🟢 已索引 {indexed_docs_cnt} 篇 · 🟠 未索引 {unindexed_docs_cnt} 篇")

                if unindexed_docs_cnt > 0:
                    if st.button(f"🚀 一键索引全部未索引 PDF ({unindexed_docs_cnt} 篇)", key="kb_index_all_unindexed", use_container_width=True, type="primary"):
                        with st.spinner("正在批量解析并索引未建索引文档 …"):
                            res = httpx.post(f"{API_BASE}/documents/index-unindexed", timeout=1900, trust_env=False)
                        if res.status_code == 200:
                            info = res.json()
                            _load_chunks.clear()
                            st.success(f"已完成 {len(info.get('indexed_files', []))} 篇文档索引。")
                            st.rerun()
                        else:
                            st.error(f"批量索引失败: {res.text}")

                for d in kb_docs:
                    fname = d["filename"]
                    c_cnt = d.get("indexed_chunks", 0)
                    on_disk = d.get("on_disk", True)
                    doc_key = hashlib.sha256(fname.encode("utf-8")).hexdigest()[:12]
                    encoded_name = quote(fname, safe="")

                    with st.container(border=True):
                        st.markdown(f"**📄 {html.escape(fname)}**")
                        if not on_disk:
                            st.caption(f"⚠️ 文件已不在 data/ · 残留索引 {c_cnt} 块")
                        elif c_cnt > 0:
                            st.caption(f"🟢 已索引 · {c_cnt} 块")
                        else:
                            st.caption("🟠 未建立索引（仅在 data/ 目录）")

                        if c_cnt == 0:
                            col_act, col_del = st.columns([0.84, 0.16])
                            if col_act.button("⚡ 独立建立索引", key=f"kb_idx_{doc_key}", use_container_width=True):
                                with st.spinner(f"正在索引 {fname} …"):
                                    r = httpx.post(f"{API_BASE}/documents/{encoded_name}/index", timeout=1900, trust_env=False)
                                if r.status_code == 200:
                                    _load_chunks.clear()
                                    st.success(f"已完成 {fname} 索引")
                                    st.rerun()
                                else:
                                    st.error(f"索引失败: {r.text}")
                            if col_del.button("🗑️", key=f"kb_rm_file_{doc_key}", help="从 data/ 目录删除该 PDF 文件"):
                                r = httpx.delete(
                                    f"{API_BASE}/documents/{encoded_name}/index",
                                    params={"delete_file": "true"},
                                    timeout=60,
                                    trust_env=False,
                                )
                                if r.status_code == 200:
                                    _load_chunks.clear()
                                    st.rerun()
                                else:
                                    st.error(f"删除失败: {r.text}")
                        else:
                            if on_disk and st.button("🔄 重新建立索引", key=f"kb_reidx_{doc_key}", use_container_width=True):
                                with st.spinner(f"正在重建 {fname} 索引 …"):
                                    r = httpx.post(f"{API_BASE}/documents/{encoded_name}/index", timeout=1900, trust_env=False)
                                if r.status_code == 200:
                                    _load_chunks.clear()
                                    st.success(f"已重建 {fname} 索引")
                                    st.rerun()
                                else:
                                    st.error(f"重建失败: {r.text}")

                            col_act, col_del = st.columns([0.84, 0.16])
                            if col_act.button("🧹 仅删除索引 (保留文件)", key=f"kb_del_idx_{doc_key}", use_container_width=True):
                                r = httpx.delete(
                                    f"{API_BASE}/documents/{encoded_name}/index",
                                    params={"delete_file": "false"},
                                    timeout=60,
                                    trust_env=False,
                                )
                                if r.status_code == 200:
                                    _load_chunks.clear()
                                    st.rerun()
                                else:
                                    st.error(f"删除索引失败: {r.text}")
                            if col_del.button("🗑️", key=f"kb_del_all_{doc_key}", help="同时删除索引和 data/ 中的 PDF 文件"):
                                r = httpx.delete(
                                    f"{API_BASE}/documents/{encoded_name}/index",
                                    params={"delete_file": "true"},
                                    timeout=60,
                                    trust_env=False,
                                )
                                if r.status_code == 200:
                                    _load_chunks.clear()
                                    st.rerun()
                                else:
                                    st.error(f"删除失败: {r.text}")

        with st.expander("📊 RAGAs 评测指标", expanded=False):
            if st.button("加载最新评测分数", use_container_width=True):
                resp = httpx.get(f"{API_BASE}/metrics", timeout=10, trust_env=False)
                scores = resp.json()
                if "detail" not in scores:
                    def _render_score_group(title_label: str, grp: dict):
                        st.markdown(f"**{title_label}**")
                        for k, v in grp.items():
                            if isinstance(v, dict):
                                continue
                            if k == "count":
                                st.caption(f"评测题数：{v} 题")
                            elif k == "avg_tokens":
                                st.metric("平均消耗 Token", f"{v:.1f}")
                            elif k == "avg_retrieval_count":
                                st.metric("平均检索轮次", f"{v:.2f}")
                            elif k == "latency_ms":
                                st.metric("平均耗时 (ms)", f"{v:.1f}")
                            elif isinstance(v, (int, float)):
                                st.metric(k.replace("_", " ").title(), f"{v:.3f}")

                    _render_score_group("📊 总体指标 (Overall)", scores)
                    if isinstance(scores.get("simple"), dict):
                        st.divider()
                        _render_score_group("🟢 简单问题 (Simple)", scores["simple"])
                    if isinstance(scores.get("complex"), dict):
                        st.divider()
                        _render_score_group("🟣 复杂问题 (Complex)", scores["complex"])
                else:
                    st.info(scores["detail"])
    else:
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
                    if isinstance(v, (int, float)) and k != "count":
                        st.metric(k.replace("_", " ").title(), f"{v:.3f}" if v <= 1.0 else f"{v:.1f}")
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
    r"\[([^\[\]]+?\.pdf\s*,[^\[\]]*?#[^\[\]]+?)\](?:\s*([。，；;.!?]))?",
    re.IGNORECASE
)


def extract_and_anchor_headings(text: str, msg_idx: int) -> tuple[str, List[Dict[str, str]]]:
    """
    Extract headings and prominent bold items from markdown text, inject inline HTML span anchors,
    and return (modified_text, list_of_toc_entries).
    Keeps leading Markdown syntax ('### ', '- ', '1. ') at line start so headings and lists render cleanly.
    """
    if not text:
        return text, []

    toc = []
    h_idx = 0
    lines = text.split("\n")
    out_lines = []

    heading_pattern = re.compile(r"^(\s*#{1,4}\s+)(.+)$")
    bold_item_pattern = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)?(\*\*([^\*]{3,45})\*\*[:：]?.*)$")

    for line in lines:
        stripped = line.strip()
        m_head = heading_pattern.match(line)
        m_bold = bold_item_pattern.match(line)

        if m_head:
            h_idx += 1
            anc = f"nav-head-{msg_idx}-{h_idx}"
            clean_title = m_head.group(2).strip()
            clean_title = re.sub(r"\[.*?\]\(.*?\)", "", clean_title)
            clean_title = re.sub(r"<.*?>", "", clean_title).strip()
            disp_title = clean_title if len(clean_title) <= 24 else clean_title[:22] + "…"
            toc.append({"anchor": anc, "title": disp_title})
            out_lines.append(f'{m_head.group(1)}<span id="{anc}" class="toc-target-heading"></span>{m_head.group(2)}')
        elif m_bold and not stripped.startswith("#"):
            prefix = m_bold.group(1) or ""
            rest = m_bold.group(2)
            clean_b = m_bold.group(3).strip()
            if len(clean_b) > 4 and not clean_b.startswith("http") and not clean_b.startswith("data:"):
                h_idx += 1
                anc = f"nav-head-{msg_idx}-{h_idx}"
                disp_title = clean_b if len(clean_b) <= 24 else clean_b[:22] + "…"
                toc.append({"anchor": anc, "title": disp_title})
                out_lines.append(f'{prefix}<span id="{anc}" class="toc-target-heading"></span>{rest}')
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)

    return "\n".join(out_lines), toc


def render_floating_toc(toc_groups: List[Dict[str, Any]]):
    """
    Render DeepSeek-style floating outline navigation widget.
    Default view shows only the high-level user question(s).
    Sub-headings/sections are collapsed inside accordion details, clean and uncluttered.
    """
    if not toc_groups:
        return

    items_html_parts = []
    total_q_count = len(toc_groups)

    for g in toc_groups:
        q_anc = html.escape(g["anchor"])
        q_tit = html.escape(g["title"])
        subs = g.get("sub_items", [])

        if not subs:
            items_html_parts.append(
                f'<a href="#{q_anc}" class="deepseek-toc-item" title="{q_tit}">'
                f'  <span class="deepseek-toc-item-title">{q_tit}</span>'
                f'  <span class="deepseek-toc-item-dash">-</span>'
                f'</a>'
            )
        else:
            sub_links = []
            for s in subs:
                s_anc = html.escape(s["anchor"])
                s_tit = html.escape(s["title"])
                sub_links.append(
                    f'<a href="#{s_anc}" class="deepseek-toc-subitem" title="{s_tit}">'
                    f'  <span class="deepseek-toc-subitem-title">{s_tit}</span>'
                    f'  <span style="color:#cbd5e1; font-size:0.8em;">-</span>'
                    f'</a>'
                )
            sub_html = "\n".join(sub_links)

            # Collapsed by default as requested by user ("剩余的子内容应该是默认不是展开的")
            items_html_parts.append(
                f'<details class="deepseek-toc-group">'
                f'  <summary class="deepseek-toc-group-summary" title="点击展开/收起该回答的小节大纲">'
                f'    <a href="#{q_anc}" style="text-decoration:none; color:inherit; flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="{q_tit}">'
                f'      {q_tit}'
                f'    </a>'
                f'    <span style="font-size:0.72em; color:#94a3b8; margin-left:4px;" title="包含 {len(subs)} 个小节">▼</span>'
                f'  </summary>'
                f'  <div class="deepseek-toc-sublist">'
                f'    {sub_html}'
                f'  </div>'
                f'</details>'
            )

    items_html = "\n".join(items_html_parts)

    st.markdown(
        f'<div class="deepseek-toc-widget">'
        f'  <details class="deepseek-toc-card" open>'
        f'    <summary class="deepseek-toc-header" title="点击展开/收起长文大纲快速定位">'
        f'      <span class="deepseek-toc-header-title">'
        f'        <span>📑</span> 快速定位大纲 <span style="font-size:0.85em; font-weight:normal; color:#64748b;">({total_q_count})</span>'
        f'      </span>'
        f'      <span style="font-size:0.75em; color:#94a3b8;">▼</span>'
        f'    </summary>'
        f'    <div class="deepseek-toc-list">'
        f'      {items_html}'
        f'    </div>'
        f'    <div class="deepseek-toc-footer">'
        f'      <a href="#page-top">⬆ 回到顶部</a>'
        f'      <a href="#page-bottom">⬇ 直达底部</a>'
        f'    </div>'
        f'  </details>'
        f'</div>',
        unsafe_allow_html=True,
    )


def format_latex_formulas(text: str) -> str:
    r"""
    Detects LaTeX math expressions incorrectly enclosed in parentheses like:
      (Z_{av}\in\mathbb{R}^{(t_a+t_v)\times d})
    and transforms them into standard KaTeX markdown:
      $Z_{av}\in\mathbb{R}^{(t_a+t_v)\times d}$
    
    Also handles standard LaTeX \( ... \) -> $ ... $ and \[ ... \] -> $$ ... $$.
    Also wraps orphaned LaTeX math commands like \mathbf{...} or \mathbb{...} in $ ... $.
    """
    if not text:
        return text

    # 1. Standard LaTeX delimiters \( ... \) and \[ ... \]
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    if "(" not in text and "\\" not in text:
        return text

    math_indicators = (
        r"\in", r"\mathbb", r"\times", r"\sum", r"\prod", r"\frac", r"\sqrt",
        r"\alpha", r"\beta", r"\gamma", r"\theta", r"\lambda", r"\sigma", r"\mu",
        r"\partial", r"\nabla", r"\mathcal", r"\mathbf", r"\mathrm", r"\text",
        r"\hat", r"\bar", r"\tilde", r"\vec", r"\cdot", r"\le", r"\ge", r"\neq",
        r"\approx", r"\infty", r"\cup", r"\cap", r"\subset", r"\subseteq",
        r"\forall", r"\exists", r"^{", r"_{", r"\to", r"\leftarrow", r"\rightarrow",
        r"\pm", r"\mp", r"\equiv", r"\sim", r"\propto", r"\circ", r"\bullet"
    )

    # Convert (math_expr) to $math_expr$
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "$":
            is_double = (i + 1 < n and text[i + 1] == "$")
            delim = "$$" if is_double else "$"
            k = text.find(delim, i + len(delim))
            if k != -1:
                result.append(text[i : k + len(delim)])
                i = k + len(delim)
                continue
        if text[i] == "(":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            if depth == 0:
                inner = text[i + 1 : j - 1]
                has_math = any(ind in inner for ind in math_indicators)
                has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in inner)
                if has_math and not has_chinese and "\n\n" not in inner and 2 < len(inner) < 400:
                    result.append(f"${inner.strip()}$")
                    i = j
                    continue
        result.append(text[i])
        i += 1
    text = "".join(result)

    # Wrap orphaned math commands like \mathbf{H}_a outside of $...$
    parts = text.split("$")
    for idx in range(0, len(parts), 2):
        p = parts[idx]
        if "\\" in p:
            p = re.sub(
                r'(\\(?:mathbf|mathbb|mathcal|mathrm|text)\{[^}]+\}(?:\s*(?:\\rightarrow|->|\^\{[^}]+\}|_\{[^}]+\}|[a-zA-Z0-9_\^]))*)',
                r'$\1$',
                p
            )
            p = p.replace("$$", "")
            parts[idx] = p
    return "$".join(parts)


def _get_visual_data_uri(v_meta: dict) -> Optional[str]:
    """Retrieve or generate base64 data URI for an image/table chunk or mentioned visual."""
    if not isinstance(v_meta, dict):
        return None
    assets = v_meta.get("asset_images") or []
    for a in assets:
        if a and isinstance(a, str) and (a.startswith("data:image/") or a.startswith("http")):
            return a
    p = resolve_image_path(v_meta)
    if p and p.is_file():
        try:
            raw = p.read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")
            suffix = p.suffix.lower().lstrip(".")
            mime = "image/png" if suffix == "png" else "image/jpeg"
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            logger.warning(f"Failed to read image for UI preview: {p}: {e}")
    return None


def linkify_citations(text: str, sources: list, msg_id: str) -> str:
    """
    Transform citations like [2501.16566v2.pdf, Page 7, Section: 4.2 Main Results, #2501.16566v2.pdf::14]
    into clickable anchor links jumping directly to the corresponding source chunk snippet,
    embed return anchors for two-way smooth jumps, and inline associated visual figures/tables
    directly under the citing paragraph.
    """
    if not text:
        return text

    # Step 1: Format LaTeX math formulas (e.g. KaTeX $...$)
    text = format_latex_formulas(text)

    if not sources:
        return text

    seen_inline_visual_ids = set()

    def replace_citation(match):
        raw_full = match.group(0)
        inner = match.group(1).strip()
        punct = match.group(2) or ""

        parts = [p.strip() for p in inner.split(",") if p.strip()]
        if len(parts) < 2:
            return raw_full

        pdf_name = re.sub(r"^(?:Source:|citation:|Citation:)\s*", "", parts[0], flags=re.IGNORECASE).strip()
        chunk_spec = re.sub(r"^(?:#|ID:\s*)", "", parts[-1], flags=re.IGNORECASE).strip()

        target_idx = None
        # 1. Exact match by chunk_id
        if chunk_spec:
            for i, s in enumerate(sources):
                s_cid = str(s.get("chunk_id", "")).strip()
                if s_cid == chunk_spec:
                    target_idx = i
                    break
            # 2. Match PDF filename + chunk_id suffix
            if target_idx is None:
                for i, s in enumerate(sources):
                    s_cid = str(s.get("chunk_id", "")).strip()
                    s_src = str(s.get("source", "")).strip().lower()
                    if s_src == pdf_name.lower() and (
                        s_cid.endswith(f"::{chunk_spec}") or chunk_spec.endswith(s_cid)
                    ):
                        target_idx = i
                        break
            # 3. Cross-source suffix fallback
            if target_idx is None:
                for i, s in enumerate(sources):
                    s_cid = str(s.get("chunk_id", "")).strip()
                    if s_cid and (s_cid.endswith(f"::{chunk_spec}") or chunk_spec.endswith(s_cid)):
                        target_idx = i
                        break

        # 4. Fallback: match by PDF filename alone
        if target_idx is None:
            for i, s in enumerate(sources):
                if str(s.get("source", "")).strip().lower() == pdf_name.lower():
                    target_idx = i
                    break

        if target_idx is not None:
            anchor_id = f"src-{msg_id}-{target_idx}"
            return_id = f"cite-return-{msg_id}-{target_idx}"
            s = sources[target_idx]
            matched_chunk = str(s.get("chunk_id") or chunk_spec).strip()
            s_type = s.get("type", "text")

            # Determine chunk type icon, label, and style badge strictly from metadata
            if s_type == "image":
                type_icon = "🖼️"
                type_label = "图片"
                type_cls = "cite-type-image"
            elif s_type == "table":
                type_icon = "📊"
                type_label = "表格"
                type_cls = "cite-type-table"
            else:
                type_icon = "📄"
                type_label = "文本"
                type_cls = "cite-type-text"

            # Assemble page, section, and short chunk ID strictly from source metadata
            disp_pdf = str(s.get("source") or pdf_name).strip()
            s_page = s.get("page", 1)
            s_page_end = s.get("page_end")
            page_str = f"{s_page}-{s_page_end}" if s_page_end and s_page_end != s_page else str(s_page)
            s_sec = str(s.get("section") or "").strip()
            section_badge = f", Section: {html.escape(s_sec)}" if s_sec and s_sec != "未标注章节" else ""
            short_id = matched_chunk.split("::")[-1] if "::" in matched_chunk else matched_chunk
            id_badge = f", #{html.escape(short_id)}" if short_id else ""

            # Keep punctuation directly attached to the pill link to prevent orphaned dots
            link_html = (
                f'<a id="{return_id}" href="#{anchor_id}" class="citation-link {type_cls} cite-target-anchor" '
                f'title="点击跳转至下方来源 [{target_idx + 1}] ({type_label} · {html.escape(matched_chunk)}) 的原文片段">'
                f'<span class="cite-tag">{type_icon} {type_label}</span>'
                f'<span>[{html.escape(disp_pdf)}, Page {html.escape(page_str)}{section_badge}{id_badge}]</span>'
                f'</a>{punct}'
            )

            # Check if this source or its mentioned visuals have images/tables to inline
            visuals_to_render = []
            if s.get("type") in ("image", "table"):
                visuals_to_render.append(s)
            if s.get("mentioned_visuals"):
                for v in s["mentioned_visuals"]:
                    if isinstance(v, dict):
                        visuals_to_render.append(v)

            visual_cards = []
            for v in visuals_to_render:
                v_key = v.get("chunk_id") or v.get("image_path") or str(v.get("caption"))
                if not v_key or v_key in seen_inline_visual_ids:
                    continue
                seen_inline_visual_ids.add(v_key)

                v_kind = v.get("type", "image")
                v_icon = "📊" if v_kind == "image" else "📋"
                v_title = "架构/曲线图切片" if v_kind == "image" else "数据表格切片"
                cap = v.get("caption", "")
                cap_str = " ".join(cap) if isinstance(cap, list) else str(cap or "")

                img_uri = _get_visual_data_uri(v)
                table_html = ""
                if v_kind == "table" and not img_uri:
                    raw_t = v.get("text", "")
                    t_matches = re.findall(r'<table\b[^>]*>.*?</table\s*>', raw_t, flags=re.I | re.S)
                    if t_matches:
                        table_html = '\n'.join(t_matches)

                if img_uri:
                    card = (
                        f'<div class="inline-visual-card">'
                        f'  <div style="font-size:0.86em; font-weight:700; color:#1a73e8; margin-bottom:6px;">'
                        f'    {v_icon} 原文引用{v_title} · <b>{html.escape(v.get("source", disp_pdf))}</b> (第 {v.get("page", page_str)} 页)'
                        f'  </div>'
                        f'  <div style="text-align:center; margin:6px 0;">'
                        f'    <img src="{img_uri}" alt="{html.escape(cap_str)}" />'
                        f'  </div>'
                        f'  {f"<div style=\'font-size:0.82em; color:#5f6368; margin-top:6px; line-height:1.4;\'><b>Caption:</b> {html.escape(cap_str)}</div>" if cap_str else ""}'
                        f'</div>'
                    )
                    visual_cards.append(card)
                elif table_html:
                    card = (
                        f'<div class="inline-visual-card">'
                        f'  <div style="font-size:0.86em; font-weight:700; color:#137333; margin-bottom:6px;">'
                        f'    📋 原文引用数据表格 · <b>{html.escape(v.get("source", disp_pdf))}</b> (第 {v.get("page", page_str)} 页)'
                        f'  </div>'
                        f'  <div style="overflow-x:auto; max-height:360px; margin:6px 0;">{table_html}</div>'
                        f'  {f"<div style=\'font-size:0.82em; color:#5f6368; margin-top:6px; line-height:1.4;\'><b>Caption:</b> {html.escape(cap_str)}</div>" if cap_str else ""}'
                        f'</div>'
                    )
                    visual_cards.append(card)

            if visual_cards:
                return link_html + "\n\n" + "\n\n".join(visual_cards) + "\n\n"
            return link_html

        return raw_full

    return CITATION_REGEX.sub(replace_citation, text)


def _render_source_cards(sources: list, msg_id: str, force_expand: bool = False):
    """Render full context source snippets with anchors, sections, chunk IDs, and back-to-answer buttons."""
    if not sources:
        return
    with st.expander(f"📚 检索召回的原文片段与多模态来源（共 {len(sources)} 处，点击回答中的引用可直接跳转）", expanded=force_expand):
        for idx, s in enumerate(sources):
            anchor_id = f"src-{msg_id}-{idx}"
            return_id = f"cite-return-{msg_id}-{idx}"
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
                f'    <div style="display:flex; align-items:center; gap:8px;">'
                f'      <span class="source-badge {badge_class}">{type_label}</span>'
                f'      <a href="#{return_id}" class="return-btn" title="点击返回回答中引用此来源的位置">↩ 返回回答位置</a>'
                f'    </div>'
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
                    st.html('<div class="source-table-scroll" style="overflow:auto; max-height:400px; margin-bottom:8px;">' + '\n'.join(tables) + '</div>')
                else:
                    st.markdown(f'<div class="source-snippet">{html.escape(raw_text)}</div>', unsafe_allow_html=True)
            elif kind == "image":
                if s.get("caption"):
                    caps = s["caption"]
                    cap_str = " ".join(caps) if isinstance(caps, list) else str(caps)
                    st.markdown(f"**图注 Caption：** {cap_str}")
                assets = s.get("asset_images") or []
                if not assets:
                    uri = _get_visual_data_uri(s)
                    if uri:
                        assets = [uri]
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


def _render_token_breakdown(token_breakdown: list, token_usage: dict = None, force_expand: bool = False):
    """Render step-by-step token consumption table with Prompt/Completion/Total breakdown."""
    if not token_breakdown:
        return
    with st.expander("🪙 详细 Token 分步消耗明细", expanded=force_expand):
        total_p = sum(item.get("prompt_tokens", 0) for item in token_breakdown)
        total_c = sum(item.get("completion_tokens", 0) for item in token_breakdown)
        grand_total = sum(item.get("total_tokens", 0) for item in token_breakdown)
        
        md_lines = [
            "| 步骤环节 | 输入 Prompt Tokens | 输出 Completion Tokens | 阶段总耗 Total Tokens | 消耗占比 |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for item in token_breakdown:
            step = item.get("step", "未知步骤")
            p = item.get("prompt_tokens", 0)
            c = item.get("completion_tokens", 0)
            tot = item.get("total_tokens", p + c)
            pct = f"{(tot / grand_total * 100):.1f}%" if grand_total > 0 else "-"
            md_lines.append(f"| **{step}** | {p:,} | {c:,} | **{tot:,}** | `{pct}` |")
        
        md_lines.append(f"| **📊 全链路累计总计** | **{total_p:,}** | **{total_c:,}** | **{grand_total:,}** | `100.0%` |")
        st.markdown("\n".join(md_lines))


def _render_decision_component(container, decision_trace: list, query_doc_history: list = None, original_query: str = "", is_active: bool = False, live_status_override: Optional[str] = None, force_expand: bool = False):
    """
    Render live/persistent transparent Agent Decision & Reflection Timeline.
    Uses native HTML details with persistent open-state tracking during live streaming
    so that advancing to new steps does NOT collapse the user's reading view or jump the page.
    """
    if not decision_trace:
        return

    latest_step = decision_trace[-1]
    step_title = latest_step.get("step", "Agent 正在执行决策")
    step_detail = live_status_override or latest_step.get("detail", "")

    # Top active/done banner
    if is_active:
        banner_html = (
            f'<div style="background: linear-gradient(90deg, #eff6ff, #f8fafc); border-left: 4px solid #2563eb; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">'
            f'  <div style="font-weight: 700; color: #1d4ed8; font-size: 0.98em; display:flex; align-items:center; gap:6px;">'
            f'    <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#2563eb; box-shadow:0 0 8px #3b82f6;"></span>'
            f'    当前执行：{html.escape(step_title)}'
            f'  </div>'
            f'  <div style="color: #4b5563; font-size: 0.91em; margin-top: 4px;">{html.escape(step_detail)}</div>'
            f'</div>'
        )
    else:
        banner_html = (
            f'<div style="background: #f0fdf4; border-left: 4px solid #16a34a; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px;">'
            f'  <span style="font-weight: 600; color: #15803d; font-size: 0.93em;">'
            f'    ✅ Agent 思考与反思全流程已完成（共 {len(decision_trace)} 个决策节点）'
            f'  </span>'
            f'</div>'
        )

    # Step items list
    steps_parts = []
    if original_query:
        steps_parts.append(f'<div style="margin-bottom:10px; font-size:0.9em;">🎯 <b>原始用户问题：</b> <code>{html.escape(original_query)}</code></div>')

    for idx, item in enumerate(decision_trace):
        s_name = item.get("step", f"Step {idx + 1}")
        s_det = item.get("detail", "")
        steps_parts.append(
            f'<div class="decision-step-item">'
            f'  <div class="decision-step-title">{idx + 1}. {html.escape(s_name)}</div>'
            f'  <div class="decision-step-desc">{html.escape(s_det)}</div>'
            f'</div>'
        )

    # Cycles & sub-queries
    cycles = query_doc_history or []
    if cycles:
        steps_parts.append('<div style="margin:14px 0 8px 0; border-top:1px solid #e2e8f0; padding-top:10px; font-weight:700; font-size:0.92em; color:#1e293b;">📑 分轮次子问题与专属 Cross-Encoder 检索文档组</div>')
        for c_idx, c_data in enumerate(cycles):
            c_num = c_data.get("cycle", c_idx + 1)
            c_name = c_data.get("cycle_name", f"第 {c_num} 轮检索")
            is_current = (c_idx == len(cycles) - 1)
            tag_text = "【当前轮次】" if is_current else "【历史轮次】"
            in_q = html.escape(c_data.get("input_query", ""))
            steps_parts.append(f'<div style="font-size:0.88em; font-weight:600; color:#334155; margin-top:8px;">{tag_text} {html.escape(c_name)} <span style="font-weight:normal; color:#64748b;">(输入: <code>{in_q}</code>)</span></div>')
            
            sub_groups = c_data.get("sub_groups", [])
            for g_idx, grp in enumerate(sub_groups):
                sub_q = html.escape(grp.get("sub_query", ""))
                docs = grp.get("docs", [])
                steps_parts.append(
                    f'<div style="margin-left: 10px; margin-top: 6px; padding: 6px 10px; background: #f1f5f9; border-radius: 4px; font-size: 0.86em;">'
                    f'<b>子问题 {g_idx + 1}：</b> <code>{sub_q}</code> '
                    f'<span style="color:#64748b;">(独立 Cross-Encoder 重排召回 {len(docs)} 篇专属证据)</span>'
                    f'</div>'
                )
                for d_idx, doc in enumerate(docs[:3]):
                    doc_source = html.escape(doc.get("source", "doc"))
                    doc_page = doc.get("page", 1)
                    doc_id = html.escape(doc.get("chunk_id", ""))
                    doc_snip = html.escape((doc.get("text", "")[:100] + "...").replace("\n", " "))
                    steps_parts.append(
                        f'<div style="margin-left: 24px; font-size: 0.82em; color: #475569; margin-top: 2px;">'
                        f'📄 文档 [{d_idx+1}] <code>{doc_id}</code> ({doc_source} p.{doc_page}): {doc_snip}'
                        f'</div>'
                    )

    trace_id = "live-dec-trace" if is_active else f"hist-dec-trace-{abs(hash(str(decision_trace))) % 100000}"
    script_hook = ""
    if is_active:
        script_hook = (
            f'<script>'
            f'(function() {{'
            f'  const el = document.getElementById("{trace_id}");'
            f'  if (el) {{'
            f'    if (sessionStorage.getItem("live_dec_open") === "1") {{ el.setAttribute("open", ""); }}'
            f'    el.addEventListener("toggle", function() {{'
            f'      sessionStorage.setItem("live_dec_open", el.open ? "1" : "0");'
            f'    }});'
            f'  }}'
            f'}})();'
            f'</script>'
        )

    open_attr = " open" if force_expand else ""
    full_component_html = (
        f'<div class="decision-component-wrapper" style="margin-bottom:12px;">'
        f'  {banner_html}'
        f'  <details id="{trace_id}" class="decision-trace-details"{open_attr}>'
        f'    <summary class="decision-trace-summary">'
        f'      <span>🧭 点击展开查看完整思考决策链路与子问题检索组 ({len(decision_trace)} 步)</span>'
        f'      <span style="font-size:0.75em; color:#94a3b8;">▼</span>'
        f'    </summary>'
        f'    <div class="decision-trace-inner">'
        f'      {"".join(steps_parts)}'
        f'    </div>'
        f'  </details>'
        f'  {script_hook}'
        f'</div>'
    )

    with container.container():
        st.markdown(full_component_html, unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
st.markdown('<div id="page-top"></div>', unsafe_allow_html=True)
toc_groups: List[Dict[str, Any]] = []

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            user_anc = f"nav-user-{msg_idx}"
            st.markdown(f'<span id="{user_anc}" class="toc-target-heading"></span>', unsafe_allow_html=True)
            u_text = str(msg.get("content", "")).strip().replace("\n", " ")
            u_disp = u_text if len(u_text) <= 22 else u_text[:20] + "…"
            toc_groups.append({
                "anchor": user_anc,
                "title": f"💬 {u_disp}",
                "sub_items": []
            })
            st.markdown(msg["content"])
        else:
            if msg.get("decision_trace"):
                dec_box = st.empty()
                _render_decision_component(
                    dec_box,
                    msg["decision_trace"],
                    query_doc_history=msg.get("query_doc_history", []),
                    original_query=msg.get("original_query", ""),
                    is_active=False,
                    force_expand=is_printing,
                )
            anchored_content, msg_toc = extract_and_anchor_headings(msg["content"], msg_idx)
            if toc_groups:
                toc_groups[-1]["sub_items"].extend(msg_toc)
            rendered = linkify_citations(anchored_content, msg.get("sources", []), f"hist-{msg_idx}")
            st.markdown(rendered, unsafe_allow_html=True)

        if msg.get("incomplete"):
            st.caption("⚠️ 回答生成中断，请重新提问。")

        if msg.get("degradation_warnings"):
            for warn in msg["degradation_warnings"]:
                st.warning(warn)

        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(4)
            cols[0].metric("耗时", f"{m['latency_ms']} ms")
            cols[1].metric("补检轮数", m['rewrite_count'])
            cols[2].metric("消耗 Token", f"{m.get('total_tokens', '-')}")
            cols[3].metric("联网回退", "是" if m['used_web'] else "否")

        if msg.get("token_breakdown"):
            _render_token_breakdown(msg["token_breakdown"], msg.get("token_usage"), force_expand=is_printing)

        if msg.get("sources"):
            _render_source_cards(msg["sources"], f"hist-{msg_idx}", force_expand=is_printing)

        if msg.get("retrieval_trace"):
            with st.expander("🔎 实际检索查询与模块配置", expanded=is_printing):
                st.json({"config": msg.get("pipeline_config", {}), "trace": msg["retrieval_trace"]}, expanded=is_printing)

st.markdown('<div id="page-bottom"></div>', unsafe_allow_html=True)
render_floating_toc(toc_groups)

# Register global beforeprint/afterprint auto-expand hooks (covers both button & manual Ctrl+P)
# and trigger window.parent.print() after the entire chat DOM has finished rendering.
_print_trigger_js = "true" if is_printing else "false"
components.html(
    f"""
    <script>
    (function() {{
        try {{
            const pWin = window.parent;
            const pDoc = pWin.document;

            function expandAllForPrint() {{
                if (pDoc.activeElement && typeof pDoc.activeElement.blur === "function") {{
                    pDoc.activeElement.blur();
                }}
                const mainArea = pDoc.querySelector('[data-testid="stMain"]') || pDoc;
                const detailsList = mainArea.querySelectorAll('details');
                if (!pWin.__ragSavedDetails) {{
                    const saved = [];
                    detailsList.forEach(function(el) {{
                        saved.push({{ el: el, wasOpen: el.open }});
                    }});
                    pWin.__ragSavedDetails = saved;
                }}
                detailsList.forEach(function(el) {{
                    el.open = true;
                }});
            }}

            function restoreAfterPrint() {{
                const saved = pWin.__ragSavedDetails;
                if (Array.isArray(saved)) {{
                    saved.forEach(function(item) {{
                        try {{
                            item.el.open = item.wasOpen;
                        }} catch (e) {{}}
                    }});
                    pWin.__ragSavedDetails = null;
                }}
            }}

            if (!pWin.__ragPrintHooked) {{
                pWin.addEventListener("beforeprint", expandAllForPrint);
                pWin.addEventListener("afterprint", restoreAfterPrint);
                pWin.__ragPrintHooked = true;
            }}

            if ({_print_trigger_js}) {{
                setTimeout(function() {{
                    expandAllForPrint();
                    pWin.focus();
                    pWin.print();
                }}, 350);
            }}
        }} catch (err) {{}}
    }})();
    </script>
    """,
    height=0,
    width=0,
)

# ── Background Worker & Chat Lifecycle ─────────────────────────────────────────
curr_sid = st.session_state.get("current_session_id")
active_task = get_active_task(curr_sid) if curr_sid else None

# 1. If a background generation task is active for this session, attach UI and stream updates
if active_task and active_task.status == "running":
    with st.chat_message("assistant"):
        live_dec_box = st.empty()
        live_ans_box = st.empty()

        while active_task.status == "running":
            with active_task.lock:
                cur_steps = list(active_task.live_steps)
                cur_cycles = list(active_task.live_cycles)
                cur_frags = "".join(active_task.fragments)
                cur_status = getattr(active_task, "current_status", "正在执行多模态反思检索与推理...")
            elapsed = int(time.time() - getattr(active_task, "created_at", time.time()))
            status_override = f"{cur_status} (已耗时 {elapsed}s)"
            _render_decision_component(
                live_dec_box,
                cur_steps,
                query_doc_history=cur_cycles,
                original_query=active_task.prompt,
                is_active=True,
                live_status_override=status_override,
            )
            if cur_frags:
                live_ans_box.markdown(cur_frags + " ▌")
            time.sleep(0.2)

        if active_task.status == "done":
            remove_task(curr_sid)
            st.session_state.messages = get_session_messages(curr_sid)
            st.rerun()
        elif active_task.status == "error":
            st.error(f"⚠️ 回答生成异常：{active_task.error}")
            remove_task(curr_sid)
            st.session_state.messages = get_session_messages(curr_sid)
            st.rerun()

# 2. If the last message is a user prompt without assistant response and no active worker, automatically resume
last_msg = st.session_state.messages[-1] if st.session_state.messages else None
if last_msg and last_msg.get("role") == "user" and not active_task:
    hist = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if not m.get("incomplete")
    ]
    start_background_task(curr_sid, last_msg.get("content", ""), chat_config, hist, API_BASE)
    st.rerun()

# 3. Chat Input: triggers persistent background worker decoupled from UI refreshes
chat_disabled = not valid_config or (active_task is not None and active_task.status == "running")
prompt = st.chat_input("Ask a question about your documents …", disabled=chat_disabled)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_session(curr_sid, st.session_state.messages, is_new_message=True)
    hist = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if not m.get("incomplete")
    ]
    start_background_task(curr_sid, prompt, chat_config, hist, API_BASE)
    st.rerun()
