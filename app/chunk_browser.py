"""Streamlit PDF chunk browser, separate from the conversation view."""

import hashlib
import math

import httpx
import streamlit as st


@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
def _load_chunks(api_base, filename, revision):
    response = httpx.get(
        f"{api_base}/documents/chunks", params={"filename": filename},
        timeout=60, trust_env=False,
    )
    response.raise_for_status()
    return response.json()


def render_chunk_browser(api_base):
    st.header("📄 PDF 分块浏览")
    st.caption("查看每个文本块的完整内容、来源页码与字符数。")
    mode = st.radio("文档来源", ["已上传的 PDF", "本地 PDF 预览"], horizontal=True)
    result = None
    try:
        if mode == "已上传的 PDF":
            if st.button("刷新文档列表"):
                _load_chunks.clear()
            response = httpx.get(f"{api_base}/documents", timeout=10, trust_env=False)
            response.raise_for_status()
            documents = response.json()["documents"]
            if not documents:
                st.info("暂无 PDF。可以在侧栏上传并索引，或切换到“本地 PDF 预览”。")
                return
            filename = st.selectbox("选择 PDF", [item["filename"] for item in documents])
            document = next(item for item in documents if item["filename"] == filename)
            with st.spinner("正在读取分块…"):
                result = _load_chunks(api_base, filename, document["revision"])
        else:
            uploaded = st.file_uploader("选择要预览的 PDF", type="pdf", key="chunk_preview_file")
            if uploaded is None:
                st.info("选择任意 PDF 后点击“生成分块预览”，无需建立向量索引。")
                return
            fingerprint = (uploaded.name, hashlib.sha256(uploaded.getvalue()).hexdigest())
            if st.button("生成分块预览", type="primary"):
                with st.spinner("正在解析和切分 PDF…"):
                    response = httpx.post(
                        f"{api_base}/documents/preview",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=120, trust_env=False,
                    )
                    response.raise_for_status()
                    st.session_state.chunk_preview = (fingerprint, response.json())
            cached = st.session_state.get("chunk_preview")
            if cached and cached[0] == fingerprint:
                result = cached[1]
            else:
                return
    except httpx.HTTPError as exc:
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                detail = exc.response.json().get("detail", "读取失败")
            except ValueError:
                detail = "读取失败"
            st.error(f"{detail}（HTTP {exc.response.status_code}）")
        else:
            st.error("无法读取分块，请确认后端已启动并稍后重试。")
        return

    if result["indexed"]:
        st.success("以下内容为当前向量索引实际使用的文本块。")
    else:
        st.info("以下为分块预览，使用与入库相同的切分规则；尚未加入向量索引。")
    stats = st.columns(3)
    stats[0].metric("总块数", result["total_chunks"])
    stats[1].metric("块大小上限（字符）", result["chunk_size"])
    stats[2].metric("重叠长度（字符）", result["chunk_overlap"])
    chunks = result["chunks"]
    if not chunks:
        st.warning("没有可展示的文本块。PDF 可能是扫描件，或提取的文字过短；当前流程未配置 OCR。")
        return

    pages = ["全部页码"] + sorted({chunk["page"] for chunk in chunks})
    filters = st.columns(2)
    page = filters[0].selectbox("筛选页码", pages, format_func=lambda value: f"第 {value} 页" if isinstance(value, int) else value)
    keyword = filters[1].text_input("搜索块内容", placeholder="输入关键词").strip().casefold()
    selected = [chunk for chunk in chunks
                if (page == "全部页码" or chunk["page"] == page)
                and (not keyword or keyword in chunk["text"].casefold())]
    st.caption(f"当前匹配 {len(selected)} / {len(chunks)} 个块 · {result['filename']}")
    if not selected:
        st.info("没有匹配的文本块，请调整页码或关键词。")
        return
    page_count = math.ceil(len(selected) / 10)
    # Reset pagination when the document or filters change.
    scope = (result["filename"], page, keyword)
    if st.session_state.get("chunk_page_scope") != scope:
        st.session_state.chunk_page_scope = scope
        st.session_state.chunk_page = 1
    if st.session_state.get("chunk_page", 1) > page_count:
        st.session_state.chunk_page = 1
    current_page = st.number_input("结果页（每页 10 块）", min_value=1, max_value=page_count, step=1, key="chunk_page")
    offset = (current_page - 1) * 10
    for chunk in selected[offset:offset + 10]:
        with st.expander(f"块 {chunk['id']} · 第 {chunk['page']} 页 · {chunk['char_count']} 字符", expanded=True):
            st.text(chunk["text"])
