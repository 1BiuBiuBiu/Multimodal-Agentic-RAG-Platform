"""Streamlit PDF chunk browser, separate from the conversation view."""

import hashlib
import math
import base64
import re
from urllib.parse import quote

import httpx
import streamlit as st


TYPE_LABELS = {'text':'正文', 'table':'表格', 'image':'图片'}


def _section_path(item):
    if item.get('section_path'):
        return tuple(item['section_path'])
    return tuple(item['section'].split(' / ')) if item.get('section') else ()


def _matches_section(item, prefix):
    path = _section_path(item)
    if prefix is None:
        return True
    return path[:len(prefix)] == prefix if prefix else not path


def _chunk_id(item, filename):
    return item.get('chunk_id', f"{filename}::{item['id']}")


def _body_outline(chunks, filename):
    outline = {}
    for chunk in chunks:
        if chunk.get('type','text') != 'text':
            continue
        path = _section_path(chunk)
        prefixes = [path[:size] for size in range(1,len(path)+1)] if path else [()]
        for prefix in prefixes:
            entry = outline.setdefault(prefix, {'path':prefix,
                'label':' / '.join(prefix) if prefix else '未标注章节',
                'target':_chunk_id(chunk,filename),'page':chunk['page'],'count':0})
            entry['count'] += 1
    return list(outline.values())


def _navigate_body_section(label, target):
    # Widget callbacks run before the next render, so hidden targets can be
    # made visible without modifying widget state after widget creation.
    st.session_state.chunk_section = label
    st.session_state.chunk_source_page = '全部页码'
    st.session_state.chunk_keyword = ''
    st.session_state.chunk_page = 1
    st.session_state.chunk_navigation_target = target


def _render_body_outline(chunks, filename, section):
    st.subheader('正文章节导航')
    st.caption('点击章节，跳到该章的第一个正文块（含子章节）。跳转时清除页码和关键词限制。')
    outline = _body_outline(chunks,filename)
    if not outline:
        st.info('该文档没有可导航的正文块。')
        return
    with st.container(height=520,border=True):
        for entry in outline:
            path = entry['path']
            title = ('　' * max(0,len(path)-1)) + (path[-1] if path else '未标注章节')
            token = hashlib.sha256(repr((filename,path)).encode()).hexdigest()[:16]
            st.button(f"{title} · {entry['count']} 块", key=f'body_chapter_{token}',
                help=f"第 {entry['page']} 页 · 跳到 {entry['target']}",
                type='primary' if section == entry['label'] else 'secondary',
                use_container_width=True, on_click=_navigate_body_section,
                args=(entry['label'],entry['target']))


def _render_assets(item):
    assets = item.get('asset_images', [])
    for data in assets:
        st.image(base64.b64decode(data.split(',',1)[1]), use_container_width=True)
    return bool(assets)


def _render_element(item, filename, show_table_images, show_raw, focused=False):
    kind = item.get('type', 'text')
    page_label = str(item['page'])
    if item.get('page_end',item['page']) != item['page']:
        page_label += f"-{item['page_end']}"
    title = f"图片 {item['id']}" if kind == 'image' else f"块 {item['id']}"
    if focused:
        title = '📍 ' + title
    with st.expander(f"{title} · 第 {page_label} 页 · {TYPE_LABELS.get(kind,'正文')}", expanded=True):
        chunk_id_str = item.get('chunk_id', filename + '::' + str(item['id']))
        if kind != 'image':
            st.caption(f"块 ID：{chunk_id_str} · {item['char_count']} 字符")
        else:
            st.caption(f"块 ID：{chunk_id_str} · 多模态索引 (Visual-BGE 密集向量 + BM25 稀疏索引)")
        st.caption(f"章节：{item.get('section') or '未标注'}")
        if kind == 'text':
            st.text(item['text'])
            return
        captions = item.get('caption') or []
        if isinstance(captions,str):
            captions = [captions]
        for caption in captions:
            st.text('Caption：' + caption)
        if kind == 'image':
            if not _render_assets(item):
                st.info('该图片的文件暂不可用，以上为提取到的图片信息。')
        else:
            tables = re.findall(r'<table\b[^>]*>.*?</table\s*>', item['text'], flags=re.I|re.S)
            if tables:
                # Streamlit sanitizes HTML; the saved chunk remains unchanged.
                st.html('<div style="overflow:auto;max-height:500px">' + '\n'.join(tables) + '</div>')
                if item.get('native_symbol_rows'):
                    st.text('PDF 原文符号校验：\n' + '\n'.join(item['native_symbol_rows']))
            else:
                st.text(item['text'])
            if show_table_images and not _render_assets(item):
                st.info('该表格的原图暂不可用。')
            if show_raw and tables:
                st.text(item['text'])
        notes = item.get('footnote') or []
        if isinstance(notes,str):
            notes = [notes]
        for note in notes:
            st.text('脚注：' + note)


@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
def _load_chunks(api_base, filename, revision):
    response = httpx.get(
        f"{api_base}/documents/chunks", params={"filename": filename},
        timeout=1900, trust_env=False,
    )
    response.raise_for_status()
    return response.json()


def render_chunk_browser(api_base):
    st.header("📄 PDF 分块浏览")
    st.caption("正文按章节、段落逐级切分；表格与图片单独成块，已建立 Visual-BGE 多模态密集向量与 BM25 稀疏索引。")
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
            doc_map = {item["filename"]: item for item in documents}

            def _format_doc_option(name: str) -> str:
                info = doc_map.get(name, {})
                if "indexed_chunks" not in info:
                    return name
                cnt = info.get("indexed_chunks", 0)
                if not info.get("on_disk", True):
                    return f"{name} (⚠️ 文件缺失 · 残留索引 {cnt} 块)"
                if cnt > 0:
                    return f"{name} (🟢 已索引 · {cnt} 块)"
                return f"{name} (🟠 未建立索引)"

            filename = st.selectbox(
                "选择 PDF",
                [item["filename"] for item in documents],
                format_func=_format_doc_option,
            )
            document = doc_map[filename]
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
                        timeout=1900, trust_env=False,
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
        st.success("正文和表格为当前索引实际使用的块；图片及其 caption 仅供预览。")
    else:
        st.info("以下为分块预览，使用与入库相同的切分规则；尚未加入向量索引。")

    if mode == "已上传的 PDF":
        encoded_name = quote(result["filename"], safe="")
        on_disk = document.get("on_disk", True)
        if not result["indexed"]:
            col_idx, col_del_file = st.columns(2)
            if col_idx.button("⚡ 为该 PDF 建立索引", key="cb_build_single_index", type="primary", use_container_width=True):
                with st.spinner(f"正在为 {result['filename']} 建立向量索引 …"):
                    resp = httpx.post(f"{api_base}/documents/{encoded_name}/index", timeout=1900, trust_env=False)
                if resp.status_code == 200:
                    _load_chunks.clear()
                    st.success(f"已完成索引：{result['filename']}（{resp.json().get('chunks', 0)} 块）")
                    st.rerun()
                else:
                    st.error(f"建立索引失败：{resp.text}")
            if col_del_file.button("🗑️ 删除该 PDF 文件", key="cb_delete_unindexed_file", use_container_width=True):
                resp = httpx.delete(
                    f"{api_base}/documents/{encoded_name}/index",
                    params={"delete_file": "true"},
                    timeout=60,
                    trust_env=False,
                )
                if resp.status_code == 200:
                    _load_chunks.clear()
                    st.rerun()
                else:
                    st.error(f"删除失败：{resp.text}")
        else:
            col_reidx, col_del_idx, col_del_all = st.columns(3)
            if on_disk and col_reidx.button("🔄 重新建立索引", key="cb_rebuild_single_index", use_container_width=True):
                with st.spinner(f"正在重建 {result['filename']} 索引 …"):
                    resp = httpx.post(f"{api_base}/documents/{encoded_name}/index", timeout=1900, trust_env=False)
                if resp.status_code == 200:
                    _load_chunks.clear()
                    st.success(f"已重建索引：{result['filename']}（{resp.json().get('chunks', 0)} 块）")
                    st.rerun()
                else:
                    st.error(f"重建索引失败：{resp.text}")
            if col_del_idx.button("🧹 仅删除索引 (保留文件)", key="cb_delete_single_index", use_container_width=True):
                resp = httpx.delete(
                    f"{api_base}/documents/{encoded_name}/index",
                    params={"delete_file": "false"},
                    timeout=60,
                    trust_env=False,
                )
                if resp.status_code == 200:
                    _load_chunks.clear()
                    st.rerun()
                else:
                    st.error(f"删除索引失败：{resp.text}")
            if col_del_all.button("🗑️ 删除索引及文件", key="cb_delete_single_all", use_container_width=True):
                resp = httpx.delete(
                    f"{api_base}/documents/{encoded_name}/index",
                    params={"delete_file": "true"},
                    timeout=60,
                    trust_env=False,
                )
                if resp.status_code == 200:
                    _load_chunks.clear()
                    st.rerun()
                else:
                    st.error(f"删除失败：{resp.text}")
    stats = st.columns(4)
    stats[0].metric("检索块数", result["total_chunks"])
    stats[1].metric("表格块数", sum(c.get('type')=='table' for c in result['chunks']))
    stats[2].metric("图片数", len(result.get('images',[])))
    stats[3].metric("正文块上限（字符）", result["chunk_size"])
    chunks = result["chunks"]
    items = chunks + result.get('images', [])
    if not items:
        st.warning("未提取到可展示的正文、表格或图片。")
        return

    pages = ["全部页码"] + sorted({number for chunk in items
        for number in range(chunk['page'],chunk.get('page_end',chunk['page'])+1)})
    if st.session_state.get('chunk_source_page','全部页码') not in pages:
        st.session_state.chunk_source_page = '全部页码'
    filters = st.columns(2)
    page = filters[0].selectbox("筛选页码", pages, key='chunk_source_page',
        format_func=lambda value: f"第 {value} 页" if isinstance(value, int) else value)
    keyword = filters[1].text_input("搜索块内容", placeholder="输入关键词",key='chunk_keyword').strip().casefold()
    sections = {'全部章节':None}
    for item in items:
        path = _section_path(item)
        if not path:
            sections.setdefault('未标注章节', ())
        for length in range(1,len(path)+1):
            sections.setdefault(' / '.join(path[:length]),path[:length])
    detail_filters = st.columns(2)
    filter_document = (result['filename'],tuple(sections))
    if st.session_state.get('chunk_section_document') != filter_document:
        st.session_state.pop('chunk_section',None)
        st.session_state.chunk_section_document = filter_document
    section = detail_filters[0].selectbox('选择章节', list(sections), key='chunk_section')
    types = detail_filters[1].multiselect('内容类型', list(TYPE_LABELS.values()),
        default=list(TYPE_LABELS.values()), key='chunk_types')
    st.caption('选择上级章节会同时包含其下所有子章节；章节、类型、页码和关键词可组合筛选。')
    show_table_images = st.checkbox('显示表格原图', value=False, key='chunk_table_images')
    show_raw = st.checkbox('显示表格原始分块', value=False, key='chunk_table_raw')
    prefix = sections[section]
    selected = [chunk for chunk in items
                if (page == "全部页码" or chunk["page"] <= page <= chunk.get('page_end',chunk['page']))
                and TYPE_LABELS.get(chunk.get('type','text'),'正文') in types
                and _matches_section(chunk,prefix)
                and (not keyword or keyword in chunk["text"].casefold())]
    st.caption(f"当前匹配 {len(selected)} / {len(items)} 项 · {result['filename']}")
    # Reset pagination when the document or filters change.
    scope = (result["filename"], page, keyword, section, tuple(types))
    if st.session_state.get("chunk_page_scope") != scope:
        st.session_state.chunk_page_scope = scope
        st.session_state.chunk_page = 1
    if '正文' in types:
        navigation, content = st.columns([1,3],gap='large')
        with navigation:
            _render_body_outline(chunks,result['filename'],section)
    else:
        content = st.container()
    with content:
        if not selected:
            st.info("没有匹配的内容，请调整章节、类型、页码或关键词。")
            if types == ['图片'] and result.get('extraction_available') is False:
                st.info('当前文档尚无图片提取缓存，可通过“本地 PDF 预览”解析后查看。')
            return
        target = st.session_state.pop('chunk_navigation_target',None)
        if target:
            position = next((i for i,item in enumerate(selected)
                if item.get('type','text') == 'text' and _chunk_id(item,result['filename']) == target),None)
            if position is not None:
                st.session_state.chunk_page = position//10+1
                st.session_state.chunk_navigation_focus = (scope,target)
        focus_scope, focus_id = st.session_state.get('chunk_navigation_focus',(None,None))
        focused_id = focus_id if focus_scope == scope else None
        if section != '全部章节':
            st.subheader('当前章节：' + section)
        if focused_id:
            st.caption(f'📍 已定位到该章首个正文块：{focused_id}')
        page_count = math.ceil(len(selected) / 10)
        if st.session_state.get("chunk_page", 1) > page_count:
            st.session_state.chunk_page = 1
        current_page = st.number_input("结果页（每页 10 块）", min_value=1, max_value=page_count, step=1, key="chunk_page")
        offset = (current_page - 1) * 10
        for chunk in selected[offset:offset + 10]:
            focused = chunk.get('type','text') == 'text' and _chunk_id(chunk,result['filename']) == focused_id
            _render_element(chunk,result['filename'],show_table_images,show_raw,focused=focused)
