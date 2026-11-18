"""Section -> paragraph -> sentence splitting, with atomic table and image chunks.
Strict rules:
1. Only chunks splitting the same paragraph receive 1-2 sentence sliding window overlap.
   Different paragraphs have NO cross-paragraph overlap (clean paragraph breaks).
2. Headings (e.g. 'ABSTRACT') are prepended as contextual headers to section body text,
   never left as isolated single-word or tiny chunks.
3. Chunks are balanced around 600-1000 characters.
4. Small trailing fragments are merged into previous chunks to avoid tiny dangling fragments.
5. Pure Table of Contents / Outline navigation blocks are filtered out.
"""
from __future__ import annotations

import re


def is_pure_toc_chunk(text: str) -> bool:
    """Detect if text is purely a table of contents or outline index without substantive prose."""
    toc_matches = re.findall(r'(?:[A-Z]\.?\d*(?:\.\d+)*\s+[A-Za-z\s-]{3,30}(?:\.{2,}|\s+)\d{1,3})', text)
    if len(toc_matches) >= 2:
        return True
    num_page_refs = len(re.findall(r'(?:\.{2,}|\s{2,})\s*\d{1,3}(?:\s|$)', text))
    if num_page_refs >= 2 and any(k in text.upper() for k in ("APPENDIX", "CONTENTS", "DETAILS", "OUTLINE")):
        return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences while keeping delimiters."""
    raw = re.split(r'(?<=[。！？!?；;\n])|(?<=\.)\s+(?=[A-Z0-9"\'\(\[\u4e00-\u9fa5])', text)
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def _split_long_paragraph(
    paragraph: str,
    target_size: int = 750,
    max_size: int = 1000,
    min_size: int = 250,
    overlap_target: int = 140,
) -> list[str]:
    """
    Split a single oversized paragraph into chunks with 1-2 sentence overlap.
    Notice: Overlap is ONLY applied within this paragraph.
    """
    sentences = _split_sentences(paragraph)
    if not sentences:
        return [paragraph]
    
    parts: list[str] = []
    i = 0
    n = len(sentences)

    while i < n:
        cur: list[str] = []
        cur_len = 0
        j = i
        while j < n:
            s = sentences[j]
            s_len = len(s)
            if cur_len + s_len + 1 <= max_size:
                cur.append(s)
                cur_len += s_len + 1
                j += 1
                if cur_len >= target_size:
                    rem_len = sum(len(sentences[k]) + 1 for k in range(j, n))
                    if rem_len < min_size and cur_len + rem_len <= max_size + 150:
                        for k in range(j, n):
                            cur.append(sentences[k])
                        j = n
                    break
            else:
                if cur:
                    break
                else:
                    cur.append(s[:max_size])
                    sentences[j] = s[max_size:]
                    j += 1
                    break

        chunk_text = " ".join(cur).strip()
        parts.append(chunk_text)

        if j >= n:
            break

        # Overlap: pick 1-2 trailing sentences from current sub-chunk
        overlap_sentences: list[int] = []
        overlap_len = 0
        k = j - 1
        while k >= i:
            cand = sentences[k]
            if overlap_len + len(cand) <= overlap_target or not overlap_sentences:
                overlap_sentences.insert(0, k)
                overlap_len += len(cand)
                k -= 1
                if overlap_len >= overlap_target:
                    break
            else:
                break
        next_i = overlap_sentences[0] if overlap_sentences else j
        if next_i <= i:
            next_i = i + 1
        i = next_i

    # Merge very small last piece
    if len(parts) > 1 and len(parts[-1]) < min_size:
        last = parts.pop()
        parts[-1] = parts[-1] + " " + last

    return parts


def _body_chunks(
    blocks: list[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 140,
    min_chunk_size: int = 250,
    max_chunk_size: int = 1000,
):
    """
    Split body blocks of a single section into well-balanced chunks:
    - Multiple paragraphs are merged up to max_chunk_size without cross-paragraph overlap.
    - Long single paragraphs (> max_chunk_size) are split with 1-2 sentence overlap within the paragraph.
    - Isolated headings (e.g. 'ABSTRACT') are prepended to subsequent text, never standalone chunks.
    - Small trailing fragments (< min_chunk_size) are merged into previous chunks.
    """
    if not blocks:
        return

    section = blocks[0].get("section", "")
    section_path = blocks[0].get("section_path", [])

    # Filter TOC and empty blocks
    valid_blocks = [b for b in blocks if b.get("text", "").strip() and not is_pure_toc_chunk(b.get("text", ""))]
    if not valid_blocks:
        return

    # Extract heading prefix if first block is a heading (< 120 chars, text_level > 0)
    heading_prefix = ""
    start_idx = 0
    first_b = valid_blocks[0]
    if first_b.get("text_level", 0) > 0 and len(first_b.get("text", "")) < 120:
        heading_prefix = first_b["text"].strip()
        start_idx = 1

    content_blocks = valid_blocks[start_idx:]
    if not content_blocks:
        # Only an isolated heading without body -> skip
        return

    # Extract raw paragraphs with page numbers
    paragraphs: list[tuple[str, int]] = []
    for b in content_blocks:
        page = b.get("page", 1)
        raw_paras = re.split(r"\n\s*\n", b.get("text", ""))
        for p in raw_paras:
            p_str = p.strip()
            if p_str and not is_pure_toc_chunk(p_str):
                paragraphs.append((p_str, page))

    if not paragraphs:
        return

    # Prepend heading to the first paragraph
    if heading_prefix:
        first_p, first_page = paragraphs[0]
        if not first_p.startswith(heading_prefix):
            paragraphs[0] = (f"【{heading_prefix}】\n{first_p}", first_page)

    chunks: list[tuple[str, dict]] = []
    pending_text = ""
    pending_pages: list[int] = []

    def commit_pending():
        nonlocal pending_text, pending_pages
        if pending_text.strip():
            # Drop tiny empty stubs (< 70 chars with no real sentence)
            if len(pending_text.strip()) >= 70 or not pending_text.strip().endswith(":"):
                meta = {
                    "type": "text",
                    "page": min(pending_pages) if pending_pages else 1,
                    "page_end": max(pending_pages) if pending_pages else 1,
                    "section": section,
                    "section_path": section_path,
                }
                chunks.append((pending_text.strip(), meta))
            pending_text = ""
            pending_pages = []

    for p_idx, (p_text, p_page) in enumerate(paragraphs):
        p_len = len(p_text)

        # Case 1: Oversized single paragraph -> needs internal sentence overlap
        if p_len > max_chunk_size:
            commit_pending()  # flush whatever came before (no overlap across paragraph boundary!)
            parts = _split_long_paragraph(
                p_text,
                target_size=chunk_size,
                max_size=max_chunk_size,
                min_size=min_chunk_size,
                overlap_target=chunk_overlap,
            )
            for part in parts:
                meta = {
                    "type": "text",
                    "page": p_page,
                    "page_end": p_page,
                    "section": section,
                    "section_path": section_path,
                }
                chunks.append((part, meta))
            continue

        # Case 2: Paragraph fits within max_chunk_size
        if pending_text:
            combined_len = len(pending_text) + 2 + p_len
            if combined_len <= max_chunk_size:
                pending_text += "\n\n" + p_text
                pending_pages.append(p_page)
            else:
                commit_pending()
                pending_text = p_text
                pending_pages = [p_page]
        else:
            pending_text = p_text
            pending_pages = [p_page]

    # Commit any trailing text
    if pending_text:
        # If pending text is too small (< min_chunk_size) and we already have chunks in this section, merge into last
        if chunks and len(pending_text) < min_chunk_size:
            prev_t, prev_m = chunks[-1]
            if len(prev_t) + 2 + len(pending_text) <= max_chunk_size + 200:
                chunks[-1] = (prev_t + "\n\n" + pending_text, {
                    **prev_m,
                    "page_end": max(prev_m.get("page_end", 1), max(pending_pages) if pending_pages else 1)
                })
                pending_text = ""
        commit_pending()

    for text, meta in chunks:
        yield text, meta


def _lines(value):
    return [value] if isinstance(value, str) else value or []


def _extract_label(captions):
    """Extract figure/table label like ('Figure', '3') from caption strings."""
    if not captions:
        return None
    text = " ".join(captions) if isinstance(captions, list) else str(captions)
    pattern = r"(Figure|Fig\.|Table|Tab\.|图|表)\s*(\d+[a-zA-Z]?)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return None


def _find_body_mention_in_chunks(label_type: str, label_num: str, text_chunks: list):
    """
    Search among already materialized text chunks to find which specific chunk
    references this Figure/Table, its exact chunk_id, and the mention context sentences.
    """
    t_lower = label_type.lower()
    if any(k in t_lower for k in ["tab", "表"]):
        prefix_pattern = r"(?:Table|Tab\.|Tab|表)"
    elif any(k in t_lower for k in ["fig", "图"]):
        prefix_pattern = r"(?:Figure|Fig\.|Fig|图)"
    else:
        prefix_pattern = re.escape(label_type)

    pattern = re.compile(
        rf"{prefix_pattern}\s*{re.escape(label_num)}\b",
        re.IGNORECASE
    )
    for c_text, c_meta in text_chunks:
        if pattern.search(c_text):
            sentences = re.split(r'(?<=[。！？!?；;])|(?<=[.])(?=\s)', c_text)
            matched_sentences = [s.strip() for s in sentences if s.strip() and pattern.search(s)]
            mention_ctx = " ".join(matched_sentences) if matched_sentences else c_text[:400]
            return c_meta.get("chunk_id"), mention_ctx, c_meta.get("section")
    return None, "", None


def chunk_document(
    parsed: dict,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 140,
    min_chunk_size: int = 250,
    max_chunk_size: int = 1000,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    raw_chunks, raw_metas, body = [], [], []

    def flush():
        if body:
            for text, meta in _body_chunks(
                body,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
            ):
                raw_chunks.append(text)
                raw_metas.append({"source": source, **meta})
            body.clear()

    elements = parsed.get("elements", [])
    for element in elements:
        kind = element["type"]
        if kind == "text":
            if element.get("text_level") or (body and body[-1].get("section") != element.get("section")):
                flush()
            body.append(element)
        elif kind == "table":
            flush()
            caption_str = "\n".join(_lines(element.get("caption"))).strip()
            body_str = str(element.get("table_body") or "").strip()
            footnote_str = "\n".join(_lines(element.get("footnote"))).strip()
            parts = [p for p in (caption_str, body_str, footnote_str) if p]
            text = "\n\n".join(parts) or "[表格：未识别到文本]"

            meta_dict = {key: value for key, value in element.items() if key != "table_body"}
            meta_dict["source"] = source
            meta_dict["page_end"] = element.get("page", 1)
            meta_dict["artifact_dir"] = parsed.get("artifact_dir", "")
            meta_dict["section"] = element.get("section", "")
            raw_chunks.append(text)
            raw_metas.append(meta_dict)
        elif kind == "image":
            flush()
            caption_str = "\n".join(_lines(element.get("caption"))).strip()
            footnote_str = "\n".join(_lines(element.get("footnote"))).strip()
            parts = [p for p in (caption_str, footnote_str) if p]
            text = "\n\n".join(parts) or "[图片：未识别到描述]"

            meta_dict = dict(element)
            meta_dict["source"] = source
            meta_dict["page_end"] = element.get("page", 1)
            meta_dict["artifact_dir"] = parsed.get("artifact_dir", "")
            meta_dict["section"] = element.get("section", "")
            raw_chunks.append(text)
            raw_metas.append(meta_dict)
    flush()

    # Assign deterministic global chunk_id for every chunk
    for idx, meta in enumerate(raw_metas):
        meta["chunk_id"] = f"{source}::{idx + 1}"

    # Extract all text chunks for cross-referencing
    text_chunks = [(c, m) for c, m in zip(raw_chunks, raw_metas) if m.get("type") == "text"]

    # Link each Figure / Table to the exact text chunk that mentions it
    for meta in raw_metas:
        if meta.get("type") in ("table", "image"):
            label_info = _extract_label(meta.get("caption"))
            if label_info:
                cid, mention_ctx, sec = _find_body_mention_in_chunks(label_info[0], label_info[1], text_chunks)
                if cid:
                    meta["mention_chunk_id"] = cid
                    meta["mention_context"] = mention_ctx
                    if sec:
                        meta["section"] = sec

    return raw_chunks, raw_metas
