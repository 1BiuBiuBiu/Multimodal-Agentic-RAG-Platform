"""
app/export_utils.py
───────────────────
Utilities to export chat sessions to:
1. Clean Markdown (.md)
2. Standalone offline HTML report (.html) with KaTeX math rendering and image embedding
3. Print-to-PDF stylesheet support
"""

from __future__ import annotations
import html
from datetime import datetime
from typing import List, Dict, Any


def _format_page_range(source: Dict[str, Any]) -> str:
    page = str(source.get("page", 1))
    page_end = source.get("page_end")
    if page_end and str(page_end) != page:
        return f"{page}-{page_end}"
    return page


def export_to_markdown(messages: List[Dict[str, Any]], session_title: str = "对话记录") -> str:
    """Generate clean, publication-grade Markdown from chat history."""
    lines = [
        f"# {session_title}",
        f"\n*导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n",
        "---\n",
    ]

    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if role == "user":
            lines.append(f"## 👤 用户提问\n\n{content}\n")
        else:
            lines.append("## 🤖 助手回答\n")

            # Decision trace & sub-query cycles
            decision_trace = msg.get("decision_trace") or []
            query_doc_history = msg.get("query_doc_history") or []
            if decision_trace:
                lines.append(f"### 🧭 完整思考决策链路与子问题检索组 ({len(decision_trace)} 步)\n")
                orig_q = msg.get("original_query", "")
                if orig_q:
                    lines.append(f"- **原始用户问题**：`{orig_q}`")
                for s_i, step in enumerate(decision_trace, 1):
                    s_name = step.get("step", f"Step {s_i}")
                    s_det = str(step.get("detail", "")).strip()
                    lines.append(f"{s_i}. **{s_name}**")
                    if s_det:
                        lines.append(f"   > {s_det.replace(chr(10), chr(10) + '   > ')}")
                if query_doc_history:
                    lines.append("\n#### 📑 分轮次子问题与专属检索文档组")
                    for c_i, c_data in enumerate(query_doc_history, 1):
                        c_name = c_data.get("cycle_name", f"第 {c_data.get('cycle', c_i)} 轮检索")
                        in_q = c_data.get("input_query", "")
                        lines.append(f"- **{c_name}** (输入: `{in_q}`)")
                        for g_i, grp in enumerate(c_data.get("sub_groups") or [], 1):
                            sub_q = grp.get("sub_query", "")
                            docs = grp.get("docs") or []
                            lines.append(f"  - **子问题 {g_i}**：`{sub_q}` (召回 {len(docs)} 篇证据)")
                            for d_i, doc in enumerate(docs[:3], 1):
                                d_src = doc.get("source", "doc")
                                d_page = doc.get("page", 1)
                                d_cid = doc.get("chunk_id", "")
                                d_snip = str(doc.get("text", "")).strip().replace("\n", " ")[:120]
                                lines.append(f"    - 文档 [{d_i}] `{d_cid}` ({d_src} p.{d_page}): {d_snip}")
                lines.append("")

            lines.append(f"{content}\n")

            # Meta performance stats
            meta = msg.get("meta") or {}
            if meta:
                lines.append(
                    f"\n> **⏱️ 耗时**: {meta.get('latency_ms', 0)} ms | "
                    f"**补检轮数**: {meta.get('rewrite_count', 0)} | "
                    f"**Token 消耗**: {meta.get('total_tokens', '-')} | "
                    f"**联网回退**: {'是' if meta.get('used_web') else '否'}\n"
                )

            # Token breakdown
            token_breakdown = msg.get("token_breakdown") or []
            if token_breakdown:
                lines.append("\n### 🪙 详细 Token 分步消耗明细\n")
                lines.append("| 步骤环节 | 输入 Prompt Tokens | 输出 Completion Tokens | 阶段总耗 Total Tokens | 消耗占比 |")
                lines.append("| :--- | :---: | :---: | :---: | :---: |")
                total_p = sum(item.get("prompt_tokens", 0) for item in token_breakdown)
                total_c = sum(item.get("completion_tokens", 0) for item in token_breakdown)
                grand_total = sum(item.get("total_tokens", 0) for item in token_breakdown)
                for item in token_breakdown:
                    step_name = item.get("step", "未知步骤")
                    p_tok = item.get("prompt_tokens", 0)
                    c_tok = item.get("completion_tokens", 0)
                    tot = item.get("total_tokens", p_tok + c_tok)
                    pct = f"{(tot / grand_total * 100):.1f}%" if grand_total > 0 else "-"
                    lines.append(f"| **{step_name}** | {p_tok:,} | {c_tok:,} | **{tot:,}** | `{pct}` |")
                lines.append(f"| **📊 全链路累计总计** | **{total_p:,}** | **{total_c:,}** | **{grand_total:,}** | `100.0%` |\n")

            # Sources (complete, no 200-char truncation)
            sources = msg.get("sources") or []
            if sources:
                lines.append(f"\n### 📚 引用证据来源（共 {len(sources)} 处）")
                for s_idx, s in enumerate(sources):
                    s_src = s.get("source", "doc")
                    s_page = _format_page_range(s)
                    s_cid = s.get("chunk_id", "")
                    s_sec = s.get("section", "未标注章节")
                    s_text = str(s.get("text", "")).strip()
                    lines.append(f"- **[{s_idx+1}] {s_src} (Page {s_page}, Section: {s_sec}, #{s_cid})**")
                    if s.get("mention_context"):
                        lines.append(f"  - **正文引用解读**：{s['mention_context']}")
                    if s.get("caption"):
                        caps = s["caption"]
                        cap_str = " ".join(caps) if isinstance(caps, list) else str(caps)
                        lines.append(f"  - **图注 Caption**：{cap_str}")
                    if s_text:
                        quoted = s_text.replace("\n", "\n  > ")
                        lines.append(f"  > {quoted}\n")

        lines.append("\n---\n")

    return "\n".join(lines)


def export_to_html(messages: List[Dict[str, Any]], session_title: str = "对话记录") -> str:
    """Generate self-contained, offline-readable HTML document."""
    escaped_title = html.escape(session_title)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    body_parts = []
    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = html.escape(msg.get("content", "")).replace("\n", "<br/>")

        if role == "user":
            card = (
                f'<div class="msg-card msg-user">'
                f'  <div class="msg-header">👤 <b>用户</b></div>'
                f'  <div class="msg-body">{content}</div>'
                f'</div>'
            )
        else:
            # Decision trace & sub-query cycles
            decision_trace = msg.get("decision_trace") or []
            query_doc_history = msg.get("query_doc_history") or []
            trace_html = ""
            if decision_trace:
                t_parts = []
                orig_q = msg.get("original_query", "")
                if orig_q:
                    t_parts.append(f'<div class="trace-orig">🎯 <b>原始用户问题：</b><code>{html.escape(orig_q)}</code></div>')
                for s_i, step in enumerate(decision_trace, 1):
                    s_name = html.escape(str(step.get("step", f"Step {s_i}")))
                    s_det = html.escape(str(step.get("detail", ""))).replace("\n", "<br/>")
                    t_parts.append(
                        f'<div class="trace-step">'
                        f'  <div class="trace-step-title">{s_i}. {s_name}</div>'
                        f'  <div class="trace-step-desc">{s_det}</div>'
                        f'</div>'
                    )
                if query_doc_history:
                    t_parts.append('<div class="trace-cycles-title">📑 分轮次子问题与专属检索文档组</div>')
                    for c_i, c_data in enumerate(query_doc_history, 1):
                        c_name = html.escape(str(c_data.get("cycle_name", f"第 {c_data.get('cycle', c_i)} 轮检索")))
                        in_q = html.escape(str(c_data.get("input_query", "")))
                        t_parts.append(f'<div class="trace-cycle"><b>{c_name}</b> (输入: <code>{in_q}</code>)</div>')
                        for g_i, grp in enumerate(c_data.get("sub_groups") or [], 1):
                            sub_q = html.escape(str(grp.get("sub_query", "")))
                            docs = grp.get("docs") or []
                            t_parts.append(
                                f'<div class="trace-subq"><b>子问题 {g_i}：</b><code>{sub_q}</code> (召回 {len(docs)} 篇证据)</div>'
                            )
                            for d_i, doc in enumerate(docs[:3], 1):
                                d_src = html.escape(str(doc.get("source", "doc")))
                                d_page = doc.get("page", 1)
                                d_cid = html.escape(str(doc.get("chunk_id", "")))
                                d_snip = html.escape(str(doc.get("text", ""))[:140]).replace("\n", " ")
                                t_parts.append(
                                    f'<div class="trace-doc">📄 文档 [{d_i}] <code>{d_cid}</code> ({d_src} p.{d_page}): {d_snip}</div>'
                                )
                trace_html = (
                    f'<details open class="trace-box">'
                    f'  <summary class="trace-summary">🧭 完整思考决策链路与子问题检索组 ({len(decision_trace)} 步)</summary>'
                    f'  <div class="trace-body">{"".join(t_parts)}</div>'
                    f'</details>'
                )

            meta = msg.get("meta") or {}
            meta_html = ""
            if meta:
                meta_html = (
                    f'<div class="meta-bar">'
                    f'  ⏱️ 耗时: {meta.get("latency_ms", 0)} ms &nbsp;|&nbsp; '
                    f'  补检轮数: {meta.get("rewrite_count", 0)} &nbsp;|&nbsp; '
                    f'  总 Token: {meta.get("total_tokens", "-")}'
                    f'</div>'
                )

            # Token breakdown table
            token_breakdown = msg.get("token_breakdown") or []
            tok_html = ""
            if token_breakdown:
                total_p = sum(item.get("prompt_tokens", 0) for item in token_breakdown)
                total_c = sum(item.get("completion_tokens", 0) for item in token_breakdown)
                grand_total = sum(item.get("total_tokens", 0) for item in token_breakdown)
                rows = []
                for item in token_breakdown:
                    step_name = html.escape(str(item.get("step", "未知步骤")))
                    p_tok = item.get("prompt_tokens", 0)
                    c_tok = item.get("completion_tokens", 0)
                    tot = item.get("total_tokens", p_tok + c_tok)
                    pct = f"{(tot / grand_total * 100):.1f}%" if grand_total > 0 else "-"
                    rows.append(
                        f'<tr><td><b>{step_name}</b></td><td>{p_tok:,}</td><td>{c_tok:,}</td><td><b>{tot:,}</b></td><td><code>{pct}</code></td></tr>'
                    )
                rows.append(
                    f'<tr class="tok-total"><td><b>📊 全链路累计总计</b></td><td><b>{total_p:,}</b></td><td><b>{total_c:,}</b></td><td><b>{grand_total:,}</b></td><td><code>100.0%</code></td></tr>'
                )
                tok_html = (
                    f'<div class="token-box">'
                    f'  <div class="sources-title">🪙 详细 Token 分步消耗明细</div>'
                    f'  <table class="token-table">'
                    f'    <thead><tr><th>步骤环节</th><th>Prompt Tokens</th><th>Completion Tokens</th><th>Total Tokens</th><th>占比</th></tr></thead>'
                    f'    <tbody>{"".join(rows)}</tbody>'
                    f'  </table>'
                    f'</div>'
                )

            sources = msg.get("sources") or []
            src_html = ""
            if sources:
                src_items = []
                for s_idx, s in enumerate(sources):
                    s_src = html.escape(str(s.get("source", "doc")))
                    s_page = html.escape(_format_page_range(s))
                    s_cid = html.escape(str(s.get("chunk_id", "")))
                    s_sec = html.escape(str(s.get("section", "未标注章节")))
                    s_text = html.escape(str(s.get("text", "")).strip()).replace("\n", "<br/>")
                    extra_parts = []
                    if s.get("mention_context"):
                        extra_parts.append(f'<div class="src-extra">📖 <b>正文引用解读：</b>{html.escape(str(s["mention_context"]))}</div>')
                    if s.get("caption"):
                        caps = s["caption"]
                        cap_str = " ".join(caps) if isinstance(caps, list) else str(caps)
                        extra_parts.append(f'<div class="src-extra">🖼️ <b>图注 Caption：</b>{html.escape(cap_str)}</div>')
                    src_items.append(
                        f'<li class="src-item">'
                        f'  <b>[{s_idx+1}] {s_src}</b> (第 {s_page} 页, 章节: {s_sec}, 块: <code>{s_cid}</code>)'
                        f'  {"".join(extra_parts)}'
                        f'  <div class="src-snip">{s_text}</div>'
                        f'</li>'
                    )
                src_html = f'<div class="sources-box"><div class="sources-title">📚 引用证据来源 ({len(sources)} 处)：</div><ul>{"".join(src_items)}</ul></div>'

            card = (
                f'<div class="msg-card msg-assistant">'
                f'  <div class="msg-header">🤖 <b>Agentic RAG Assistant</b></div>'
                f'  {trace_html}'
                f'  <div class="msg-body">{content}</div>'
                f'  {meta_html}'
                f'  {tok_html}'
                f'  {src_html}'
                f'</div>'
            )
        body_parts.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>{escaped_title} - 会话导出</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: #1e293b;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    background: #f8fafc;
}}
.page-title {{
    font-size: 1.8em;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
}}
.export-meta {{
    font-size: 0.88em;
    color: #64748b;
    margin-bottom: 30px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 12px;
}}
.msg-card {{
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
.msg-user {{
    border-left: 5px solid #2563eb;
    background: #f0f7ff;
}}
.msg-assistant {{
    border-left: 5px solid #10b981;
}}
.msg-header {{
    font-size: 1.02em;
    margin-bottom: 10px;
    color: #1e293b;
}}
.msg-body {{
    font-size: 0.98em;
    line-height: 1.7;
    white-space: normal;
}}
.trace-box {{
    margin-bottom: 14px;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #f8fafc;
}}
.trace-summary {{
    padding: 8px 12px;
    font-weight: 700;
    font-size: 0.9em;
    color: #334155;
    cursor: pointer;
}}
.trace-body {{
    padding: 10px 14px;
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
    font-size: 0.88em;
}}
.trace-orig {{
    margin-bottom: 8px;
}}
.trace-step {{
    margin-bottom: 8px;
    break-inside: avoid;
}}
.trace-step-title {{
    font-weight: 700;
    color: #1e293b;
}}
.trace-step-desc {{
    margin-left: 8px;
    padding: 4px 10px;
    border-left: 3px solid #60a5fa;
    background: #f8fafc;
    color: #475569;
}}
.trace-cycles-title {{
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid #e2e8f0;
    font-weight: 700;
}}
.trace-cycle {{
    margin-top: 6px;
}}
.trace-subq {{
    margin-left: 10px;
    margin-top: 4px;
    padding: 4px 8px;
    background: #f1f5f9;
    border-radius: 4px;
}}
.trace-doc {{
    margin-left: 22px;
    margin-top: 2px;
    color: #475569;
    font-size: 0.92em;
}}
.meta-bar {{
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #f1f5f9;
    font-size: 0.84em;
    color: #64748b;
}}
.token-box {{
    margin-top: 14px;
    padding: 12px 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-size: 0.86em;
}}
.token-table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 6px;
}}
.token-table th, .token-table td {{
    border: 1px solid #e2e8f0;
    padding: 6px 8px;
    text-align: center;
}}
.token-table th:first-child, .token-table td:first-child {{
    text-align: left;
}}
.token-table thead {{
    background: #f1f5f9;
}}
.sources-box {{
    margin-top: 14px;
    padding: 12px 14px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    font-size: 0.88em;
}}
.sources-title {{
    font-weight: 700;
    color: #334155;
    margin-bottom: 8px;
}}
.sources-box ul {{
    margin: 0;
    padding-left: 20px;
}}
.sources-box li {{
    margin-bottom: 10px;
    break-inside: avoid;
}}
.src-extra {{
    margin-top: 4px;
    color: #334155;
}}
.src-snip {{
    margin-top: 4px;
    color: #475569;
    font-size: 0.92em;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 6px 8px;
}}
code {{
    background: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.9em;
}}
@media print {{
    body {{
        background: #ffffff;
        padding: 0;
        margin: 0;
        max-width: 100%;
    }}
    .msg-card {{
        box-shadow: none;
        border: 1px solid #cbd5e1;
        break-inside: auto;
        page-break-inside: auto;
    }}
}}
</style>
</head>
<body>
<div class="page-title">📄 {escaped_title}</div>
<div class="export-meta">导出时间: {now_str} · Enterprise Agentic RAG Platform</div>
{"".join(body_parts)}
</body>
</html>"""
    return html_content
