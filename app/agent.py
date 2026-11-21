"""
app/agent.py
────────────
Corrective RAG Agent built with LangGraph 0.4+ with:
- Intelligent Routing (documents vs. llm_only)
- Sub-question decomposition & hybrid retrieval
- Two-stage Relevance & Sufficiency Grading with Information Gap Detection
- Gap-driven targeted follow-up retrieval & context merging
- Multimodal VLM generation with citations
- Full-link Transparent Degradation Sentinel (explicit warnings on fallback)
"""

from __future__ import annotations
import base64
import mimetypes
import os
import re
from typing import Any, Dict, List, Annotated, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer
from loguru import logger

from app.llm import create_llm
from app.pipeline_config import PipelineConfig
from embeddings.visual_embedder import resolve_image_path
from vectorstore.store import load_snapshot

load_dotenv()

# ── LLM (DeepSeek API) ─────────────────────────────────────────────────────
llm = create_llm()

# ── Web search fallback (free Tavily key at tavily.com) ──────────────────────
web_search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
)

MAX_REWRITES = 2   # prevent infinite loops


# ── Agent state ──────────────────────────────────────────────────────────────

class _AgentStateRequired(TypedDict):
    messages:      Annotated[List[BaseMessage], add_messages]   # full conversation history
    query:         str
    rewrite_count: int
    context:       List[Dict[str, Any]]   # retrieved chunk dicts from hybrid_search
    web_results:   List[str]
    answer:        str

class AgentState(_AgentStateRequired, total=False):
    """LangGraph state — dynamic switches and trace info."""
    _route: str   # 'documents' | 'llm_only'
    pipeline_config: dict
    original_query: str
    retrieval_snapshot: Any
    translation_cache: Any
    retrieval_trace: list
    _grade: str   # 'generate' | 'rewrite' | 'web_search'
    information_gap: str  # missing facts or tables identified during sufficiency evaluation
    degradation_warnings: List[str]  # transparent alerts for fallbacks
    accepted_chunk_ids: List[str]  # evaluation cache: chunk IDs confirmed relevant
    rejected_chunk_ids: List[str]  # evaluation cache: chunk IDs confirmed irrelevant
    decision_trace: List[Dict[str, Any]]  # persistent end-to-end decision & reflection events
    token_usage: Dict[str, int]  # prompt, completion, total token consumption
    token_breakdown: List[Dict[str, Any]]  # step-by-step breakdown of token consumption
    new_candidate_count: int  # count of brand-new chunks retrieved in this cycle
    query_doc_history: List[Dict[str, Any]]  # hierarchical cycle -> sub_queries -> dedicated doc groups
    targeted_sub_queries: List[str]  # directly formulated by Stage 2 sufficiency check


# ── Node helpers ─────────────────────────────────────────────────────────────

def _last_human_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return state.get("query") or ""  # type: ignore[union-attr]


def _options(state: AgentState) -> PipelineConfig:
    return PipelineConfig.model_validate(state.get("pipeline_config", {}))


def retrieve_documents(*args, **kwargs):
    from app.retrieval import retrieve_documents as retrieve_impl
    return retrieve_impl(*args, **kwargs)


def _record_warning(state: AgentState, warning_msg: str) -> List[str]:
    warnings = list(state.get("degradation_warnings") or [])
    if warning_msg and warning_msg not in warnings:
        warnings.append(warning_msg)
        logger.warning(f"[Degradation Sentinel] {warning_msg}")
    return warnings


def _track_tokens(
    state: AgentState,
    response: Any = None,
    prompt_text: str = "",
    completion_text: str = "",
    usage_override: Optional[Dict[str, int]] = None,
    step_name: str = "",
) -> tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Accurately accumulate token consumption from LLM calls (API metadata + fallback estimation) and record step breakdown."""
    usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    breakdown = list(state.get("token_breakdown") or [])
    p_tok = 0
    c_tok = 0
    if usage_override:
        p_tok = int(usage_override.get("prompt_tokens", 0) or 0)
        c_tok = int(usage_override.get("completion_tokens", 0) or 0)
    elif response is not None:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = response.usage_metadata.get("input_tokens") or response.usage_metadata.get("prompt_tokens") or 0
            c_tok = response.usage_metadata.get("output_tokens") or response.usage_metadata.get("completion_tokens") or 0
        elif hasattr(response, "response_metadata") and response.response_metadata:
            meta_usage = response.response_metadata.get("token_usage") or {}
            p_tok = meta_usage.get("prompt_tokens", 0)
            c_tok = meta_usage.get("completion_tokens", 0)

    if not usage_override:
        if p_tok == 0 and prompt_text:
            p_tok = max(1, len(prompt_text) // 4)
        if c_tok == 0:
            if completion_text:
                c_tok = max(1, len(completion_text) // 4)
            elif response is not None and hasattr(response, "content") and response.content:
                c_tok = max(1, len(str(response.content)) // 4)

    usage["prompt_tokens"] = usage.get("prompt_tokens", 0) + p_tok
    usage["completion_tokens"] = usage.get("completion_tokens", 0) + c_tok
    usage["total_tokens"] = usage.get("total_tokens", 0) + p_tok + c_tok

    if step_name and (p_tok > 0 or c_tok > 0):
        breakdown.append({
            "step": step_name,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": p_tok + c_tok,
        })

    return usage, breakdown


def _append_decision(state: AgentState, step: str, detail: str, data: Any = None) -> List[Dict[str, Any]]:
    trace = list(state.get("decision_trace") or [])
    entry = {
        "step": step,
        "detail": detail,
        "data": data,
    }
    trace.append(entry)
    try:
        writer = get_stream_writer()
        if writer:
            writer({
                "type": "decision_step",
                "step": step,
                "detail": detail,
                "data": data,
                "history": trace,
            })
    except Exception:
        pass
    return trace


def _build_cognitive_memory_tree(state: AgentState, is_eval: bool = False) -> str:
    """
    Construct a unified, hierarchical Cognitive Working Memory Tree.
    Organizes multi-turn retrieval evidence by explicit exploration cycles, sub-questions, and dedicated document groups.
    Features:
    - Smart Cross-Reference Pointer: Repeated chunks across sub-questions are rendered in full on first appearance,
      and rendered as lightweight cross-reference pointers subsequently (0% token redundancy, 100% semantic preservation).
    """
    original_query = state.get("original_query", state.get("query", ""))
    history_cycles = list(state.get("query_doc_history") or [])
    
    sections = [
        f"【用户原始核心目标 (Original User Goal)】：\n{original_query}\n",
        "【系统多轮探索与证据推理演进树 (Cognitive Evidence Tree)】："
    ]

    seen_chunk_ids = set()
    first_location = {}  # cid -> location description (e.g. "子问题 1")

    def _format_chunk_visual_attachments(c_dict: Dict[str, Any]) -> str:
        parts = []
        for v in c_dict.get("mentioned_visuals", []):
            v_kind = "表格 (Table)" if v.get("type") == "table" else "图片/架构图 (Figure)"
            v_cap = " ".join(v.get("caption", [])) if isinstance(v.get("caption"), list) else (v.get("caption") or "")
            v_text = v.get("text", "")
            v_src = v.get("source") or c_dict.get("source", "")
            v_cid = v.get("chunk_id")
            tag_parts = [p for p in (v_src, v_cid if str(v_cid).startswith("#") else (f"#{v_cid}" if v_cid else "")) if p]
            tag_str = f" [{', '.join(tag_parts)}]" if tag_parts else ""
            part = f"\n    【正文引用关联{v_kind}附件{tag_str}】：{v_cap}"
            if v_text:
                part += f"\n    {v_text[:350]}"
            parts.append(part)
        return "".join(parts)

    def _chunk_tag(item: Dict[str, Any], fallback_idx: Optional[int] = None) -> str:
        src = item.get("source", "doc")
        raw_cid = item.get("chunk_id")
        if raw_cid is None:
            raw_cid = f"{fallback_idx}" if fallback_idx is not None else "N/A"
        cid_s = str(raw_cid)
        cid_tok = cid_s if cid_s.startswith("#") else f"#{cid_s}"
        return f"[{src}, {cid_tok}]"

    if not history_cycles:
        context = state.get("context") or []
        sections.append("\n▶ 【当前最新探索轮次】首轮检索证据组：")
        for idx, c in enumerate(context):
            kind = c.get("type", "text")
            snip = c.get("text", "")
            body = snip
            if kind in ("table", "image") and c.get("mention_context"):
                body += f"\n    【正文引用与深度解读】：\n    {c['mention_context']}"
            body += _format_chunk_visual_attachments(c)
            sections.append(f"  • {_chunk_tag(c, idx + 1)}:\n{body}")
        return "\n".join(sections)

    for c_idx, cycle in enumerate(history_cycles):
        c_num = cycle.get("cycle", c_idx + 1)
        c_name = cycle.get("cycle_name", f"第 {c_num} 轮检索")
        input_q = cycle.get("input_query", "")
        c_gap = cycle.get("gap", "")
        sub_groups = cycle.get("sub_groups", [])
        
        is_latest = (c_idx == len(history_cycles) - 1)
        cycle_header = f"▶ 【当前最新探索轮次】{c_name}" if is_latest else f"▶ 【历史探索轮次】{c_name}"
        sections.append(f"\n{cycle_header} (输入查询: `{input_q}`)")
        if c_gap and c_num > 1:
            sections.append(f"  └─ 🎯 本轮针对解决的历史信息缺口 (Targeted GAP): \"{c_gap}\"")

        if not sub_groups:
            docs = cycle.get("new_docs", [])
            sections.append("  └─ 专属证据组:")
            for d in docs:
                cid = d.get("chunk_id", "N/A")
                snip = d.get("text", "")
                if cid != "N/A" and cid in seen_chunk_ids:
                    ref_loc = first_location.get(cid, "此前轮次")
                    sections.append(f"     • {_chunk_tag(d)} (← 亦高度支撑本轮次，详见 {ref_loc} 证据)")
                    continue
                if cid != "N/A":
                    seen_chunk_ids.add(cid)
                    first_location[cid] = f"第 {c_num} 轮检索"
                body = snip + _format_chunk_visual_attachments(d)
                sections.append(f"     • {_chunk_tag(d)}:\n{body}")
        else:
            for g_idx, grp in enumerate(sub_groups):
                sub_q = grp.get("sub_query", "")
                docs = grp.get("docs", [])
                loc_desc = f"第 {c_num} 轮子问题 {g_idx + 1}" if len(history_cycles) > 1 else f"子问题 {g_idx + 1}"
                sections.append(f"  ├─ 探究子问题 {g_idx + 1}: \"{sub_q}\"")
                if not docs:
                    sections.append("  │   └─ (本子问题未召回高相关证据)")
                else:
                    sections.append(f"  │   └─ 专属 Cross-Encoder 精排证据组 (共 {len(docs)} 篇):")
                    for d in docs:
                        cid = d.get("chunk_id", "N/A")
                        kind = d.get("type", "text")
                        label = {"text": "正文", "table": "Table / 表格", "image": "Figure / 图片"}.get(kind, kind)
                        
                        # Cross-reference deduplication pointer
                        if cid != "N/A" and cid in seen_chunk_ids:
                            first_ref = first_location.get(cid, "此前子问题")
                            if is_eval:
                                sections.append(f"         • {_chunk_tag(d)} (← 亦高度支撑本子问题，内容见 {first_ref})")
                            else:
                                sections.append(f"         • {_chunk_tag(d)} (← 亦高度支撑本子问题，详见上方 {first_ref} 完整证据)")
                            continue

                        # First encounter: record and render in full
                        if cid != "N/A":
                            seen_chunk_ids.add(cid)
                            first_location[cid] = loc_desc

                        snip = d.get("text", "")
                        body = snip
                        if kind in ("table", "image") and d.get("mention_context"):
                            body += f"\n           【该{label}在正文中的完整解读】：\n           {d['mention_context']}"
                        body += _format_chunk_visual_attachments(d)
                        sections.append(f"         • {_chunk_tag(d)}:\n{body}")

        if c_idx < len(history_cycles) - 1:
            prev_gap = state.get("information_gap", "")
            if prev_gap:
                sections.append(f"  └─ ⚠️ 上轮反思识别出的核心信息缺口 (Identified GAP): \"{prev_gap}\"")

    return "\n".join(sections)


# ── Nodes ────────────────────────────────────────────────────────────────────

def route_query(state: AgentState) -> AgentState:
    """
    Ask LLM whether the question needs document retrieval.
    Returns updated state; routing decision is in 'query' metadata.
    """
    query = _last_human_query(state)
    options = _options(state)
    prompt = (
        "You are a routing assistant. Given the user question below, decide:\n"
        "- Reply 'documents' if the answer likely requires looking up specific documents.\n"
        "- Reply 'llm_only' if it's general knowledge you can answer without documents.\n\n"
        f"Question: {query}\n\nReply with exactly one word: documents OR llm_only"
    )
    if not options.route_query:
        trace = _append_decision(state, "🟢 意图路由", "问题路由开关已关闭，默认执行知识库深度检索。", {"route": "documents"})
        return {**state, "query": query, "_route": "documents", "decision_trace": trace}

    response = llm.invoke([HumanMessage(content=prompt)])
    token_usage, token_breakdown = _track_tokens(state, response, prompt, step_name="🧭 意图路由")
    decision = response.content.strip().lower()
    if "documents" not in decision:
        decision = "llm_only"
    logger.info(f"[route_query] decision='{decision}' for query='{query[:60]}'")
    
    route_label = "知识库多模态检索 (documents)" if decision == "documents" else "通用大模型直接回答 (llm_only)"
    trace = _append_decision(state, "🟢 意图路由", f"分析用户意图，决策路由至：{route_label}", {"route": decision})
    return {
        **state,
        "query": query,
        "_route": decision,
        "decision_trace": trace,
        "token_usage": token_usage,
        "token_breakdown": token_breakdown,
    }


def _deduplicate_mention_chunks(context: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Deduplicate identical chunk_ids while preserving all cited body text chunks alongside
    referenced Table/Figure chunks so downstream generation and RAGAs evaluation retain full context.
    Returns (unique_chunks, list_of_removed_chunk_ids).
    """
    if not context:
        return [], []

    unique_chunks = []
    seen_ids = set()
    for c in context:
        cid = c.get("chunk_id")
        if cid:
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(c)
        else:
            unique_chunks.append(c)

    return unique_chunks, []


def _attach_mentioned_visual_chunks(chunks: List[Dict[str, Any]], snapshot: Any) -> None:
    """Find and attach referenced Table/Figure chunks to text chunks that mention them."""
    if not chunks or not snapshot:
        return

    visual_by_mention: Dict[str, List[Dict[str, Any]]] = {}
    visual_by_label: Dict[tuple, Dict[str, Any]] = {}
    label_pattern = re.compile(r"(Figure|Fig\.|Table|Tab\.|图|表)\s*(\d+[a-zA-Z]?)", re.IGNORECASE)

    for i, meta in enumerate(snapshot.metadatas):
        if meta.get("type") in ("table", "image"):
            v_dict = {
                **meta,
                "text": snapshot.chunks[i],
                "chunk_id": snapshot.chunk_ids[i],
            }
            mcid = meta.get("mention_chunk_id")
            if mcid:
                visual_by_mention.setdefault(mcid, []).append(v_dict)

            cap = meta.get("caption")
            cap_text = " ".join(cap) if isinstance(cap, list) else str(cap or "")
            m = label_pattern.search(cap_text)
            if m:
                l_type = "table" if any(k in m.group(1).lower() for k in ["tab", "表"]) else "image"
                l_num = m.group(2).lower()
                visual_by_label[(meta.get("source"), l_type, l_num)] = v_dict

    for c in chunks:
        if c.get("type") != "text":
            continue
        c_source = c.get("source")
        c_id = c.get("chunk_id")
        c_text = c.get("text", "")

        attached = list(c.get("mentioned_visuals") or [])
        seen_v_ids = {v.get("chunk_id") for v in attached if v.get("chunk_id")}

        if c_id and c_id in visual_by_mention:
            for v in visual_by_mention[c_id]:
                vid = v.get("chunk_id")
                if vid and vid not in seen_v_ids:
                    seen_v_ids.add(vid)
                    attached.append(v)

        for m in label_pattern.finditer(c_text):
            l_type = "table" if any(k in m.group(1).lower() for k in ["tab", "表"]) else "image"
            l_num = m.group(2).lower()
            v = visual_by_label.get((c_source, l_type, l_num))
            if v:
                vid = v.get("chunk_id")
                if vid and vid not in seen_v_ids:
                    seen_v_ids.add(vid)
                    attached.append(v)

        if attached:
            c["mentioned_visuals"] = attached


def retrieve(state: AgentState) -> AgentState:
    """Hybrid search → cross-encoder rerank → store chunks in state."""
    query = state["query"]
    options = _options(state)
    targeted = state.get("targeted_sub_queries")
    snapshot = state.get("retrieval_snapshot") or load_snapshot(multimodal=options.multimodal)
    
    result = retrieve_documents(
        query, options, snapshot=snapshot,
        translation_cache=state.get("translation_cache"),
        sub_queries=targeted,
    )
    cycle = (state.get("rewrite_count") or 0) + 1
    step_label = "🔍 首轮多查询分解与多模态初检" if cycle == 1 and not targeted else f"🔍 靶向补检与多模态重排 (第 {cycle} 轮)"
    token_usage, token_breakdown = _track_tokens(state, usage_override=result.get("token_usage"), step_name=step_label)
    
    # Merge with existing context if this is a follow-up retrieval
    existing_context = list(state.get("context") or [])
    new_sources = result.get("sources", [])

    _attach_mentioned_visual_chunks(new_sources, snapshot)
    if existing_context:
        _attach_mentioned_visual_chunks(existing_context, snapshot)

    def _light_doc_for_trace(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: v for k, v in d.items() if k != "asset_images"}
        if d.get("mentioned_visuals"):
            out["mentioned_visuals"] = [
                {k: v for k, v in mv.items() if k != "asset_images"}
                for mv in d["mentioned_visuals"]
                if isinstance(mv, dict)
            ]
        return out

    trace = list(state.get("retrieval_trace", [])) + [{
        "input_query": query,
        "retrieval_query": result.get("retrieval_query", query),
        "queries": result.get("queries", [query]),
        "snapshot_id": result.get("snapshot_id"),
        "chunk_ids": [chunk.get("chunk_id") for chunk in result.get("sources", [])],
        "sub_groups": [
            {
                "sub_query": g.get("sub_query", ""),
                "chunk_ids": [d.get("chunk_id") for d in g.get("docs", []) if d.get("chunk_id")],
                "docs": [_light_doc_for_trace(d) for d in g.get("docs", [])],
            }
            for g in result.get("sub_groups", [])
            if g.get("docs")
        ],
    }]

    rejected_ids = set(state.get("rejected_chunk_ids") or [])
    accepted_ids = set(state.get("accepted_chunk_ids") or [])
    
    # Track brand-new candidate chunks not seen in cache
    brand_new = [s for s in new_sources if s.get("chunk_id") not in accepted_ids and s.get("chunk_id") not in rejected_ids]
    new_candidate_count = len(brand_new)

    sub_groups = result.get("sub_groups", [])
    history_cycles = list(state.get("query_doc_history") or [])
    current_cycle_num = len(history_cycles) + 1
    
    current_gap = state.get("information_gap", "")
    current_cycle_data = {
        "cycle": current_cycle_num,
        "cycle_name": f"第 {current_cycle_num} 轮针对缺口补检" if current_cycle_num > 1 else "首轮多模态初检",
        "input_query": query,
        "gap": current_gap,
        "sub_groups": sub_groups,
        "new_docs": new_sources,
    }
    updated_query_doc_history = history_cycles + [current_cycle_data]

    if existing_context:
        seen_ids = {c.get("chunk_id") for c in existing_context if c.get("chunk_id")}
        seen_texts = {c.get("text", "")[:120] for c in existing_context}
        merged = list(existing_context)
        for s in new_sources:
            cid = s.get("chunk_id")
            snippet = s.get("text", "")[:120]
            # If evaluation cache is enabled, skip previously rejected chunks immediately
            if options.eval_memory and cid and cid in rejected_ids:
                continue
            if cid and cid in seen_ids:
                continue
            if snippet not in seen_texts:
                if cid:
                    seen_ids.add(cid)
                seen_texts.add(snippet)
                merged.append(s)
        final_context, removed_ids = _deduplicate_mention_chunks(merged)
        logger.info(f"[retrieve] Merged previous {len(existing_context)} chunks with new {len(new_sources)} chunks (brand-new={new_candidate_count}) -> total {len(final_context)}")
        dedup_info = f"，并执行跨模态去重剔除图表已覆盖正文 {len(removed_ids)} 处" if removed_ids else ""
        step_detail = (
            f"针对缺口重写查询检索，划分 {len(sub_groups)} 个独立子问题分别完成专属 Cross-Encoder 精排；"
            f"召回 {len(new_sources)} 处候选分块（其中全新增量分块 {new_candidate_count} 处）；"
            f"与历史上下文合并去重后，当前总候选分块共 {len(final_context)} 处{dedup_info}。"
        )
        dec_trace = _append_decision(state, "🔍 针对缺口补充检索与合并", step_detail, {
            "new_sources": len(new_sources),
            "new_candidate_count": new_candidate_count,
            "total_context": len(final_context),
            "removed_ids": removed_ids,
            "current_cycle": current_cycle_data,
            "history_cycles": history_cycles,
            "sub_groups": sub_groups,
        })
    else:
        final_context, removed_ids = _deduplicate_mention_chunks(new_sources)
        dedup_info = f"，触发跨模态图表去重剔除覆盖正文 {len(removed_ids)} 处" if removed_ids else ""
        step_detail = (
            f"首轮检索完成意图分解：产生 {len(sub_groups)} 个子问题并各自独立执行专属 Cross-Encoder 重排；"
            f"全部保留共召回 {len(new_sources)} 处候选分块{dedup_info}，保留 {len(final_context)} 处优质证据分块。"
        )
        dec_trace = _append_decision(state, "🔍 知识召回与跨模态去重", step_detail, {
            "initial_retrieved": len(new_sources),
            "retained": len(final_context),
            "removed_ids": removed_ids,
            "current_cycle": current_cycle_data,
            "history_cycles": history_cycles,
            "sub_groups": sub_groups,
        })

    return {
        **state,
        "context": final_context,
        "retrieval_trace": trace,
        "decision_trace": dec_trace,
        "new_candidate_count": new_candidate_count,
        "query_doc_history": updated_query_doc_history,
        "token_usage": token_usage,
        "token_breakdown": token_breakdown,
    }


def _parse_irrelevant_chunk_ids(eval_resp: str, latest_cids: set[str]) -> set[str]:
    """Extract irrelevant chunk_ids from the merged evaluation response."""
    match = re.search(r"IRRELEVANT_CHUNK_IDS:\s*([^\n]+)", eval_resp, re.IGNORECASE)
    if not match:
        return set()
    raw = match.group(1).strip()
    if not raw or raw.upper() in ("NONE", "N/A", "无", "NULL"):
        return set()

    matched: set[str] = set()
    tokens = [t.strip().lstrip("#") for t in re.split(r"[,;\s]+", raw) if t.strip()]
    for tok in tokens:
        if tok in latest_cids:
            matched.add(tok)
        else:
            for cid in latest_cids:
                if cid.endswith(f"::{tok}") or tok.endswith(cid):
                    matched.add(cid)
    return matched


def grade_documents(state: AgentState) -> AgentState:
    """
    Merged Single-Pass Evaluation with Memory Cache & Circuit Breaker:
    In one LLM call over the Cognitive Evidence Tree, jointly:
    1. Identify irrelevant chunks from the latest retrieval cycle.
    2. Assess overall evidence sufficiency for the original user question.
    3. Formulate targeted, self-contained English sub-questions if insufficient.
    """
    query = state.get("original_query", state["query"])
    context = state.get("context") or []
    rewrite_count = int(state.get("rewrite_count") or 0)
    options = _options(state)
    use_memory = options.eval_memory

    if not options.grade_documents:
        trace = _append_decision(state, "⚖️ 文档相关性校验", "相关性评估开关已关闭，全部召回文档直接送入生成。", {"status": "skipped"})
        return {**state, "_grade": "generate", "decision_trace": trace}

    accepted_ids = set(state.get("accepted_chunk_ids") or [])
    rejected_ids = set(state.get("rejected_chunk_ids") or [])

    if not context:
        trace = _append_decision(state, "⚠️ 检索结果判空", "当前未召回任何候选文档片段，准备补充检索。", None)
        return {
            **state,
            "context": [],
            "_grade": "rewrite",
            "information_gap": "未检索到关于该问题的相关文档内容。",
            "targeted_sub_queries": [query],
            "accepted_chunk_ids": list(accepted_ids),
            "rejected_chunk_ids": list(rejected_ids),
            "decision_trace": trace,
        }

    # Circuit Breaker: If follow-up retrieval brought 0 brand-new valid candidates, stop looping!
    new_candidate_count = state.get("new_candidate_count", len(context))
    if use_memory and rewrite_count > 0 and new_candidate_count == 0:
        logger.info(f"[grade_documents] Circuit breaker triggered at rewrite={rewrite_count}: 0 brand-new candidate chunks retrieved.")
        breaker_detail = (
            f"第 {rewrite_count} 轮补检未从知识库检索到任何全新未审候选分块（新分块增量为 0）。"
            f"判定本地知识库已穷尽（Exhausted），触发早停熔断机制，立即终止本地循环！"
        )
        if options.web_search:
            breaker_detail += " 准备自动无缝退回到联网搜索以补充外部数据。"
            next_grade = "web_search"
        else:
            breaker_detail += " 准备基于当前已有知识库材料直接收敛生成。"
            next_grade = "generate"

        trace = _append_decision(state, "⚡ 知识库枯竭早停熔断 (Circuit Breaker)", breaker_detail, {
            "rewrite_count": rewrite_count,
            "new_candidate_count": new_candidate_count,
            "next_grade": next_grade,
        })
        warnings = _record_warning(state, "⚠️ 本地知识库相关资料已检索完毕且无增量信息，系统已触发早停熔断防止无效重试。")
        return {
            **state,
            "context": context,
            "_grade": next_grade,
            "information_gap": "",
            "accepted_chunk_ids": list(accepted_ids),
            "rejected_chunk_ids": list(rejected_ids),
            "decision_trace": trace,
            "degradation_warnings": warnings,
        }

    # Single-pass merged evaluation: latest-cycle irrelevance filtering + sufficiency + targeted sub-queries
    history_cycles = list(state.get("query_doc_history") or [])
    latest_docs = history_cycles[-1].get("new_docs", []) if history_cycles else context
    latest_cids = {str(d.get("chunk_id")) for d in latest_docs if d.get("chunk_id")}

    all_explored_sub_queries = []
    for cycle in history_cycles:
        for grp in cycle.get("sub_groups", []):
            sq = grp.get("sub_query", "").strip()
            if sq and sq not in all_explored_sub_queries:
                all_explored_sub_queries.append(sq)

    memory_tree = _build_cognitive_memory_tree(state, is_eval=True)
    eval_prompt = (
        "You are an expert evaluator for a multi-turn scientific document retrieval system.\n"
        "Given the Cognitive Evidence Tree below (which contains the original user question, historical exploration cycles with their targeted GAPs, and the latest retrieved chunks), complete the following tasks in a single structured response:\n\n"
        f"{memory_tree}\n\n"
        "Task 1: Filter Irrelevant Chunks from the LATEST Cycle\n"
        "- Inspect ONLY the chunks marked under 【当前最新探索轮次】.\n"
        "- Identify any chunk IDs that are completely irrelevant to BOTH the original user question AND every sub-question.\n"
        "- If all chunks in the latest cycle are relevant or potentially useful, output \"NONE\".\n\n"
        "Task 2: Assess Overall Evidence Sufficiency & Pinpoint Missing Information\n"
        "- Evaluate whether the collective relevant evidence across all cycles in the Cognitive Evidence Tree is SUFFICIENT to completely and specifically answer the original user question.\n"
        "- Reply \"DECISION: SUFFICIENT\" if the collective evidence provides enough specific facts, technical details, or data to fully address the original user question.\n"
        "- Reply \"DECISION: INSUFFICIENT\" if important information needed to fully answer the original user question is still missing or incomplete.\n"
        "- If DECISION is INSUFFICIENT, state what specific missing information is still needed after \"GAP:\".\n\n"
        "Task 3: Plan Targeted English Sub-Questions for the Current GAP (Only if INSUFFICIENT)\n"
        "Based directly on the specific GAP identified above and what has already been retrieved in the Cognitive Evidence Tree:\n"
        "- Determine what exact missing facts, technical details, sections, tables, or figures from the paper are needed to close this GAP.\n"
        "- Formulate concise, self-contained, search-friendly English sub-questions specifically targeting those missing elements.\n"
        "- Do NOT repeat or query topics that have already been retrieved and covered in the Cognitive Evidence Tree.\n"
        "- Mark each sub-question with \"### \" at the beginning.\n\n"
        "Output Format (strictly follow this structure):\n"
        "IRRELEVANT_CHUNK_IDS: <comma-separated chunk_ids from the latest cycle, or NONE>\n"
        "DECISION: <SUFFICIENT or INSUFFICIENT>\n"
        "GAP: <concise description of the missing information needed, or NONE if SUFFICIENT>\n"
        "TARGETED_SUB_QUERIES:\n"
        "### <targeted English sub-question 1>\n"
        "### <targeted English sub-question 2>\n"
        "(or NONE if SUFFICIENT)"
    )

    eval_msg = llm.invoke([HumanMessage(content=eval_prompt)])
    eval_resp = str(getattr(eval_msg, "content", eval_msg)).strip()
    cycle = (state.get("rewrite_count") or 0) + 1
    token_usage, token_breakdown = _track_tokens(
        state, eval_msg, eval_prompt, eval_resp,
        step_name=f"⚖️ 证据融合与缺口研判 (第 {cycle} 轮)"
    )
    state["token_usage"] = token_usage
    state["token_breakdown"] = token_breakdown
    logger.info(f"[grade_documents] Merged evaluation response: {repr(eval_resp[:160])}")

    new_rejected = _parse_irrelevant_chunk_ids(eval_resp, latest_cids)
    new_accepted = latest_cids - new_rejected
    all_accepted = list(accepted_ids | new_accepted)
    all_rejected = list(rejected_ids | new_rejected)

    rejected_set = set(all_rejected)
    relevant = [c for c in context if c.get("chunk_id") not in rejected_set]

    if not relevant:
        next_grade = "generate" if rewrite_count >= MAX_REWRITES else "rewrite"
        trace = _append_decision(
            state,
            "⚖️ 融合评估：相关性过滤与充分性判定",
            f"过滤最新轮无关分块 {len(new_rejected)} 处后，有效证据为空，准备重新检索。",
            {"irrelevant_count": len(new_rejected), "relevant_count": 0, "verdict": "INSUFFICIENT"},
        )
        return {
            **state,
            "context": [],
            "_grade": next_grade,
            "information_gap": "未检索到关于该问题的相关文档内容。",
            "targeted_sub_queries": [query],
            "accepted_chunk_ids": all_accepted,
            "rejected_chunk_ids": all_rejected,
            "decision_trace": trace,
        }

    if rewrite_count >= MAX_REWRITES:
        logger.info(f"[grade_documents] Max rewrites ({MAX_REWRITES}) reached; proceeding to generate with {len(relevant)} filtered chunks.")
        trace = _append_decision(
            state,
            "⏳ 达到最大重试上限",
            f"过滤最新轮无关分块 {len(new_rejected)} 处后，已达到检索上限（{MAX_REWRITES} 次），以当前累积的 {len(relevant)} 处证据分块进入生成。",
            {"irrelevant_count": len(new_rejected), "relevant_count": len(relevant)},
        )
        return {
            **state,
            "context": relevant,
            "_grade": "generate",
            "information_gap": "",
            "accepted_chunk_ids": all_accepted,
            "rejected_chunk_ids": all_rejected,
            "decision_trace": trace,
        }

    is_insufficient = bool(re.search(r"DECISION:\s*INSUFFICIENT", eval_resp, re.IGNORECASE)) or (
        "INSUFFICIENT" in eval_resp and "SUFFICIENT" not in eval_resp.replace("INSUFFICIENT", "")
    )
    if is_insufficient:
        gap_match = re.search(r"GAP:\s*([^\n]+)", eval_resp, re.IGNORECASE)
        gap = gap_match.group(1).strip() if gap_match else f"Missing details needed to fully answer: {query}"

        targeted_sub_queries = []
        if "TARGETED_SUB_QUERIES:" in eval_resp:
            subq_part = eval_resp.split("TARGETED_SUB_QUERIES:")[1].strip()
            seen_history_cf = {sq.casefold() for sq in all_explored_sub_queries}
            for line in subq_part.splitlines():
                clean_q = re.sub(r"^\s*(?:###|[-*•]|\d+[.)])\s*", "", line).strip()
                if clean_q and len(clean_q) > 4:
                    if clean_q.upper() in ("NONE", "END") or clean_q.startswith("DECISION") or clean_q.startswith("GAP"):
                        continue
                    if clean_q.casefold() not in seen_history_cf and clean_q not in targeted_sub_queries:
                        targeted_sub_queries.append(clean_q)
        if not targeted_sub_queries:
            targeted_sub_queries = [gap]

        logger.info(f"[grade_documents] INSUFFICIENT (filtered {len(new_rejected)} irrelevant). Planned GAP: {gap} | Targeted: {targeted_sub_queries}")
        trace = _append_decision(
            state,
            "⚖️ 融合评估：无关块过滤、缺口研判与靶向子问题制定",
            f"过滤最新轮无关分块 {len(new_rejected)} 处（保留 {len(relevant)} 处）；"
            f"判定当前证据尚不充分 (INSUFFICIENT)，定位信息缺口 GAP：'{gap}'；"
            f"直接制定 {len(targeted_sub_queries)} 个正交靶向子问题执行补检：\n"
            + "\n".join(f"  • {q}" for q in targeted_sub_queries),
            {
                "verdict": "INSUFFICIENT",
                "irrelevant_chunk_ids": sorted(new_rejected),
                "relevant_count": len(relevant),
                "gap": gap,
                "targeted_sub_queries": targeted_sub_queries,
            },
        )
        return {
            **state,
            "context": relevant,
            "_grade": "rewrite",
            "information_gap": gap,
            "targeted_sub_queries": targeted_sub_queries,
            "accepted_chunk_ids": all_accepted,
            "rejected_chunk_ids": all_rejected,
            "decision_trace": trace,
        }

    trace = _append_decision(
        state,
        "⚖️ 融合评估：无关块过滤与充分性判定",
        f"过滤最新轮无关分块 {len(new_rejected)} 处（保留 {len(relevant)} 处）；"
        f"评估当前证据已充分 (SUFFICIENT)，直接进入多模态回答生成。",
        {"verdict": "SUFFICIENT", "irrelevant_chunk_ids": sorted(new_rejected), "relevant_count": len(relevant)},
    )
    return {
        **state,
        "context": relevant,
        "_grade": "generate",
        "information_gap": "",
        "accepted_chunk_ids": all_accepted,
        "rejected_chunk_ids": all_rejected,
        "decision_trace": trace,
    }


def rewrite_query(state: AgentState) -> AgentState:
    """
    Corrective RAG: Advance rewrite count and pass targeted sub-queries
    directly formulated by the unified evaluation node to retrieve().
    """
    query = state.get("original_query", state["query"])
    gap = state.get("information_gap", "").strip()
    rewrite_count = int(state.get("rewrite_count") or 0) + 1
    targeted = list(state.get("targeted_sub_queries") or [])

    summary_query = " | ".join(targeted) if targeted else (gap if gap else query)

    logger.info(f"[rewrite_query] attempt {rewrite_count} with {len(targeted)} targeted sub-queries for gap '{gap[:60]}': {targeted}")

    trace = _append_decision(
        state,
        "🔄 针对缺口补充检索",
        f"第 {rewrite_count} 轮补检：根据信息缺口 GAP，直接执行已制定的 {len(targeted)} 个靶向子问题检索：\n"
        + "\n".join(f"  • {q}" for q in targeted),
        {"attempt": rewrite_count, "gap": gap, "targeted_sub_queries": targeted},
    )
    return {
        **state,
        "query": summary_query,
        "targeted_sub_queries": targeted,
        "rewrite_count": rewrite_count,
        "decision_trace": trace,
    }


def web_search(state: AgentState) -> AgentState:
    """Tavily web search as last-resort fallback with explicit transparent warning."""
    query   = state.get("original_query", state["query"])
    warnings = _record_warning(
        state,
        "⚠️ 本地知识库文档未能充分回答此问题，系统已自动降级为联网搜索（Tavily）获取外部公开资料补充。"
    )
    trace = _append_decision(
        state,
        "🌐 联网搜索降级回退 (Web Fallback)",
        "本地知识库信息不足且达到检索上限或枯竭，触发 Tavily 外部联网实时搜索补充事实。",
        {"query": query}
    )
    results = web_search_tool.invoke(query)
    snippets = [r["content"] for r in results if "content" in r]
    logger.info(f"[web_search] got {len(snippets)} web results")
    return {**state, "web_results": snippets, "degradation_warnings": warnings, "decision_trace": trace}


def _extract_context_images(context: List[Dict[str, Any]], state: Optional[AgentState] = None) -> List[Dict[str, Any]]:
    images = []
    seen_img_ids = set()

    def try_add_image(c_meta: Dict[str, Any]) -> None:
        cid = c_meta.get("chunk_id", "")
        if cid and cid in seen_img_ids:
            return
        is_visual = c_meta.get("type") in ("image", "table")
        path = resolve_image_path(c_meta)
        if path and path.is_file():
            mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            try:
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"
                caption_text = c_meta.get("text", "")
                captions = c_meta.get("caption", [])
                if isinstance(captions, list) and captions:
                    caption_text = " ".join(captions)
                images.append({
                    "data_uri": data_uri,
                    "mime": mime,
                    "caption": caption_text,
                    "source": c_meta.get("source", ""),
                    "page": c_meta.get("page", 1),
                    "section": c_meta.get("section") or "未标注章节",
                    "chunk_id": cid,
                    "type": c_meta.get("type", "image"),
                })
                if cid:
                    seen_img_ids.add(cid)
                if "asset_images" not in c_meta:
                    c_meta["asset_images"] = [data_uri]
            except Exception as e:
                logger.warning(f"Failed to read image for LLM: {path}: {e}")
                if state is not None and c_meta.get("type") == "image":
                    _record_warning(
                        state,
                        f"⚠️ 图像文件 [{c_meta.get('source', '')} Page {c_meta.get('page', 1)}] 读取受阻，已降级为仅依据图注文本推理。"
                    )
        elif is_visual and c_meta.get("type") == "image" and state is not None:
            _record_warning(
                state,
                f"⚠️ 图像分块 [{c_meta.get('source', '')} Page {c_meta.get('page', 1)}] 物理路径缺失，已降级为仅依据图注文本推理。"
            )

    for c in context:
        if c.get("type") in ("image", "table"):
            try_add_image(c)
        for v in c.get("mentioned_visuals", []):
            try_add_image(v)

    return images


def generate(state: AgentState) -> AgentState:
    """
    Final generation node — synthesises answer from context + conversation history.
    Adds citations (source + page + chunk_id) when using document context.
    Passes images + captions to DeepSeek / VLM when multimodal context is present.
    """
    query:       str                    = state.get("original_query", state["query"])
    raw_context: List[Dict[str, Any]]  = state.get("context") or []       # type: ignore[assignment]
    context, _                          = _deduplicate_mention_chunks(raw_context)
    web_results: List[str]             = state.get("web_results") or []    # type: ignore[assignment]
    history:     List[BaseMessage]     = state.get("messages") or []       # type: ignore[assignment]

    options = _options(state)
    images_to_send = _extract_context_images(context, state=state) if options.multimodal else []

    # Build context block using Cognitive Working Memory Tree
    if context:
        ctx_block = _build_cognitive_memory_tree(state, is_eval=False)
        source_note = (
            "\n\n【严格推理与引用规则】：\n"
            "1. 你的回答必须沿着上述 Cognitive Evidence Tree 的逻辑探索路径展开，以提供的多轮事实、图表和数据为唯一依据。严禁凭空捏造！\n"
            "2. 每一个要点、结论或具体陈述之后，必须就地紧跟对应的来源引用标识（不要把多个引用集中堆在段落末尾），格式严格为：\n"
            "   [文件名.pdf, #chunk_id]\n"
            "   （例如：[2501.16566v2.pdf, #2501.16566v2.pdf::14]）。\n"
            "3. 严禁给出未标注具体来源文件名和 #chunk_id 的孤立列表或无头陈述！\n"
            "4. 【数学公式与符号规范】：涉及数学符号、公式、变量、维度时，必须使用标准 LaTeX 格式规范排版：行内公式必须用单美元符号包裹（例如 `$Z_{av} \\in \\mathbb{R}^{(t_a+t_v)\\times d}$`），独立大公式必须单独占行并用双美元符号包裹（如 `$$...$$`）。严禁用普通的圆括号 `(...)` 包裹数学公式！"
        )
    elif web_results:
        ctx_block   = "\n\n".join(web_results)
        source_note = "These results are from the web."
    else:
        ctx_block   = ""
        source_note = "Answer from your general knowledge."

    # Conversation history (last 6 turns for context window efficiency)
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-6:]
    )

    system = (
        "You are a precise, helpful multimodal document assistant. "
        + ("Use only the provided context and attached images; if insufficient, state clearly what is missing. " if context or web_results else "")
        + ("When figure or table images are attached, inspect their visual details, labels, and layout carefully. " if images_to_send else "")
        + ("When tables appear in the context, both the parsed text representation and the corresponding table image may be provided. Always cross-check the parsed table text against the attached table image—parsed text may contain formatting or recognition errors such as misaligned columns, missing cells, or symbol errors. If any discrepancy exists between the parsed text and the table image, rely on the table image. " if context else "")
        + ("Always answer in Simplified Chinese, including when the question or source text is English. "
           if options.chinese_answer else "Answer in the language of the user question. ")
        + source_note
    )
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}"
    )
    writer = get_stream_writer()
    parts = []
    usage_chunk = None

    # Attempt multimodal streaming with images if available
    multimodal_stream_succeeded = False
    if images_to_send:
        try:
            multimodal_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}\n\n(Attached {len(images_to_send)} retrieved table/figure image(s) below; use each label to align with its parsed text in Context)"}
            ]
            for img in images_to_send:
                kind_label = "Table" if img.get("type") == "table" else "Figure"
                multimodal_content.append({
                    "type": "text",
                    "text": f"Attached {kind_label} Image for [{img.get('source')}, #{img.get('chunk_id')}]:"
                })
                multimodal_content.append({
                    "type": "image_url",
                    "image_url": {"url": img["data_uri"]}
                })
            
            for chunk in llm.stream([HumanMessage(content=multimodal_content)]):
                if getattr(chunk, "usage_metadata", None) or (getattr(chunk, "response_metadata", None) or {}).get("token_usage"):
                    usage_chunk = chunk
                if isinstance(chunk.content, str) and chunk.content:
                    parts.append(chunk.content)
                    writer({"type": "token", "content": chunk.content})
            multimodal_stream_succeeded = True
        except Exception as exc:
            logger.warning(f"Multimodal image streaming failed with LLM ({exc}); falling back gracefully to text prompt with captions.")
            _record_warning(state, "⚠️ 大模型图像通道请求异常，已降级为纯文本提示词交互。")
            parts.clear()
            usage_chunk = None

    if not multimodal_stream_succeeded:
        for chunk in llm.stream([
            HumanMessage(content=f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}")
        ]):
            if getattr(chunk, "usage_metadata", None) or (getattr(chunk, "response_metadata", None) or {}).get("token_usage"):
                usage_chunk = chunk
            if isinstance(chunk.content, str) and chunk.content:
                parts.append(chunk.content)
                writer({"type": "token", "content": chunk.content})

    answer = "".join(parts)
    token_usage, token_breakdown = _track_tokens(
        state, usage_chunk, f"{system}\n{user_prompt}", answer,
        step_name="📝 终审多模态生成"
    )
    logger.info(f"[generate] answer length={len(answer)} chars, multimodal_images={len(images_to_send)}")
    
    trace = _append_decision(
        state,
        "📝 终审多模态生成",
        f"装配 {len(context)} 处高相关证据分块（含 {len(images_to_send)} 处多模态原图切片），生成完整因果链分析并打上唯一 chunk_id 原文精准引用。",
        {"context_count": len(context), "images_count": len(images_to_send)}
    )
    return {
        **state,
        "answer":   answer,
        "context":  context,
        "messages": state["messages"] + [AIMessage(content=answer)],
        "decision_trace": trace,
        "token_usage": token_usage,
        "token_breakdown": token_breakdown,
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_routing(state: AgentState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.get("_route") == "documents" else "generate"  # type: ignore[union-attr]


def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "web_search"]:
    grade = state.get("_grade") or "generate"                               # type: ignore[union-attr]
    if grade == "generate":
        return "generate"
    if grade == "web_search":
        return "web_search"
    options = _options(state)
    if options.corrective_rag and int(state.get("rewrite_count") or 0) < MAX_REWRITES:
        return "rewrite_query"
    return "web_search" if options.web_search else "generate"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route_query",     route_query)
    g.add_node("retrieve",        retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query",   rewrite_query)
    g.add_node("web_search",      web_search)
    g.add_node("generate",        generate)

    g.set_entry_point("route_query")

    g.add_conditional_edges("route_query",     route_after_routing,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve",         "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            {"generate": "generate",
                             "rewrite_query": "rewrite_query",
                             "web_search": "web_search"})
    g.add_edge("rewrite_query",    "retrieve")
    g.add_edge("web_search",       "generate")
    g.add_edge("generate",         END)

    return g.compile()


# Singleton compiled graph
rag_agent = build_graph()


# ── Public interface ──────────────────────────────────────────────────────────

def _initial_state(query: str, history=None, config=None, snapshot=None, translation_cache=None) -> AgentState:
    messages = (history or []) + [HumanMessage(content=query)]
    return {
        "messages":      messages,
        "query":         query,
        "original_query": query,
        "pipeline_config": (config or PipelineConfig()).model_dump(),
        "retrieval_snapshot": snapshot,
        "translation_cache": translation_cache,
        "retrieval_trace": [],
        "rewrite_count": 0,
        "context":       [],
        "web_results":   [],
        "answer":        "",
        "information_gap": "",
        "degradation_warnings": [],
        "accepted_chunk_ids": [],
        "rejected_chunk_ids": [],
        "decision_trace": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "token_breakdown": [],
        "new_candidate_count": 0,
        "query_doc_history": [],
        "targeted_sub_queries": [],
    }


def _result(final_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer":               final_state.get("answer", ""),
        "sources":              final_state.get("context", []),
        "route":                final_state.get("_route", "documents"),
        "rewrite_count":        final_state.get("rewrite_count", 0),
        "used_web":             bool(final_state.get("web_results")),
        "web_contexts":         final_state.get("web_results", []),
        "retrieval_trace":      final_state.get("retrieval_trace", []),
        "pipeline_config":      final_state.get("pipeline_config", {}),
        "degradation_warnings": final_state.get("degradation_warnings", []),
        "decision_trace":       final_state.get("decision_trace", []),
        "token_usage":          final_state.get("token_usage", {}),
        "token_breakdown":      final_state.get("token_breakdown", []),
        "total_tokens":         (final_state.get("token_usage") or {}).get("total_tokens", 0),
        "query_doc_history":    final_state.get("query_doc_history", []),
        "original_query":       final_state.get("original_query", ""),
    }


def ask(query: str, history: Optional[List[BaseMessage]] = None, *, config=None, snapshot=None, translation_cache=None) -> Dict[str, Any]:
    """Complete-response interface for evaluation and CLI clients."""
    return _result(rag_agent.invoke(_initial_state(query, history, config, snapshot, translation_cache)))


async def astream_answer(query: str, history: Optional[List[BaseMessage]] = None, *, config=None):
    """Stream stage updates, final-answer tokens and the completed result."""
    yield {"type": "status", "content": "正在分析问题…"}
    async for mode, update in rag_agent.astream(
        _initial_state(query, history, config), stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            yield update
            continue
        for node, state in update.items():
            if node == "generate":
                yield {"type": "done", **_result(state)}
            else:
                if node == "route_query":
                    label = "正在检索文档…" if state.get("_route") == "documents" else "正在生成回答…"
                elif node == "retrieve":
                    label = "正在评估文档充分性与缺口…"
                elif node == "grade_documents":
                    next_node = route_after_grading(state)
                    label = {"generate": "正在生成回答…", "rewrite_query": "正在针对信息缺口补充检索…", "web_search": "正在降级搜索网页…"}[next_node]
                elif node == "rewrite_query":
                    label = "正在针对缺口补充检索文档…"
                else:
                    label = "正在生成回答…"
                yield {"type": "status", "content": label}
