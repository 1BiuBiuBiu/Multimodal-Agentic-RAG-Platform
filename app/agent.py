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


def _track_tokens(state: AgentState, response: Any = None, prompt_text: str = "", completion_text: str = "") -> Dict[str, int]:
    """Accurately accumulate token consumption from LLM calls (API metadata + fallback estimation)."""
    usage = dict(state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    p_tok = 0
    c_tok = 0
    if response is not None:
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = response.usage_metadata.get("input_tokens") or response.usage_metadata.get("prompt_tokens") or 0
            c_tok = response.usage_metadata.get("output_tokens") or response.usage_metadata.get("completion_tokens") or 0
        elif hasattr(response, "response_metadata") and response.response_metadata:
            meta_usage = response.response_metadata.get("token_usage") or {}
            p_tok = meta_usage.get("prompt_tokens", 0)
            c_tok = meta_usage.get("completion_tokens", 0)

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
    return usage


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

    if not history_cycles:
        context = state.get("context") or []
        sections.append("\n▶ [当前检索证据组]：")
        for idx, c in enumerate(context[:6]):
            cid = c.get("chunk_id", f"#{idx+1}")
            src = c.get("source", "doc")
            p = c.get("page", 1)
            kind = c.get("type", "text")
            label = {"text": "正文", "table": "Table / 表格", "image": "Figure / 图片"}.get(kind, kind)
            snip = c.get("text", "")
            if is_eval:
                sections.append(f"  • [{src}, Page {p}, ID: {cid}, Type: {label}]: {snip[:300]}...")
            else:
                body = snip
                if kind in ("table", "image") and c.get("mention_context"):
                    body += f"\n    【正文引用与深度解读】：\n    {c['mention_context']}"
                sections.append(f"  • [Source: {src}, Page {p}, ID: {cid}, Type: {label}]\n{body}")
        return "\n".join(sections)

    for c_idx, cycle in enumerate(history_cycles):
        c_num = cycle.get("cycle", c_idx + 1)
        c_name = cycle.get("cycle_name", f"第 {c_num} 轮检索")
        input_q = cycle.get("input_query", "")
        sub_groups = cycle.get("sub_groups", [])
        
        is_latest = (c_idx == len(history_cycles) - 1)
        cycle_header = f"▶ 【当前最新探索轮次】{c_name}" if is_latest else f"▶ 【历史探索轮次】{c_name}"
        sections.append(f"\n{cycle_header} (输入查询: `{input_q}`)")

        if not sub_groups:
            docs = cycle.get("new_docs", [])
            sections.append("  └─ 专属证据组:")
            for d in docs[:4]:
                cid = d.get("chunk_id", "N/A")
                snip = d.get("text", "")
                if cid != "N/A" and cid in seen_chunk_ids:
                    ref_loc = first_location.get(cid, "此前轮次")
                    sections.append(f"     • [{d.get('source')} p.{d.get('page')} #{cid}] (← 亦高度支撑本轮次，详见 {ref_loc} 证据)")
                    continue
                if cid != "N/A":
                    seen_chunk_ids.add(cid)
                    first_location[cid] = f"第 {c_num} 轮检索"
                if is_eval:
                    sections.append(f"     • [{d.get('source')} p.{d.get('page')} #{cid}]: {snip[:260]}...")
                else:
                    sections.append(f"     • [Source: {d.get('source')}, Page {d.get('page')}, #{cid}]:\n{snip}")
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
                    for d in docs[:4]:
                        cid = d.get("chunk_id", "N/A")
                        kind = d.get("type", "text")
                        label = {"text": "正文", "table": "Table / 表格", "image": "Figure / 图片"}.get(kind, kind)
                        
                        # Cross-reference deduplication pointer
                        if cid != "N/A" and cid in seen_chunk_ids:
                            first_ref = first_location.get(cid, "此前子问题")
                            if is_eval:
                                sections.append(f"         • [{d.get('source')} p.{d.get('page')} #{cid}, {label}] (← 亦高度支撑本子问题，内容见 {first_ref})")
                            else:
                                sections.append(f"         • [Source: {d.get('source')}, Page {d.get('page')}, #{cid}, Type: {label}] (← 亦高度支撑本子问题，详见上方 {first_ref} 完整证据)")
                            continue

                        # First encounter: record and render in full
                        if cid != "N/A":
                            seen_chunk_ids.add(cid)
                            first_location[cid] = loc_desc

                        snip = d.get("text", "")
                        if is_eval:
                            sections.append(f"         • [{d.get('source')} p.{d.get('page')} #{cid}, {label}]: {snip[:260]}...")
                        else:
                            body = snip
                            if kind in ("table", "image") and d.get("mention_context"):
                                body += f"\n           【该{label}在正文中的完整解读】：\n           {d['mention_context']}"
                            sections.append(f"         • [Source: {d.get('source')}, Page {d.get('page')}, #{cid}, Type: {label}]:\n{body}")

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
    token_usage = _track_tokens(state, response, prompt)
    decision = response.content.strip().lower()
    if "documents" not in decision:
        decision = "llm_only"
    logger.info(f"[route_query] decision='{decision}' for query='{query[:60]}'")
    
    route_label = "知识库多模态检索 (documents)" if decision == "documents" else "通用大模型直接回答 (llm_only)"
    trace = _append_decision(state, "🟢 意图路由", f"分析用户意图，决策路由至：{route_label}", {"route": decision})
    return {**state, "query": query, "_route": decision, "decision_trace": trace, "token_usage": token_usage}


def _deduplicate_mention_chunks(context: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[str]]:
    """
    Deduplicate redundant text chunks that are strictly covered by retrieved Table/Figure chunks.
    Preserves Table/Figure chunks with their rich mention metadata as primary.
    Returns (deduped_context, list_of_removed_chunk_ids).
    """
    if not context:
        return [], []

    # 1. Deduplicate identical chunk_ids first
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

    # 2. Collect all text chunk_ids mentioned by retrieved tables/images
    covered_chunk_ids = {
        c.get("mention_chunk_id")
        for c in unique_chunks
        if c.get("type") in ("table", "image") and c.get("mention_chunk_id")
    }

    # 3. Filter out pure text chunks whose full paragraph is already covered by a table/image
    deduped = []
    removed_ids = []
    for c in unique_chunks:
        if c.get("type") == "text" and c.get("chunk_id") in covered_chunk_ids:
            removed_ids.append(c.get("chunk_id"))
            continue
        deduped.append(c)

    if removed_ids:
        logger.info(f"[dedup] Deduplicated {len(removed_ids)} text chunks covered by tables/images: {removed_ids}")

    return deduped, removed_ids


def retrieve(state: AgentState) -> AgentState:
    """Hybrid search → cross-encoder rerank → store chunks in state."""
    query = state["query"]
    options = _options(state)
    targeted = state.get("targeted_sub_queries")
    
    result = retrieve_documents(
        query, options, snapshot=state.get("retrieval_snapshot"),
        translation_cache=state.get("translation_cache"),
        sub_queries=targeted,
    )
    
    trace = list(state.get("retrieval_trace", [])) + [{
        "input_query": query,
        "retrieval_query": result.get("retrieval_query", query),
        "queries": result.get("queries", [query]),
        "snapshot_id": result.get("snapshot_id"),
        "chunk_ids": [chunk.get("chunk_id") for chunk in result.get("sources", [])],
    }]
    
    # Merge with existing context if this is a follow-up retrieval
    existing_context = list(state.get("context") or [])
    new_sources = result.get("sources", [])
    
    rejected_ids = set(state.get("rejected_chunk_ids") or [])
    accepted_ids = set(state.get("accepted_chunk_ids") or [])
    
    # Track brand-new candidate chunks not seen in cache
    brand_new = [s for s in new_sources if s.get("chunk_id") not in accepted_ids and s.get("chunk_id") not in rejected_ids]
    new_candidate_count = len(brand_new)

    sub_groups = result.get("sub_groups", [])
    history_cycles = list(state.get("query_doc_history") or [])
    current_cycle_num = len(history_cycles) + 1
    
    current_cycle_data = {
        "cycle": current_cycle_num,
        "cycle_name": f"第 {current_cycle_num} 轮针对缺口补检" if current_cycle_num > 1 else "首轮多模态初检",
        "input_query": query,
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
    }


def grade_documents(state: AgentState) -> AgentState:
    """
    Two-stage Evaluation with Memory Cache & Circuit Breaker:
    Stage 1: Filter irrelevant chunks (with evaluation cache to avoid duplicate LLM calls).
    Stage 2: Assess Information Sufficiency & Gap.
             If brand-new candidates are 0 and context is insufficient, trigger circuit breaker.
    """
    query   = state.get("original_query", state["query"])
    context = state.get("context") or []
    rewrite_count = int(state.get("rewrite_count") or 0)
    options = _options(state)
    use_memory = options.eval_memory
    
    if not options.grade_documents:
        trace = _append_decision(state, "⚖️ 文档相关性校验", "相关性评估开关已关闭，全部召回文档直接送入生成。", {"status": "skipped"})
        return {**state, "_grade": "generate", "decision_trace": trace}

    # Stage 1: Individual chunk relevance (with Evaluation Cache)
    gap = state.get("information_gap", "")
    gap_context = f" or relevant to addressing the missing gap: '{gap}'" if gap else ""
    
    accepted_ids = set(state.get("accepted_chunk_ids") or [])
    rejected_ids = set(state.get("rejected_chunk_ids") or [])
    
    relevant = []
    new_accepted = set()
    new_rejected = set()
    cached_hits = 0
    new_evaluated = 0

    for chunk in context:
        cid = chunk.get("chunk_id")
        if use_memory and cid:
            if cid in rejected_ids:
                cached_hits += 1
                continue
            if cid in accepted_ids:
                cached_hits += 1
                relevant.append(chunk)
                continue

        # Evaluate brand-new chunk with LLM
        new_evaluated += 1
        is_img = chunk.get("type") == "image"
        chunk_desc = f"[Figure/Image Information]\n{chunk['text']}" if is_img else chunk["text"]
        prompt = (
            f"Question: {query}\n"
            + (f"Information Gap being addressed: {gap}\n" if gap else "")
            + f"\nDocument chunk ({'Figure/Image' if is_img else 'Text'}):\n{chunk_desc[:1200]}\n\n"
            f"Is this chunk relevant to answering the question{gap_context}? Reply yes or no."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        token_usage = _track_tokens(state, resp, prompt)
        state["token_usage"] = token_usage

        if "yes" in resp.content.lower():
            relevant.append(chunk)
            if cid:
                new_accepted.add(cid)
        else:
            if cid:
                new_rejected.add(cid)

    all_accepted = list(accepted_ids | new_accepted)
    all_rejected = list(rejected_ids | new_rejected)

    cache_note = f"（命中评估记忆池免审 {cached_hits} 处，全新调用打分 {new_evaluated} 处）" if use_memory else "（评估记忆池未启用）"
    stage1_detail = f"对 {len(context)} 处候选分块完成相关性校验{cache_note}：保留高相关分块 {len(relevant)} 处，丢弃不相关分块 {len(context) - len(relevant)} 处。"
    trace = _append_decision(state, "⚖️ Stage 1: 文档相关性校验", stage1_detail, {
        "total_evaluated": len(context),
        "cached_hits": cached_hits,
        "new_evaluated": new_evaluated,
        "relevant_count": len(relevant),
    })

    logger.info(f"[grade_documents] Stage 1 relevance: {len(relevant)}/{len(context)} relevant (cache_hits={cached_hits}, new_eval={new_evaluated})")

    if not relevant:
        trace = _append_decision(state, "⚠️ 检索结果判空", "当前未筛选出任何与问题相关的有效文档片段，准备重写查询。", None)
        return {
            **state,
            "context": [],
            "_grade": "rewrite",
            "information_gap": "未检索到关于该问题的相关文档内容。",
            "accepted_chunk_ids": all_accepted,
            "rejected_chunk_ids": all_rejected,
            "decision_trace": trace,
        }

    # Circuit Breaker: If follow-up retrieval brought 0 brand-new valid candidates, stop looping!
    new_candidate_count = state.get("new_candidate_count", len(context))
    if use_memory and rewrite_count > 0 and new_candidate_count == 0:
        logger.info(f"[grade_documents] Circuit breaker triggered at rewrite={rewrite_count}: 0 brand-new candidate chunks retrieved.")
        breaker_detail = (
            f"第 {rewrite_count} 轮补检检索未从知识库检索到任何全新未审候选分块（新分块增量为 0）。"
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
            "next_grade": next_grade
        })
        warnings = _record_warning(state, "⚠️ 本地知识库相关资料已检索完毕且无增量信息，系统已触发早停熔断防止无效重试。")
        return {
            **state,
            "context": relevant,
            "_grade": next_grade,
            "information_gap": "",
            "accepted_chunk_ids": all_accepted,
            "rejected_chunk_ids": all_rejected,
            "decision_trace": trace,
            "degradation_warnings": warnings,
        }

    # If we already reached max rewrites, proceed directly to generate
    if rewrite_count >= MAX_REWRITES:
        logger.info(f"[grade_documents] Max rewrites ({MAX_REWRITES}) reached; proceeding to generate with available context.")
        trace = _append_decision(state, "⏳ 达到最大重试上限", f"已达到重写检索上限（{MAX_REWRITES} 次），以当前累积的 {len(relevant)} 处证据分块进入生成。", None)
        return {
            **state,
            "context": relevant,
            "_grade": "generate",
            "information_gap": "",
            "accepted_chunk_ids": all_accepted,
            "rejected_chunk_ids": all_rejected,
            "decision_trace": trace,
        }

    # Stage 2: Information Sufficiency Assessment using Cognitive Working Memory Tree
    memory_tree = _build_cognitive_memory_tree(state, is_eval=True)
    sufficiency_prompt = (
        "You are an expert evaluator assessing whether the retrieved context across multi-turn exploration is SUFFICIENT to fully and specifically answer the user's question.\n\n"
        f"{memory_tree}\n\n"
        "Evaluation Guidelines:\n"
        "- Carefully review the multi-turn exploration memory tree above: compare what was missing in earlier cycles with the newly retrieved evidence in the current cycle.\n"
        "- Reply 'DECISION: SUFFICIENT' if the collective evidence across all cycles contains enough specific facts, metrics, module details, or benchmark tables to give a complete, concrete answer.\n"
        "- Reply 'DECISION: INSUFFICIENT' if the context is relevant but missing core details (e.g. mentions overall superiority but lacks specific task scores/tables, or mentions a model but lacks the key requested sub-components).\n"
        "If INSUFFICIENT, directly formulate 1 to 3 distinct, highly targeted atomic English search sub-queries to fill the exact missing gaps (such as specific table numbers, concrete dataset statistics, or ablation modules).\n"
        "Format your response strictly as:\n"
        "DECISION: INSUFFICIENT\n"
        "GAP: <concise English phrase explaining what specific table, task list, or data is missing>\n"
        "TARGETED_SUB_QUERIES:\n"
        "- <targeted English sub-query 1>\n"
        "- <targeted English sub-query 2>\n\n"
        "Your verdict:"
    )
    suff_resp = str(llm.invoke([HumanMessage(content=sufficiency_prompt)]).content).strip()
    token_usage = _track_tokens(state, None, sufficiency_prompt, suff_resp)
    state["token_usage"] = token_usage

    logger.info(f"[grade_documents] Stage 2 Sufficiency verdict: {repr(suff_resp[:120])}")

    if "DECISION: INSUFFICIENT" in suff_resp or "INSUFFICIENT" in suff_resp:
        gap = ""
        gap_match = re.search(r"GAP:\s*([^\n]+)", suff_resp, re.IGNORECASE)
        if gap_match:
            gap = gap_match.group(1).strip()
        else:
            gap = "specific benchmark comparison tables and detailed task metric scores"

        # Parse targeted sub-queries formulated directly by Stage 2
        targeted_sub_queries = []
        if "TARGETED_SUB_QUERIES:" in suff_resp:
            subq_part = suff_resp.split("TARGETED_SUB_QUERIES:")[1].strip()
            for line in subq_part.splitlines():
                clean_q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
                if clean_q and len(clean_q) > 4 and not clean_q.startswith("DECISION") and not clean_q.startswith("GAP"):
                    targeted_sub_queries.append(clean_q)
        if not targeted_sub_queries:
            targeted_sub_queries = [gap]

        logger.info(f"[grade_documents] Context deemed INSUFFICIENT. Identified Gap: {gap} | Targeted Sub-queries: {targeted_sub_queries}")
        trace = _append_decision(
            state,
            "🤔 Stage 2: 证据充分性与缺口评估",
            f"评估当前证据尚不充分 (INSUFFICIENT)。精准定位信息缺口 GAP：'{gap}'，直接制定 {len(targeted_sub_queries)} 个靶向补检子问题触发定向补检。",
            {"verdict": "INSUFFICIENT", "gap": gap, "targeted_sub_queries": targeted_sub_queries}
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
    
    trace = _append_decision(state, "🤔 Stage 2: 证据充分性与缺口评估", f"评估当前证据已充分具体 (SUFFICIENT)。包含完整技术事实与指标数据，直接进入多模态回答生成。", {"verdict": "SUFFICIENT"})
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
    Corrective RAG: Advance rewrite count and prepare targeted follow-up sub-queries
    directly formulated by Stage 2 sufficiency evaluation (eliminating redundant LLM roundtrips).
    """
    query = state.get("original_query", state["query"])
    gap = state.get("information_gap", "")
    rewrite_count = int(state.get("rewrite_count") or 0) + 1
    targeted = list(state.get("targeted_sub_queries") or [])

    if not targeted and gap:
        # Fallback if targeted subqueries were not provided
        prompt = (
            f"Original Question: {query}\n"
            f"Missing Information Gap: {gap}\n\n"
            "Generate 1 to 2 precise, technical English retrieval queries specifically targeting the missing information.\n"
            "Return one query per line, without bullets or numbers."
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        token_usage = _track_tokens(state, response, prompt)
        state["token_usage"] = token_usage
        lines = [line.strip() for line in str(response.content).splitlines() if line.strip()]
        targeted = lines if lines else [gap]

    summary_query = " | ".join(targeted)
    logger.info(f"[rewrite_query] attempt {rewrite_count} targeting gap '{gap[:40]}': {targeted}")
    
    trace = _append_decision(
        state,
        "🔄 针对缺口重写查询",
        f"结合已识别信息缺口 GAP 直接制定 {len(targeted)} 个靶向补检子问题（第 {rewrite_count} 轮补检）：\n"
        + "\n".join(f"  • {q}" for q in targeted),
        {"attempt": rewrite_count, "gap": gap, "targeted_sub_queries": targeted}
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
    for c in context:
        is_visual = c.get("type") in ("image", "table")
        path = resolve_image_path(c)
        if path and path.is_file():
            mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            try:
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                data_uri = f"data:{mime};base64,{b64}"
                caption_text = c.get("text", "")
                captions = c.get("caption", [])
                if isinstance(captions, list) and captions:
                    caption_text = " ".join(captions)
                images.append({
                    "data_uri": data_uri,
                    "mime": mime,
                    "caption": caption_text,
                    "source": c.get("source", ""),
                    "page": c.get("page", 1),
                    "chunk_id": c.get("chunk_id", ""),
                    "type": c.get("type", "image"),
                })
                if "asset_images" not in c:
                    c["asset_images"] = [data_uri]
            except Exception as e:
                logger.warning(f"Failed to read image for LLM: {path}: {e}")
                if state is not None and c.get("type") == "image":
                    _record_warning(
                        state,
                        f"⚠️ 图像文件 [{c.get('source', '')} Page {c.get('page', 1)}] 读取受阻，已降级为仅依据图注文本推理。"
                    )
        elif is_visual and c.get("type") == "image" and state is not None:
            _record_warning(
                state,
                f"⚠️ 图像分块 [{c.get('source', '')} Page {c.get('page', 1)}] 物理路径缺失，已降级为仅依据图注文本推理。"
            )
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
            "2. 每一个要点、结论、具体数值或任务表现之后，必须紧跟来源引用标识，格式严格为：[文件名.pdf, Page X, #chunk_id]（例如：[2501.16566v2.pdf, Page 7, #2501.16566v2.pdf::14]）。\n"
            "3. 严禁给出未标注具体来源页码和 #chunk_id 的孤立列表或无头陈述！"
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
        + ("Use only the provided context and images; if insufficient, state clearly what is missing. " if context or web_results else "")
        + ("When images are provided in the context, refer to their visual details and captions accurately. " if images_to_send else "")
        + ("When tables are provided in the context, analyze all columns, rows, bold numbers, and baseline comparisons accurately. " if context else "")
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

    # Attempt multimodal streaming with images if available
    multimodal_stream_succeeded = False
    if images_to_send:
        try:
            multimodal_content = [
                {"type": "text", "text": f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}\n\n(Attached {len(images_to_send)} retrieved figure(s) with captions below)"}
            ]
            for img in images_to_send:
                multimodal_content.append({
                    "type": "image_url",
                    "image_url": {"url": img["data_uri"]}
                })
            
            for chunk in llm.stream([HumanMessage(content=multimodal_content)]):
                if isinstance(chunk.content, str) and chunk.content:
                    parts.append(chunk.content)
                    writer({"type": "token", "content": chunk.content})
            multimodal_stream_succeeded = True
        except Exception as exc:
            logger.warning(f"Multimodal image streaming failed with LLM ({exc}); falling back gracefully to text prompt with captions.")
            _record_warning(state, "⚠️ 大模型图像通道请求异常，已降级为纯文本提示词交互。")
            parts.clear()

    if not multimodal_stream_succeeded:
        for chunk in llm.stream([
            HumanMessage(content=f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}")
        ]):
            if isinstance(chunk.content, str) and chunk.content:
                parts.append(chunk.content)
                writer({"type": "token", "content": chunk.content})

    answer = "".join(parts)
    token_usage = _track_tokens(state, None, user_prompt, answer)
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
        "new_candidate_count": 0,
        "query_doc_history": [],
        "targeted_sub_queries": [],
    }


def _result(final_state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "answer":               final_state.get("answer", ""),
        "sources":              final_state.get("context", []),
        "rewrite_count":        final_state.get("rewrite_count", 0),
        "used_web":             bool(final_state.get("web_results")),
        "web_contexts":         final_state.get("web_results", []),
        "retrieval_trace":      final_state.get("retrieval_trace", []),
        "pipeline_config":      final_state.get("pipeline_config", {}),
        "degradation_warnings": final_state.get("degradation_warnings", []),
        "decision_trace":       final_state.get("decision_trace", []),
        "token_usage":          final_state.get("token_usage", {}),
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
