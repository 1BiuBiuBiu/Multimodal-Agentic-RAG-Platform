"""English query planning, LangChain MultiQueryRetriever and late reranking."""
import re
from collections import Counter
from concurrent.futures import Future
from threading import Lock
from typing import Any

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.llm import create_llm
from app.pipeline_config import PipelineConfig
from embeddings.embedder import rerank_indices, rerank_scores
from ingestion.chunking import is_pure_toc_chunk
from vectorstore.store import load_snapshot, search_candidates


class TranslationCache:
    """One batch shares the same English translation across all variants."""
    def __init__(self):
        self.lock = Lock()
        self.futures = {}

    def get(self, key, translate):
        with self.lock:
            owner = key not in self.futures
            future = self.futures.setdefault(key, Future())
        if owner:
            try:
                future.set_result(translate())
            except Exception as exc:
                future.set_exception(exc)
        return future.result()


def _is_english_query(text):
    return not re.search(r"[\u3400-\u9fff]", text)


def english_query(question, llm, cache=None):
    if _is_english_query(question):
        return question.strip()
    def translate():
        answer = str(llm.invoke(
            "Translate the following retrieval question into English. Preserve all names, "
            "acronyms, document references and technical details. "
            "Return only the English question.\nQuestion: " + question
        ).content).strip()
        if not answer or not _is_english_query(answer):
            raise ValueError("查询翻译未返回有效英文，请重试。")
        return answer
    return cache.get(question, translate) if cache else translate()


class SnapshotRetriever(BaseRetriever):
    snapshot: Any
    options: PipelineConfig
    queries: list[str] = Field(default_factory=list)
    ranks: list[list[int]] = Field(default_factory=list)

    def _get_relevant_documents(self, query, *, run_manager):
        if self.options.translate_query and not _is_english_query(query):
            raise ValueError("多查询扩展包含非英文查询，请重试。")
        self.queries.append(query)
        ids = search_candidates(query, self.snapshot, self.options)
        self.ranks.append(ids)
        return [Document(page_content=self.snapshot.chunks[i], metadata={
            **self.snapshot.metadatas[i], "index_id": i,
            "chunk_id": self.snapshot.chunk_ids[i],
        }) for i in ids]


class QueryExpander(MultiQueryRetriever):
    def generate_queries(self, question, run_manager):
        lines = super().generate_queries(question, run_manager)
        result, seen = [], {question.casefold().strip()}
        for line in lines:
            query = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if query and query.casefold() not in seen:
                seen.add(query.casefold())
                result.append(query)
        return result


def _extract_response_tokens(response: Any, prompt_text: str = "", completion_text: str = "") -> dict[str, int]:
    """Extract real token usage from LLM response metadata, falling back to char estimate if absent."""
    p_tok = 0
    c_tok = 0
    if response is not None:
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            p_tok = usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0
            c_tok = usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0
        else:
            resp_meta = getattr(response, "response_metadata", None) or {}
            meta_usage = resp_meta.get("token_usage") or {}
            p_tok = meta_usage.get("prompt_tokens", 0)
            c_tok = meta_usage.get("completion_tokens", 0)

    if p_tok == 0 and prompt_text:
        p_tok = max(1, len(prompt_text) // 4)
    if c_tok == 0:
        if completion_text:
            c_tok = max(1, len(completion_text) // 4)
        elif response is not None and getattr(response, "content", None):
            c_tok = max(1, len(str(response.content)) // 4)

    return {
        "prompt_tokens": int(p_tok),
        "completion_tokens": int(c_tok),
        "total_tokens": int(p_tok) + int(c_tok),
    }


def _parse_decomposed_queries(raw_text: str, original_question: str) -> list[str]:
    """Parse 'End | <query>' or '### <sub-question>' lines from decomposition LLM output."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return [original_question]

    first = lines[0]
    if re.match(r"^end(?:\s*\||$)", first, re.IGNORECASE):
        if "|" in first:
            single = first.split("|", 1)[1].strip()
            if single:
                return [single]
        return [original_question]

    hash_lines = [
        re.sub(r"^###\s*", "", line).strip()
        for line in lines
        if line.startswith("###")
    ]
    candidates = hash_lines if hash_lines else [
        re.sub(r"^\s*(?:###|[-*•]|\d+[.)])\s*", "", line).strip()
        for line in lines
    ]

    unique_lines = []
    for item in candidates:
        if item and item not in unique_lines:
            unique_lines.append(item)
    return unique_lines or [original_question]


def retrieve_documents(question, config=None, *, llm=None, snapshot=None, translation_cache=None, sub_queries=None):
    config = config or PipelineConfig()
    snapshot = snapshot or load_snapshot(multimodal=config.multimodal)
    llm = llm or create_llm(temperature=0)
    base_query = question
    retrieval_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 1. Sub-question planning (If caller already provided targeted sub-queries, use them directly!)
    if sub_queries:
        # Direct bypass from merged sufficiency evaluation targeted sub-queries
        sub_queries = [q.strip() for q in sub_queries if q and q.strip()]
    elif config.multi_query:
        target_language_instruction = (
            "Write all output sub-questions in English, translating terms accurately while preserving proper nouns, names, and acronyms."
            if config.translate_query
            else "Write all output sub-questions in the same language as Q."
        )
        prompt_text = (
            "You are an expert retrieval planner for a document retrieval system.\n"
            "Your task is to analyze the given input (which can be a user question or a description of missing information), translate it if required, and plan the essential information items needed from the documents to answer the question or satisfy the goal.\n\n"
            "Language Rule:\n"
            f"- {target_language_instruction}\n\n"
            "Planning & Retrieval Guidelines:\n"
            "- Determine what specific information items or evidence blocks are required to address the input.\n"
            "- If the input is already a simple, single-focus request that requires only one information item, output exactly one line starting with \"End | \" followed by that single retrieval target in the target language.\n"
            "- Otherwise, list the distinct, search-friendly information items needed—one target per line.\n"
            "- Each item MUST be concise, self-contained, and focused on a single concrete information target (avoid question-style filler words like \"what are the specific...\" or conversational padding).\n"
            "- Avoid over-splitting: Do NOT generate redundant, near-duplicate, or overlapping items for the same information target.\n"
            "- Mark each information item with \"### \" at the beginning, one per line.\n\n"
            "=== Examples ===\n"
            "Example 1 (Compound multi-aspect question -> list of required information items):\n"
            "Question: 机器学习模型在不同数据预处理技术下的准确率有何差异？每种方法的计算成本是多少？对于实时推荐系统哪种组合最优？\n"
            "Output:\n"
            "### data preprocessing impact on machine learning model accuracy\n"
            "### computational cost of data preprocessing methods\n"
            "### optimal preprocessing combinations for real-time recommendation systems\n\n"
            "Example 2 (Simple single-focus question -> single information item):\n"
            "Question: AffectGPT 骨干网络的参数量是多少？\n"
            "Output:\n"
            "End | AffectGPT backbone network parameter size\n\n"
            "Example 3 (Missing information gap without references -> list of required information items):\n"
            "Question: 仍缺少 AffectGPT 在 MER-UniBench 各任务上的具体评测表现，以及训练数据的配比，无法确认哪些任务最优。\n"
            "Output:\n"
            "### AffectGPT performance on MER-UniBench\n"
            "### AffectGPT training data composition\n\n"
            "Example 4 (Missing information gap with references -> essential information items):\n"
            "Question: 仍缺少核心对比指标（如 Table 4 的完整数据）以及附录中的实验细节（如 Appendix B），因此无法完整回答模型表现。\n"
            "Output:\n"
            "### Table 4\n"
            "### Appendix B\n"
            "=== End Examples ===\n\n"
            f"Input question:\n{question}"
        )
        expanded_msg = llm.invoke(prompt_text)
        expanded = str(getattr(expanded_msg, "content", expanded_msg)).strip()
        retrieval_tokens = _extract_response_tokens(expanded_msg, prompt_text, expanded)
        sub_queries = _parse_decomposed_queries(expanded, question)
        base_query = sub_queries[0] if len(sub_queries) == 1 else question
    else:
        if config.translate_query and not _is_english_query(question):
            base_query = english_query(question, llm, translation_cache)
            retrieval_tokens = _extract_response_tokens(None, question, base_query)
        else:
            base_query = question
        sub_queries = [base_query]

    # 2. Individual Retrieval & Dedicated Cross-Encoder Reranking per Sub-question (NO RRF Mixing)
    sub_groups = []
    for q in sub_queries:
        if config.translate_query and not _is_english_query(q):
            continue
        # Retrieve candidate pool for this specific sub-question
        cand_ids = search_candidates(q, snapshot, config)
        
        # Filter pure bibliography and table of contents noise
        is_asking_bib = any(w in q.lower() for w in ["reference", "bibliography", "author", "cite", "citation", "biblio", "谁写", "作者", "参考文献"])
        if not is_asking_bib:
            cand_ids = [
                i for i in cand_ids
                if str(snapshot.metadatas[i].get("section", "")).strip().upper() not in ("REFERENCES", "BIBLIOGRAPHY", "REFERENCE", "参考文献")
                and not is_pure_toc_chunk(snapshot.chunks[i])
            ]
        else:
            cand_ids = [i for i in cand_ids if not is_pure_toc_chunk(snapshot.chunks[i])]
        cand_ids = cand_ids[:config.candidate_k]

        # Dedicated Cross-encoder reranking specifically against this sub-question q
        if config.rerank and cand_ids:
            texts = [snapshot.chunks[i] for i in cand_ids]
            top_n = min(config.top_k, len(cand_ids))
            order = rerank_indices(q, texts, top_n=top_n)
            ranked_ids = [cand_ids[idx] for idx in order]
        else:
            ranked_ids = cand_ids[:config.top_k]

        # Build dedicated document group for this sub-question
        q_docs = [
            {
                "text": snapshot.chunks[i],
                **snapshot.metadatas[i],
                "index_id": i,
                "chunk_id": snapshot.chunk_ids[i],
                "sub_query": q,
            }
            for i in ranked_ids
        ]
        sub_groups.append({
            "sub_query": q,
            "docs": q_docs
        })

    # 3. Preserve ALL sub-question documents without loss (Deduplicated union)
    merged_sources = []
    seen_ids = set()
    for grp in sub_groups:
        for doc in grp["docs"]:
            cid = doc.get("chunk_id")
            if cid:
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged_sources.append(doc)
            else:
                merged_sources.append(doc)

    return {
        "sources": merged_sources,
        "sub_groups": sub_groups,
        "queries": [g["sub_query"] for g in sub_groups],
        "retrieval_query": base_query,
        "snapshot_id": snapshot.fingerprint,
        "token_usage": retrieval_tokens,
    }
