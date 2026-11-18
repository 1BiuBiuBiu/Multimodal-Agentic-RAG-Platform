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


def retrieve_documents(question, config=None, *, llm=None, snapshot=None, translation_cache=None, sub_queries=None):
    config = config or PipelineConfig()
    snapshot = snapshot or load_snapshot(multimodal=config.multimodal)
    llm = llm or create_llm(temperature=0)
    base_query = english_query(question, llm, translation_cache) if config.translate_query else question

    # 1. Sub-question planning (If caller already provided targeted sub-queries, use them directly!)
    if sub_queries:
        # Direct bypass from Stage 2 Targeted Sub-queries
        sub_queries = [q.strip() for q in sub_queries if q and q.strip()]
    elif config.multi_query:
        language = "English only" if config.translate_query else "the same language as the input question"
        prompt_text = (
            "You are an expert retrieval planner for scientific and technical multimodal research papers.\n"
            "Your task is to analyze the user question and decide whether it needs decomposition into orthogonal, highly focused atomic English sub-queries.\n\n"
            "Guidelines:\n"
            "1. For a single-focus, atomic question that does NOT benefit from decomposition, return ONLY the original question on a single line.\n"
            "2. For a compound question containing multiple distinct aspects (such as model architecture, training datasets/data volume, training methodology, benchmark metrics, ablation study), "
            "decompose it into distinct, search-friendly atomic English queries—one per aspect.\n"
            "3. NEVER repeat the whole compound question as one of the sub-queries! Each sub-query must target EXACTLY ONE specific aspect.\n"
            "4. If an aspect touches on model architecture or system workflow, include terms targeting framework figures, pipeline diagrams, and module illustrations.\n"
            f"5. Write every sub-query in {language}. Exactly one query per line, without numbering, bullets, or introductory text.\n\n"
            "=== Examples ===\n"
            "Example 1 (Compound question with multiple technical aspects):\n"
            "Question: 介绍一下EMOPREFER的核心架构，训练的数据集以及数据量，以及训练方式\n"
            "Output:\n"
            "EMOPREFER model core architecture, framework pipeline diagram, and module components\n"
            "EMOPREFER training datasets, corpus sources, and data sample volume statistics\n"
            "EMOPREFER training methodology, loss functions, optimization objectives, and preference alignment\n\n"
            "Example 2 (Compound question inquiring about tasks and benchmark scores):\n"
            "Question: AffectGPT在哪些任务上表现最优？对比基线有哪些具体跑分？\n"
            "Output:\n"
            "AffectGPT evaluation benchmarks, task categories, and overall performance comparison\n"
            "AffectGPT main results tables, Table 2 benchmark scores, and baseline model comparison metrics\n\n"
            "Example 3 (Simple, focused single-aspect question - NO decomposition needed):\n"
            "Question: What is the parameter size of the AffectGPT backbone?\n"
            "Output:\n"
            "What is the parameter size of the AffectGPT backbone?\n"
            "=== End Examples ===\n\n"
            f"Question: {base_query}"
        )
        expanded = str(llm.invoke(prompt_text).content).strip()
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in expanded.splitlines() if line.strip()]
        
        # Deduplicate while preserving order
        unique_lines = []
        for l in lines:
            if l and l not in unique_lines:
                unique_lines.append(l)

        # If LLM decomposed into distinct atomic sub-queries, use them directly (do NOT prepend compound base_query)!
        if unique_lines:
            sub_queries = unique_lines
        else:
            sub_queries = [base_query]
    else:
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
    }
