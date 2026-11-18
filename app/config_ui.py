"""Shared widget controls for chat and experiment variants."""
import streamlit as st
from app.pipeline_config import PipelineConfig

LABELS = {
    "multimodal": "多模态向量检索 (Visual-BGE)",
    "translate_query": "英文查询翻译",
    "multi_query": "多查询分解（MultiQueryRetriever）",
    "dense": "向量检索",
    "bm25": "BM25 关键词检索",
    "rerank": "Cross-encoder 重排",
    "route_query": "问题路由",
    "grade_documents": "文档相关性判断",
    "corrective_rag": "失败后改写重试",
    "eval_memory": "评估记忆池与早停熔断",
    "web_search": "网页回退",
    "chinese_answer": "中文回答",
}


def config_controls(prefix, *, full=True):
    default = PipelineConfig()
    values = {}
    for name, label in LABELS.items():
        if not full and name in ("route_query", "grade_documents", "corrective_rag", "eval_memory", "web_search"):
            values[name] = getattr(default, name)
        else:
            values[name] = st.checkbox(label, value=getattr(default, name), key=f"{prefix}_{name}")
    if values.get("eval_memory"):
        st.caption("评估记忆与早停已启用：记忆已过滤的无关文档避免重复打分，且在增量文档为空时及时熔断早停。")
    if values.get("multimodal"):
        st.caption("多模态向量检索已启用：图片块融合视觉特征与 caption 进行 Visual-BGE 跨模态检索。")
    else:
        st.caption("多模态已关闭：使用 BGE-M3 纯文本基线（仅利用 caption 纯文本进行检索，用于消融对比）。")
    if not values["translate_query"]:
        st.caption("翻译已关闭：中文查询会直接用于检索和英文 cross-encoder，可用于语言对齐消融。")
    if full and not values["grade_documents"]:
        st.caption("相关性判断已关闭，后续改写重试和网页回退不会触发。")
    if values["multi_query"]:
        st.caption("模型自行判断是否分解及查询数量；简单问题可以直接使用原查询。")
    values["top_k"] = st.slider("最终返回块数 top_k", 1, 20, 5, key=f"{prefix}_top_k")
    values["candidate_k"] = st.slider("融合候选块数 candidate_k", 20, 100, 20, step=5, key=f"{prefix}_candidate_k")
    return values
