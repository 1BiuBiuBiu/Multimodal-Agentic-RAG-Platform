"""Per-request immutable switches shared by chat and ablation experiments."""
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    multimodal: bool = True
    translate_query: bool = True
    multi_query: bool = True
    dense: bool = True
    bm25: bool = True
    rerank: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)
    route_query: bool = True
    grade_documents: bool = True
    corrective_rag: bool = True
    eval_memory: bool = True
    web_search: bool = True
    chinese_answer: bool = True

    @model_validator(mode="before")
    @classmethod
    def remove_legacy_query_count(cls, value):
        # Previously saved variants may contain this removed setting. Preserve
        # the rest of their configuration without imposing the obsolete count.
        if isinstance(value, dict):
            clean = {key: item for key, item in value.items() if key != "num_queries"}
            clean.setdefault("multimodal", True)
            return clean
        return value

    @model_validator(mode="after")
    def valid_retrieval(self):
        if not self.dense and not self.bm25:
            raise ValueError("向量检索和 BM25 至少启用一项。")
        if self.candidate_k < self.top_k:
            raise ValueError("候选块数必须大于或等于返回块数。")
        return self


def default_variants():
    full = PipelineConfig()
    return [{"name": name, "config": config.model_dump()} for name, config in [
        ("多模态完整检索 (Visual-BGE)", full),
        ("纯文本向量基线 (BGE-M3)", full.model_copy(update={"multimodal": False})),
        ("关闭多查询", full.model_copy(update={"multi_query": False})),
        ("关闭翻译", full.model_copy(update={"translate_query": False})),
        ("关闭重排", full.model_copy(update={"rerank": False})),
        ("仅密集向量检索", full.model_copy(update={"bm25": False})),
        ("关闭评估记忆与早停", full.model_copy(update={"eval_memory": False})),
    ]]
