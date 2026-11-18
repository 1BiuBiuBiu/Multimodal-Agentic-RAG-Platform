"""
evaluation/evaluate.py
──────────────────────
Runs RAGAs evaluation on your RAG pipeline and saves scores to
evaluation/latest_scores.json (served by GET /metrics).

Usage:
    python -m evaluation.evaluate

What it measures
────────────────
  faithfulness      — does the answer stick to the retrieved context?
  answer_relevancy  — is the answer relevant to the question?
  context_precision — are the retrieved chunks actually useful?
  context_recall    — are all necessary facts retrieved?

Add your own test questions to EVAL_DATASET below.
"""

from __future__ import annotations
import json
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from loguru import logger
from ragas import evaluate
from evaluation.judge import DeepSeekJudge
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.agent import ask
from app.llm import create_llm
from embeddings.embedder import EMBED_MODEL
from evaluation.eval_data import EVAL_DATASET

load_dotenv()

SCORES_FILE = Path("evaluation/latest_scores.json")
SCORES_FILE.parent.mkdir(exist_ok=True)

# ── Use DeepSeek as the RAGAs judge LLM ─────────────────────────────────
RAGAS_LLM = DeepSeekJudge(create_llm(temperature=0))

# Apply the LLM to metrics that need a judge
for _metric in (faithfulness, answer_relevancy, context_precision, context_recall):
    _metric.llm = RAGAS_LLM  # type: ignore[attr-defined]

# ── Replace/extend with questions about YOUR actual documents ──────────────
# ground_truth: the ideal answer (used for context_recall only)
# EVAL_DATASET = [
#     {
#         "question":     "What is the main topic of the document?",
#         "ground_truth": "The document covers the primary subject matter in detail.",
#     },
#     {
#         "question":     "What are the key findings or conclusions?",
#         "ground_truth": "The key findings are summarised in the conclusion section.",
#     },
#     {
#         "question":     "What methodology was used?",
#         "ground_truth": "The methodology section describes the approach taken.",
#     },
#     {
#         "question":     "Who are the main authors or contributors?",
#         "ground_truth": "The authors are listed on the title page.",
#     },
#     {
#         "question":     "What recommendations are made?",
#         "ground_truth": "Recommendations are provided in the final section.",
#     },
# ]


from concurrent.futures import ThreadPoolExecutor, as_completed

def run_evaluation(concurrency: int = 4) -> dict:
    logger.info(f"Running RAGAs evaluation on {len(EVAL_DATASET)} questions (concurrency={concurrency}) …")

    ordered_results = [None] * len(EVAL_DATASET)

    def _eval_one(idx_item):
        idx, item = idx_item
        q = item["question"]
        gt = item["ground_truth"]
        result = ask(q)
        answer = result["answer"]
        ctx_texts = [c["text"] for c in result["sources"]]
        trace = result.get("retrieval_trace", [])
        retrieval_count = len(trace) if trace else (result.get("rewrite_count", 0) + 1)
        return idx, {
            "question": q,
            "answer": answer,
            "contexts": ctx_texts if ctx_texts else ["No context retrieved."],
            "ground_truth": gt,
            "retrieval_count": retrieval_count,
        }

    max_workers = min(concurrency, len(EVAL_DATASET)) if EVAL_DATASET else 1
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="eval-runner") as executor:
        futures = [executor.submit(_eval_one, (i, item)) for i, item in enumerate(EVAL_DATASET)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            ordered_results[idx] = res

    questions = [r["question"] for r in ordered_results]
    answers = [r["answer"] for r in ordered_results]
    contexts = [r["contexts"] for r in ordered_results]
    ground_truths = [r["ground_truth"] for r in ordered_results]
    retrieval_counts = [r["retrieval_count"] for r in ordered_results]
    avg_retrieval_count = round(sum(retrieval_counts) / len(retrieval_counts), 2) if retrieval_counts else 1.0

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # Explicit local embeddings prevent RAGAs from defaulting to OpenAI embeddings.
    eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    ))
    result = evaluate(
        dataset,
        llm=RAGAS_LLM,
        embeddings=eval_embeddings,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # RAGAs 0.2 returns EvaluationResult; aggregate per-question metric scores.
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    means = result.to_pandas()[metric_names].mean()
    scores = {k: round(float(v), 4) for k, v in means.items()}
    scores["avg_retrieval_count"] = avg_retrieval_count
    SCORES_FILE.write_text(json.dumps(scores, indent=2))
    logger.success(f"Scores saved to {SCORES_FILE}")

    print("\n── RAGAs Evaluation Results ─────────────────────")
    for metric, score in scores.items():
        if metric == "avg_retrieval_count":
            print(f"  {metric:<25} {score:.2f} 次 (平均检索次数)")
        else:
            bar = "█" * int(score * 20)
            print(f"  {metric:<25} {score:.4f}  {bar}")
    print("─────────────────────────────────────────────────\n")

    return scores


if __name__ == "__main__":
    run_evaluation()
