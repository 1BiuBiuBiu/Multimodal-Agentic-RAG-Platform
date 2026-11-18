import os
import sys
from app.llm import create_llm

llm = create_llm(temperature=0)
prompt_text = (
    "You are an expert retrieval planner for scientific and technical multimodal research papers.\n"
    "Your task is to analyze the user question and decide whether it needs decomposition into orthogonal, highly focused atomic English sub-queries.\n\n"
    "Guidelines:\n"
    "1. For a single-focus, atomic question that does NOT benefit from decomposition, return ONLY the original question on a single line.\n"
    "2. For a compound question containing multiple distinct aspects (such as model architecture, training datasets/data volume, training methodology, benchmark metrics, ablation study), "
    "decompose it into distinct, search-friendly atomic English queries—one per aspect.\n"
    "3. NEVER repeat the whole compound question as one of the sub-queries! Each sub-query must target EXACTLY ONE specific aspect.\n"
    "4. If an aspect touches on model architecture or system workflow, include terms targeting framework figures, pipeline diagrams, and module illustrations.\n"
    "5. Write every sub-query in English only. Exactly one query per line, without numbering, bullets, or introductory text.\n\n"
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
    "Question: Introduce the core architecture of EMOPREFER, the training datasets and data volume, and the training method."
)

res = llm.invoke(prompt_text)
print("Result with English input:")
print(res.content)

prompt_text_zh = prompt_text.replace(
    "Question: Introduce the core architecture of EMOPREFER, the training datasets and data volume, and the training method.",
    "Question: 介绍一下EMOPREFER的核心架构，训练的数据集以及数据量，以及训练方式"
)
res_zh = llm.invoke(prompt_text_zh)
print("\nResult with Chinese input:")
print(res_zh.content)
