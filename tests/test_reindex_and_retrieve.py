import sys
sys.path.insert(0, ".")
sys.path.insert(0, r"D:\study\codex\Enterprise-Agentic-RAG-Platform-main\.venv\Lib\site-packages")

import pickle
from vectorstore.store import build_index, META_PATH
from app.pipeline_config import PipelineConfig
from app.retrieval import retrieve_documents
from app.llm import create_llm

with open(META_PATH, "rb") as f:
    data = pickle.load(f)

chunks = list(data["chunks"])
metas = list(data["metadatas"])

print(f"Total chunks: {len(chunks)}")
updated_images = 0
for i, m in enumerate(metas):
    if m.get("type") == "image":
        section = m.get("section", "")
        # Check if section prefix already added
        if section and not chunks[i].startswith("所属章节："):
            chunks[i] = f"所属章节：{section}\n\n{chunks[i]}"
            updated_images += 1

print(f"Updated {updated_images} image chunks with section context.")
print("Rebuilding FAISS indexes (multimodal + text) and BM25 ...")
build_index(chunks, metas)
print("Index rebuild complete!")

# Now test retrieval for the exact user question
query = "affectgpt的模型架构图有哪些内容"
config = PipelineConfig(top_k=5, candidate_k=20, multimodal=True)
llm = create_llm(temperature=0)
res = retrieve_documents(query, config, llm=llm)

print("\n--- Retrieval Results for: affectgpt的模型架构图有哪些内容 ---")
for idx, s in enumerate(res["sources"]):
    print(f"[{idx+1}] type={s.get('type')} source={s.get('source')} page={s.get('page')}")
    print(f"    text: {s.get('text')[:120]}...")
