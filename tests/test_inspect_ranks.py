import sys
sys.path.insert(0, ".")
sys.path.insert(0, r"D:\study\codex\Enterprise-Agentic-RAG-Platform-main\.venv\Lib\site-packages")

import pickle
import numpy as np
import faiss
from embeddings.embedder import embed_query

with open("vectorstore/index/metadata.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
metas = data["metadatas"]

query_eng = "What does the model architecture diagram of affectgpt contain?"
q_vec = np.array([embed_query(query_eng)], dtype="float32")

txt_index = faiss.read_index("vectorstore/index/faiss_text.index")
_, idxs_t = txt_index.search(q_vec, 20)
print("Top 20 in pure-text index:")
for r, i in enumerate(idxs_t[0]):
    print(f"  Rank {r+1}: Chunk {i} (Type: {metas[i].get('type')}, ID: {metas[i].get('chunk_id')}, Source: {metas[i].get('source')}, Page: {metas[i].get('page')})")
