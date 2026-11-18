from app.retrieval import retrieve_documents
from app.pipeline_config import PipelineConfig

config = PipelineConfig(multimodal=True, multi_query=True, rerank=True)
query = "介绍一下EMOPREFER的核心架构，训练的数据集以及数据量，以及训练方式"

res = retrieve_documents(query, config=config)
print("Queries generated:", res.get("queries"))
print("Sub-groups count:", len(res.get("sub_groups", [])))
for idx, grp in enumerate(res.get("sub_groups", [])):
    print(f"\nSub-group {idx+1}: {grp.get('sub_query')}")
    docs = grp.get("docs", [])
    print(f"  Docs count: {len(docs)}")
    for d in docs[:3]:
        print(f"    - [{d.get('chunk_id')}] (p.{d.get('page')}) {d.get('text')[:80]}...")
