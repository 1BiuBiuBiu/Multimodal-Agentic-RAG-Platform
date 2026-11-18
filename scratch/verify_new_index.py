from vectorstore.store import load_snapshot
from ingestion.chunking import is_pure_toc_chunk

idx = load_snapshot(multimodal=True)
print(f"Total chunks in new index: {len(idx.chunks)}")

# Check for tiny chunks (< 100 chars)
tiny_chunks = []
toc_chunks = []
for i, c in enumerate(idx.chunks):
    if len(c.strip()) < 100:
        tiny_chunks.append((idx.chunk_ids[i], c))
    if is_pure_toc_chunk(c):
        toc_chunks.append((idx.chunk_ids[i], c))

print(f"Tiny chunks (< 100 chars): {len(tiny_chunks)}")
for cid, c in tiny_chunks:
    print(f"  Tiny chunk {cid}: '{c}'")

print(f"TOC chunks: {len(toc_chunks)}")
for cid, c in toc_chunks:
    print(f"  TOC chunk {cid}: '{c[:60]}...'")

# Specifically check if the old chunk 88 (C Details of EmoPrefer-Data 17) exists
found_old_toc = [c for c in idx.chunks if "C Details of EmoPrefer-Data 17" in c]
print(f"Found old TOC text in index: {len(found_old_toc)}")
