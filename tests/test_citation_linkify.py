import sys
sys.path.insert(0, ".")
sys.path.insert(0, r"D:\study\codex\Enterprise-Agentic-RAG-Platform-main\.venv\Lib\site-packages")
from app.main import linkify_citations

sources = [
    {"source": "2501.16566v2.pdf", "page": 2, "page_end": 2, "text": "Intro text snippet"},
    {"source": "2312.00752v2.pdf", "page": 3, "page_end": 4, "text": "SSMs text snippet"}
]

text = "根据报告，前言描述了该任务 [2501.16566v2.pdf, Page 2]，并在核心章节给出了架构对比 [2312.00752v2.pdf, Page 3]。"
res = linkify_citations(text, sources, "msg-0")
print("Converted result:")
print(res)
assert 'href="#src-msg-0-0"' in res, "Missing anchor for source 0"
assert 'href="#src-msg-0-1"' in res, "Missing anchor for source 1"
print("SUCCESS: Citation linkify test passed!")
