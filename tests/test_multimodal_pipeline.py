"""
End-to-end verification script for Multimodal Index, Retrieval and DeepSeek Assistant.
"""

import sys
import os
sys.path.insert(0, ".")
sys.path.insert(0, r"D:\study\codex\Enterprise-Agentic-RAG-Platform-main\.venv\Lib\site-packages")

from loguru import logger
from app.pipeline_config import PipelineConfig, default_variants
from vectorstore.store import load_snapshot, search_candidates, hybrid_search
from app.retrieval import retrieve_documents
from app.agent import ask


def test_snapshots():
    logger.info("=== 1. Testing Multimodal and Text Snapshots ===")
    snap_mm = load_snapshot(multimodal=True)
    snap_txt = load_snapshot(multimodal=False)
    logger.success(f"Multimodal snapshot fingerprint: {snap_mm.fingerprint[:16]}, chunks: {len(snap_mm.chunks)}")
    logger.success(f"Text baseline snapshot fingerprint: {snap_txt.fingerprint[:16]}, chunks: {len(snap_txt.chunks)}")
    assert len(snap_mm.chunks) == len(snap_txt.chunks)
    assert snap_mm.multimodal is True
    assert snap_txt.multimodal is False


def test_retrieval_comparison():
    logger.info("=== 2. Testing Multimodal vs Text Retrieval Comparison ===")
    query = "Structured SSMs channel mapping architecture figure"
    
    # Multimodal config
    cfg_mm = PipelineConfig(multimodal=True, translate_query=False, multi_query=False, top_k=5)
    res_mm = retrieve_documents(query, config=cfg_mm)
    logger.info(f"Multimodal results count: {len(res_mm['sources'])}")
    has_image_mm = any(s.get("type") == "image" for s in res_mm["sources"])
    logger.info(f"Multimodal retrieved image chunk? {has_image_mm}")
    for idx, s in enumerate(res_mm["sources"]):
        logger.info(f"  [{idx+1}] type={s.get('type')} source={s.get('source')} page={s.get('page')}")

    # Pure text config
    cfg_txt = PipelineConfig(multimodal=False, translate_query=False, multi_query=False, top_k=5)
    res_txt = retrieve_documents(query, config=cfg_txt)
    logger.info(f"Text-baseline results count: {len(res_txt['sources'])}")
    for idx, s in enumerate(res_txt["sources"]):
        logger.info(f"  [{idx+1}] type={s.get('type')} source={s.get('source')} page={s.get('page')}")


def test_variants_presets():
    logger.info("=== 3. Testing Ablation Variants Presets ===")
    presets = default_variants()
    names = [p["name"] for p in presets]
    logger.info(f"Default variants: {names}")
    assert any("多模态" in n for n in names)
    assert any("纯文本" in n for n in names)


def test_agent_qa_with_images():
    logger.info("=== 4. Testing Agent Ask with Image Context ===")
    query = "请解释一下论文中关于 Structured SSMs 架构示意图（Figure 1）的内容。"
    cfg = PipelineConfig(multimodal=True, translate_query=False, multi_query=False, top_k=3)
    res = ask(query, config=cfg)
    logger.info(f"Answer generated ({len(res['answer'])} chars): {res['answer'][:150]}...")
    logger.info(f"Sources count: {len(res['sources'])}")
    for s in res["sources"]:
        logger.info(f"Source: {s.get('source')} Page {s.get('page')} Type {s.get('type')} HasAsset: {bool(s.get('asset_images'))}")


if __name__ == "__main__":
    test_snapshots()
    test_retrieval_comparison()
    test_variants_presets()
    test_agent_qa_with_images()
    logger.success("ALL MULTIMODAL PIPELINE TESTS PASSED!")
