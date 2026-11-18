"""
embeddings/model_downloader.py
──────────────────────────────
Strict model verification and automatic downloader for multimodal RAG.
Guarantees both BGE-M3 (text) and Visualized-BGE (vision) models are present.
Fail-fast principle: Never silently fall back to text-only degradation.
"""

from __future__ import annotations
import os
import urllib.request
from pathlib import Path
from loguru import logger

HF_HOME = Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface" / "hub")
DEFAULT_VISUAL_WEIGHT_DIR = HF_HOME / "models--BAAI--bge-visualized"
PRIMARY_URL = "https://hf-mirror.com/BAAI/bge-visualized/resolve/main/Visualized_m3.pth"
BACKUP_URL = "https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth"


def verify_text_bge_model() -> Path:
    """Ensure BAAI/bge-m3 is present in the local huggingface hub cache."""
    target_dir = HF_HOME / "models--BAAI--bge-m3"
    snapshots_dir = target_dir / "snapshots"
    if snapshots_dir.is_dir():
        for snap in snapshots_dir.iterdir():
            if snap.is_dir():
                return snap
    if target_dir.is_dir():
        return target_dir
    raise FileNotFoundError(
        f"BAAI/bge-m3 model directory not found in {target_dir}. "
        "Please ensure BGE-M3 text model is cached."
    )


def _download_stream(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 2 * 1024 * 1024  # 2MB
        last_log = 0
        logger.info(f"Target size: {total / 1024 / 1024:.2f} MB. Streaming...")
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_log >= 100 * 1024 * 1024 or downloaded == total:
                pct = (downloaded / total * 100) if total else 0
                logger.info(f"Downloading Visualized_m3.pth: {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct:.1f}%)")
                last_log = downloaded
    if temp_path.stat().st_size < 100 * 1024 * 1024:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file size is suspiciously small.")
    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)


def ensure_visual_bge_weight() -> Path:
    """
    Ensure Visualized_m3.pth exists locally.
    If absent, warn loudly and automatically stream-download it.
    If download fails, raise RuntimeError immediately (no silent fallback).
    """
    custom_path = os.getenv("VISUAL_BGE_WEIGHT_PATH")
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    dest_file = DEFAULT_VISUAL_WEIGHT_DIR / "Visualized_m3.pth"
    candidates = [
        dest_file,
        Path("data/models/Visualized_m3.pth"),
        Path("Visualized_m3.pth"),
    ]
    for c in candidates:
        if c.is_file() and c.stat().st_size > 500 * 1024 * 1024:
            return c

    # Search in HF cache snapshots
    if DEFAULT_VISUAL_WEIGHT_DIR.is_dir():
        for f in DEFAULT_VISUAL_WEIGHT_DIR.rglob("Visualized_m3.pth"):
            if f.is_file() and f.stat().st_size > 500 * 1024 * 1024:
                return f

    logger.warning("=" * 60)
    logger.warning("【多模态模型警报】本地未检测到 Visualized-BGE 视觉权重 Visualized_m3.pth")
    logger.warning("正在自动从镜像源 (BAAI/bge-visualized) 高速下载权重 (~1.66 GB)...")
    logger.warning("=" * 60)

    for url in (PRIMARY_URL, BACKUP_URL):
        try:
            logger.info(f"Attempting download from: {url}")
            _download_stream(url, dest_file)
            logger.success(f"Visualized_m3.pth 下载完成: {dest_file} ({dest_file.stat().st_size / 1024 / 1024:.2f} MB)")
            return dest_file
        except Exception as exc:
            logger.warning(f"Download from {url} failed: {exc}")

    raise RuntimeError(
        "无法获取多模态模型权重 Visualized_m3.pth。\n"
        "系统严禁无模型静默降级！请检查网络连接或手动下载至: " + str(dest_file)
    )
