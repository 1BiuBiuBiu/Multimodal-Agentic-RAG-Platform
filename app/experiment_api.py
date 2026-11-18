"""Experiment configuration and asynchronous evaluation job API."""
import json
from fastapi import APIRouter, HTTPException
from app.pipeline_config import default_variants
from evaluation import experiments as runner
from evaluation.eval_data import EVAL_DATASET

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get("/variants")
def get_variants():
    variants = json.loads(runner.VARIANTS_FILE.read_text(encoding="utf-8")) if runner.VARIANTS_FILE.exists() else default_variants()
    return {"variants": [runner.Variant.model_validate(item).model_dump() for item in variants]}


@router.put("/variants")
def save_variants(variants: list[runner.Variant]):
    if not 1 <= len(variants) <= 10 or len({v.name for v in variants}) != len(variants):
        raise HTTPException(status_code=422, detail="请保存 1–10 个名称不同的变体。")
    with runner.LOCK:
        runner._write(runner.VARIANTS_FILE, [v.model_dump() for v in variants])
    return {"variants": [v.model_dump() for v in variants]}


@router.get("/dataset")
def get_dataset():
    return {"dataset": [runner.EvalQuestion.model_validate(row).model_dump() for row in EVAL_DATASET]}


@router.post("/jobs", status_code=202)
def start_job(request: runner.BatchRequest):
    try:
        return runner.start_batch(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/jobs")
def list_jobs():
    paths = sorted(runner.RUN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
    jobs = [runner.get_job(path.stem) for path in paths]
    return {"jobs": [{"id": job["id"], "status": job["status"], "created_at": job["created_at"]} for job in jobs]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return runner.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="评估任务不存在。")
