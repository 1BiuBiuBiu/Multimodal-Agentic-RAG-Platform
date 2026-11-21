"""Experiment configuration and asynchronous evaluation job API."""
import importlib
import json
from fastapi import APIRouter, HTTPException
from app.pipeline_config import default_variants
from evaluation import eval_data, experiments as runner

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get("/variants")
def get_variants():
    variants = json.loads(runner.VARIANTS_FILE.read_text(encoding="utf-8")) if runner.VARIANTS_FILE.exists() else default_variants()
    return {
        "variants": [runner.Variant.model_validate(item).model_dump() for item in variants],
        "last_request": runner.get_last_request_prefs(),
    }


@router.put("/variants")
def save_variants(variants: list[runner.Variant]):
    if not 1 <= len(variants) <= 10 or len({v.name for v in variants}) != len(variants):
        raise HTTPException(status_code=422, detail="请保存 1–10 个名称不同的变体。")
    with runner.LOCK:
        runner._write(runner.VARIANTS_FILE, [v.model_dump() for v in variants])
    return {"variants": [v.model_dump() for v in variants]}


@router.get("/dataset")
def get_dataset():
    reloaded = importlib.reload(eval_data)
    simple = [
        {**row, "question_type": row.get("question_type", "simple")}
        for row in getattr(reloaded, "EVAL_DATASET_SIMPLE", [])
    ]
    complex_qs = [
        {**row, "question_type": row.get("question_type", "complex")}
        for row in getattr(reloaded, "EVAL_DATASET_COMPLEX", [])
    ]
    raw_items = (simple + complex_qs) if (simple or complex_qs) else list(getattr(reloaded, "EVAL_DATASET", []))
    return {"dataset": [runner.EvalQuestion.model_validate(row).model_dump() for row in raw_items]}


@router.post("/jobs", status_code=202)
def start_job(request: runner.BatchRequest):
    try:
        return runner.start_batch(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/jobs/ragas", status_code=202)
def start_ragas_jobs(request: runner.RagasEvalRequest):
    try:
        return runner.start_ragas_evaluation(
            request.job_ids,
            request.ragas_concurrency,
            selected_metrics=request.selected_metrics,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="所选评估任务不存在。")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/jobs")
def list_jobs():
    paths = [
        p for p in sorted(runner.RUN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if p.stem.isalnum() and len(p.stem) == 32
    ][:20]
    jobs = [runner.get_job(path.stem) for path in paths]
    return {
        "jobs": [
            {
                "id": job["id"],
                "status": job["status"],
                "created_at": job["created_at"],
                "scope": (job.get("request") or {}).get("scope", "retrieval"),
                "question_count": len((job.get("request") or {}).get("dataset") or []),
                "variant_names": [
                    v.get("name", "")
                    for v in ((job.get("request") or {}).get("variants") or [])
                    if isinstance(v, dict)
                ],
                "ragas_scored": runner._is_job_ragas_scored(job),
            }
            for job in jobs
        ]
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return runner.get_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="评估任务不存在。")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return runner.cancel_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="评估任务不存在。")
