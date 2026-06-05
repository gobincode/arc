import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse
from typing import Optional

from ..workers.tasks import process_video_task, task_results

router = APIRouter()

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/archery/uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/archery/outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@router.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    draw_side: str = Form(default="right"),
):
    """Upload a video and start analysis. Returns a job_id to poll for results."""
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    job_id = str(uuid.uuid4())
    ext = os.path.splitext(video.filename)[-1] or ".mp4"
    save_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    out_dir = os.path.join(OUTPUT_DIR, job_id)
    background_tasks.add_task(
        process_video_task, job_id, save_path, out_dir, draw_side
    )

    return {"job_id": job_id, "status": "processing"}


@router.get("/results/{job_id}")
async def get_results(job_id: str):
    """Poll for analysis results by job_id."""
    result = task_results.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found or still processing.")
    return result


@router.get("/video/{job_id}")
async def get_annotated_video(job_id: str):
    """Download the annotated output video."""
    result = task_results.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    video_path = result.get("annotated_video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Annotated video not available.")
    return FileResponse(video_path, media_type="video/mp4", filename="annotated.mp4")
