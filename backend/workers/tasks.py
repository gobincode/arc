"""
Background task worker (in-process for simplicity; swap for Celery when scaling).
"""
import traceback
from dataclasses import asdict
from ..analysis.pipeline import run_pipeline

# In-memory store; replace with Redis/DB for production
task_results: dict[str, dict] = {}


def process_video_task(
    job_id: str,
    video_path: str,
    output_dir: str,
    draw_side: str = "right",
):
    task_results[job_id] = {"status": "processing"}
    try:
        result = run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            draw_side=draw_side,
        )
        payload = {
            "status": "done",
            "job_id": job_id,
            "video_info": result.video_info,
            "shot_count": result.shot_count,
            "annotated_video_path": result.annotated_video_path,
            "shots": [],
            "comparison": None,
            "error": result.error,
        }
        for sf in result.shots:
            shot_dict = {
                "shot_id": sf.shot_id,
                "consistency_score": sf.consistency_score,
                "consistency_notes": sf.consistency_notes,
                "summary": sf.summary,
                "full_draw_angles": sf.full_draw_angles,
                "issues": [
                    {
                        "field": i.field,
                        "label": i.label,
                        "severity": i.severity,
                        "measured": i.measured,
                        "ideal": i.ideal,
                        "deviation": i.deviation,
                        "feedback": i.feedback,
                    }
                    for i in sf.issues
                ],
            }
            payload["shots"].append(shot_dict)

        if result.comparison:
            c = result.comparison
            payload["comparison"] = {
                "reference_shot_id": c.reference_shot_id,
                "compare_shot_id": c.compare_shot_id,
                "deltas": c.deltas,
                "changed_fields": c.changed_fields,
                "summary": c.summary,
            }

        task_results[job_id] = payload

    except Exception as e:
        task_results[job_id] = {
            "status": "error",
            "job_id": job_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
