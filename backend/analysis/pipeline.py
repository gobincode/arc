"""
Full analysis pipeline: video in -> feedback + annotated video out.
"""
import os
from dataclasses import dataclass, asdict
from typing import Optional

from .pose import extract_poses, get_video_info
from .angles import compute_angles
from .shot_detector import detect_shots, Phase
from .feedback import analyze_shot, compare_shots, ShotFeedback, ComparisonFeedback
from .overlay import annotate_video


@dataclass
class PipelineResult:
    video_info: dict
    shot_count: int
    shots: list[ShotFeedback]
    comparison: Optional[ComparisonFeedback]
    annotated_video_path: Optional[str]
    error: Optional[str] = None


def run_pipeline(
    video_path: str,
    output_dir: str,
    draw_side: str = "right",
    reference_shot_id: Optional[int] = None,
    compare_shot_id: Optional[int] = None,
) -> PipelineResult:
    """
    Run full archery analysis pipeline on a video file.

    Args:
        video_path: Path to input video.
        output_dir: Directory to write annotated video.
        draw_side: 'right' or 'left'.
        reference_shot_id: Shot index to use as reference for comparison.
        compare_shot_id: Shot index to compare against reference.
    """
    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(video_path)

    # Step 1: Extract poses
    poses = extract_poses(video_path, skip_frames=1)
    if not poses:
        return PipelineResult(
            video_info=info, shot_count=0, shots=[], comparison=None,
            annotated_video_path=None, error="No pose data detected in video."
        )

    # Step 2: Detect shots
    shots = detect_shots(poses, draw_side=draw_side)
    if not shots:
        # Fall back: treat whole video as one shot
        from .shot_detector import DetectedShot, ShotPhase
        mid = len(poses) // 2
        fallback_shot = DetectedShot(shot_id=0, phases=[
            ShotPhase(Phase.FULL_DRAW, 0, len(poses) - 1, mid)
        ])
        shots = [fallback_shot]

    # Step 3: Analyze each shot
    shot_feedbacks = [analyze_shot(s, poses, draw_side) for s in shots]

    # Step 4: Comparison
    comparison = None
    if len(shot_feedbacks) >= 2:
        ref_id = reference_shot_id if reference_shot_id is not None else 0
        cmp_id = compare_shot_id if compare_shot_id is not None else 1
        ref_id = min(ref_id, len(shot_feedbacks) - 1)
        cmp_id = min(cmp_id, len(shot_feedbacks) - 1)
        if ref_id != cmp_id:
            comparison = compare_shots(shot_feedbacks[ref_id], shot_feedbacks[cmp_id])

    # Step 5: Build angle and phase maps for overlay
    angles_map = {fp.frame_idx: compute_angles(fp, draw_side) for fp in poses}
    phase_map: dict[int, str] = {}
    for shot in shots:
        for phase in shot.phases:
            for fi in range(phase.start_frame, phase.end_frame + 1):
                phase_map[fi] = phase.phase.value

    # Step 6: Annotate video
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{base_name}_annotated.mp4")
    try:
        annotate_video(video_path, out_video, poses, angles_map, phase_map, draw_side)
    except Exception as e:
        out_video = None

    return PipelineResult(
        video_info=info,
        shot_count=len(shots),
        shots=shot_feedbacks,
        comparison=comparison,
        annotated_video_path=out_video,
    )
