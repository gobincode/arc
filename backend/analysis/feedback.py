"""
Rule-based feedback engine — no LLM.
Aggregates per-frame angles across shot phases and generates structured feedback.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

from .angles import ShotAngles, compute_angles
from .baseline import check_against_baseline, FormIssue
from .shot_detector import DetectedShot, Phase
from .pose import FramePose


ANGLE_FIELDS = [
    "draw_elbow_angle",
    "bow_elbow_angle",
    "shoulder_tilt",
    "draw_shoulder_elevation",
    "bow_shoulder_depression",
    "head_tilt",
    "head_forward_lean",
    "stance_width_ratio",
    "draw_wrist_to_bow_shoulder",
]


@dataclass
class ShotFeedback:
    shot_id: int
    full_draw_angles: Optional[dict]
    issues: list[FormIssue]
    consistency_score: float        # 0-100, higher is better
    consistency_notes: list[str]
    summary: str


@dataclass
class ComparisonFeedback:
    reference_shot_id: int
    compare_shot_id: int
    deltas: dict[str, float]        # field -> delta value
    changed_fields: list[dict]      # list of {field, label, ref_val, cmp_val, delta, note}
    summary: str


def _mean_angles(angles_list: list[ShotAngles]) -> dict[str, Optional[float]]:
    result = {}
    for field in ANGLE_FIELDS:
        vals = [getattr(a, field) for a in angles_list if getattr(a, field) is not None]
        result[field] = float(np.mean(vals)) if vals else None
    return result


def _std_angles(angles_list: list[ShotAngles]) -> dict[str, Optional[float]]:
    result = {}
    for field in ANGLE_FIELDS:
        vals = [getattr(a, field) for a in angles_list if getattr(a, field) is not None]
        result[field] = float(np.std(vals)) if len(vals) >= 2 else None
    return result


def analyze_shot(
    shot: DetectedShot,
    poses: list[FramePose],
    draw_side: str = "right",
) -> ShotFeedback:
    """Analyze a single detected shot and return form feedback."""

    # Compute angles for full draw phase frames
    full_draw_phase = next((p for p in shot.phases if p.phase == Phase.FULL_DRAW), None)
    if full_draw_phase is None:
        return ShotFeedback(
            shot_id=shot.shot_id,
            full_draw_angles=None,
            issues=[],
            consistency_score=0.0,
            consistency_notes=["Could not detect full draw phase."],
            summary="Shot phase detection failed.",
        )

    # Get frames in full draw phase
    phase_poses = [
        p for p in poses
        if full_draw_phase.start_frame <= p.frame_idx <= full_draw_phase.end_frame
    ]
    if not phase_poses:
        phase_poses = [poses[full_draw_phase.key_frame]] if full_draw_phase.key_frame < len(poses) else []

    angles_list = [compute_angles(fp, draw_side) for fp in phase_poses]
    if not angles_list:
        return ShotFeedback(
            shot_id=shot.shot_id,
            full_draw_angles=None,
            issues=[],
            consistency_score=0.0,
            consistency_notes=["No valid pose data in full draw phase."],
            summary="Insufficient pose data.",
        )

    mean_a = _mean_angles(angles_list)
    std_a = _std_angles(angles_list)

    # Check against baseline
    issues = check_against_baseline(mean_a)

    # Consistency: measure std across frames in full draw (should be stable)
    consistency_notes = []
    instability_count = 0
    INSTABILITY_THRESHOLDS = {
        "draw_elbow_angle": 5.0,
        "bow_elbow_angle": 5.0,
        "shoulder_tilt": 3.0,
        "head_tilt": 3.0,
        "draw_wrist_to_bow_shoulder": 0.02,
    }
    for field, threshold in INSTABILITY_THRESHOLDS.items():
        std = std_a.get(field)
        if std is not None and std > threshold:
            instability_count += 1
            consistency_notes.append(
                f"{field.replace('_', ' ').title()} is unstable during full draw (std={std:.2f})."
            )

    max_unstable = len(INSTABILITY_THRESHOLDS)
    consistency_score = max(0.0, 100.0 * (1 - instability_count / max_unstable))

    sig_count = sum(1 for i in issues if i.severity == "significant")
    mod_count = sum(1 for i in issues if i.severity == "moderate")
    if sig_count == 0 and mod_count == 0:
        summary = "Good shot. Minor adjustments may help refine form."
    elif sig_count > 0:
        top = issues[0]
        summary = f"Significant issue detected: {top.label}. {top.feedback}"
    else:
        top = issues[0]
        summary = f"Moderate issue: {top.label}. {top.feedback}"

    return ShotFeedback(
        shot_id=shot.shot_id,
        full_draw_angles=mean_a,
        issues=issues,
        consistency_score=round(consistency_score, 1),
        consistency_notes=consistency_notes,
        summary=summary,
    )


def compare_shots(
    ref_feedback: ShotFeedback,
    cmp_feedback: ShotFeedback,
) -> ComparisonFeedback:
    """Compare two shots and highlight what changed."""
    if not ref_feedback.full_draw_angles or not cmp_feedback.full_draw_angles:
        return ComparisonFeedback(
            reference_shot_id=ref_feedback.shot_id,
            compare_shot_id=cmp_feedback.shot_id,
            deltas={},
            changed_fields=[],
            summary="Could not compare — angle data missing for one or both shots.",
        )

    deltas = {}
    changed_fields = []

    CHANGE_THRESHOLDS = {
        "draw_elbow_angle": 5.0,
        "bow_elbow_angle": 5.0,
        "shoulder_tilt": 3.0,
        "draw_shoulder_elevation": 4.0,
        "bow_shoulder_depression": 4.0,
        "head_tilt": 3.0,
        "head_forward_lean": 2.0,
        "stance_width_ratio": 0.15,
    }

    FIELD_LABELS = {
        "draw_elbow_angle": "Draw elbow angle",
        "bow_elbow_angle": "Bow arm elbow angle",
        "shoulder_tilt": "Shoulder level",
        "draw_shoulder_elevation": "Draw shoulder elevation",
        "bow_shoulder_depression": "Bow shoulder position",
        "head_tilt": "Head tilt",
        "head_forward_lean": "Head forward lean",
        "stance_width_ratio": "Stance width",
    }

    for field, threshold in CHANGE_THRESHOLDS.items():
        ref_val = ref_feedback.full_draw_angles.get(field)
        cmp_val = cmp_feedback.full_draw_angles.get(field)
        if ref_val is None or cmp_val is None:
            continue
        delta = cmp_val - ref_val
        deltas[field] = round(delta, 2)
        if abs(delta) >= threshold:
            direction = "increased" if delta > 0 else "decreased"
            changed_fields.append({
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "ref_val": round(ref_val, 2),
                "cmp_val": round(cmp_val, 2),
                "delta": round(delta, 2),
                "note": f"{FIELD_LABELS.get(field, field)} {direction} by {abs(delta):.1f}° compared to reference shot.",
            })

    if not changed_fields:
        summary = "Shots are consistent — no significant changes detected."
    else:
        top = changed_fields[0]
        summary = f"{len(changed_fields)} change(s) detected. Main change: {top['note']}"

    return ComparisonFeedback(
        reference_shot_id=ref_feedback.shot_id,
        compare_shot_id=cmp_feedback.shot_id,
        deltas=deltas,
        changed_fields=changed_fields,
        summary=summary,
    )
