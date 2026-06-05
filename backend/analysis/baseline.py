"""
Recurve archery ideal form baseline (World Archery / KSL shot cycle standards).
All angle values are in degrees unless noted.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AngleRange:
    ideal: float
    tolerance: float  # acceptable deviation before flagging
    label: str
    low_feedback: str   # feedback when value is too low
    high_feedback: str  # feedback when value is too high


RECURVE_BASELINE: dict[str, AngleRange] = {
    "draw_elbow_angle": AngleRange(
        ideal=160.0,
        tolerance=10.0,
        label="Draw elbow angle",
        low_feedback="Draw elbow is too bent — not enough back tension. Focus on using back muscles to pull.",
        high_feedback="Draw elbow is hyperextended. Relax the draw arm slightly and engage back muscles.",
    ),
    "bow_elbow_angle": AngleRange(
        ideal=175.0,
        tolerance=8.0,
        label="Bow arm elbow angle",
        low_feedback="Bow arm elbow is bent too much — risk of string slap. Extend the bow arm more.",
        high_feedback="Bow arm is over-extended. Slight soft elbow prevents injury and string contact.",
    ),
    "shoulder_tilt": AngleRange(
        ideal=0.0,
        tolerance=5.0,
        label="Shoulder level",
        low_feedback="Shoulders are tilted — left shoulder too high. Keep shoulders level.",
        high_feedback="Shoulders are tilted — right shoulder too high. Keep shoulders level.",
    ),
    "draw_shoulder_elevation": AngleRange(
        ideal=10.0,
        tolerance=8.0,
        label="Draw shoulder elevation",
        low_feedback="Draw shoulder is too low — may lack power in the draw.",
        high_feedback="Draw shoulder is raised (shrugging). Drop the draw shoulder down for cleaner form.",
    ),
    "bow_shoulder_depression": AngleRange(
        ideal=8.0,
        tolerance=6.0,
        label="Bow shoulder position",
        low_feedback="Bow shoulder is too low — may cause instability in the bow arm.",
        high_feedback="Bow shoulder is raised. Press it down for a stable bow arm platform.",
    ),
    "head_tilt": AngleRange(
        ideal=0.0,
        tolerance=5.0,
        label="Head tilt",
        low_feedback="Head is tilted to the left. Keep the head upright for consistent anchor.",
        high_feedback="Head is tilted to the right. Keep the head upright for consistent anchor.",
    ),
    "head_forward_lean": AngleRange(
        ideal=0.0,
        tolerance=3.0,
        label="Head forward lean",
        low_feedback="Head is leaning back. Bring chin slightly forward to meet the string.",
        high_feedback="Head is leaning too far forward. Keep a natural upright head position.",
    ),
    "stance_width_ratio": AngleRange(
        ideal=1.2,
        tolerance=0.2,
        label="Stance width",
        low_feedback="Stance is too narrow — less than shoulder width. Widen feet for better stability.",
        high_feedback="Stance is too wide. Bring feet closer to about shoulder-width apart.",
    ),
}


@dataclass
class FormIssue:
    field: str
    label: str
    severity: str       # "minor" | "moderate" | "significant"
    measured: float
    ideal: float
    deviation: float
    feedback: str


def classify_severity(deviation: float, tolerance: float) -> str:
    ratio = abs(deviation) / (tolerance + 1e-9)
    if ratio < 1.2:
        return "minor"
    elif ratio < 2.0:
        return "moderate"
    return "significant"


def check_against_baseline(angles_dict: dict[str, Optional[float]]) -> list[FormIssue]:
    """Compare measured angles against the recurve baseline and return issues."""
    issues: list[FormIssue] = []

    for field, baseline in RECURVE_BASELINE.items():
        value = angles_dict.get(field)
        if value is None:
            continue

        deviation = value - baseline.ideal
        if abs(deviation) <= baseline.tolerance:
            continue

        feedback = baseline.high_feedback if deviation > 0 else baseline.low_feedback
        severity = classify_severity(deviation, baseline.tolerance)

        issues.append(FormIssue(
            field=field,
            label=baseline.label,
            severity=severity,
            measured=round(value, 2),
            ideal=baseline.ideal,
            deviation=round(deviation, 2),
            feedback=feedback,
        ))

    # Sort by severity
    order = {"significant": 0, "moderate": 1, "minor": 2}
    issues.sort(key=lambda x: order[x.severity])
    return issues
