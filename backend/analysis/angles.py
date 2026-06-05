import numpy as np
from dataclasses import dataclass
from typing import Optional
from .pose import FramePose, Landmark, lm_to_array


@dataclass
class ShotAngles:
    frame_idx: int
    timestamp_ms: float

    # Draw arm
    draw_elbow_angle: Optional[float] = None        # shoulder-elbow-wrist
    draw_shoulder_elevation: Optional[float] = None  # vertical angle of draw shoulder

    # Bow arm
    bow_elbow_angle: Optional[float] = None         # shoulder-elbow-wrist
    bow_shoulder_depression: Optional[float] = None  # bow shoulder angle vs horizontal

    # Alignment
    shoulder_tilt: Optional[float] = None           # degrees off horizontal
    hip_shoulder_alignment: Optional[float] = None  # lateral rotation

    # Head / anchor
    head_tilt: Optional[float] = None               # lateral head tilt
    head_forward_lean: Optional[float] = None

    # Stance
    stance_width_ratio: Optional[float] = None      # foot width / shoulder width

    # Draw length proxy
    draw_wrist_to_bow_shoulder: Optional[float] = None  # normalized distance


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by a-b-c in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def horizontal_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of vector a->b relative to horizontal (y-axis in image coords)."""
    delta = b - a
    angle = float(np.degrees(np.arctan2(abs(delta[1]), abs(delta[0]))))
    return angle


def _vis(lm: Optional[Landmark], threshold: float = 0.5) -> bool:
    return lm is not None and lm.visibility >= threshold


def compute_angles(frame_pose: FramePose, draw_side: str = "right") -> ShotAngles:
    """
    Compute all relevant angles for a single frame.
    draw_side: 'right' for right-handed archers, 'left' for left-handed.
    """
    lm = frame_pose.landmarks
    bow_side = "left" if draw_side == "right" else "right"

    angles = ShotAngles(
        frame_idx=frame_pose.frame_idx,
        timestamp_ms=frame_pose.timestamp_ms,
    )

    # --- Draw elbow angle ---
    d_shoulder = lm.get(f"{draw_side}_shoulder")
    d_elbow = lm.get(f"{draw_side}_elbow")
    d_wrist = lm.get(f"{draw_side}_wrist")
    if all(_vis(x) for x in [d_shoulder, d_elbow, d_wrist]):
        angles.draw_elbow_angle = angle_between(
            lm_to_array(d_shoulder)[:2],
            lm_to_array(d_elbow)[:2],
            lm_to_array(d_wrist)[:2],
        )

    # --- Bow elbow angle ---
    b_shoulder = lm.get(f"{bow_side}_shoulder")
    b_elbow = lm.get(f"{bow_side}_elbow")
    b_wrist = lm.get(f"{bow_side}_wrist")
    if all(_vis(x) for x in [b_shoulder, b_elbow, b_wrist]):
        angles.bow_elbow_angle = angle_between(
            lm_to_array(b_shoulder)[:2],
            lm_to_array(b_elbow)[:2],
            lm_to_array(b_wrist)[:2],
        )

    # --- Shoulder tilt (how level shoulders are) ---
    l_sh = lm.get("left_shoulder")
    r_sh = lm.get("right_shoulder")
    if _vis(l_sh) and _vis(r_sh):
        delta_y = lm_to_array(r_sh)[1] - lm_to_array(l_sh)[1]
        delta_x = lm_to_array(r_sh)[0] - lm_to_array(l_sh)[0]
        angles.shoulder_tilt = float(np.degrees(np.arctan2(delta_y, delta_x + 1e-9)))

    # --- Draw shoulder elevation ---
    if _vis(d_shoulder) and _vis(d_elbow):
        angles.draw_shoulder_elevation = horizontal_angle(
            lm_to_array(d_shoulder)[:2], lm_to_array(d_elbow)[:2]
        )

    # --- Bow shoulder depression ---
    if _vis(b_shoulder) and _vis(b_elbow):
        angles.bow_shoulder_depression = horizontal_angle(
            lm_to_array(b_shoulder)[:2], lm_to_array(b_elbow)[:2]
        )

    # --- Head tilt ---
    nose = lm.get("nose")
    l_ear = lm.get("left_ear")
    r_ear = lm.get("right_ear")
    if _vis(nose) and _vis(l_ear) and _vis(r_ear):
        mid_ear = (lm_to_array(l_ear) + lm_to_array(r_ear)) / 2.0
        ear_delta = lm_to_array(r_ear) - lm_to_array(l_ear)
        angles.head_tilt = float(np.degrees(np.arctan2(ear_delta[1], ear_delta[0] + 1e-9)))
        # Forward lean: nose vs mid-ear horizontal offset
        angles.head_forward_lean = float(
            (lm_to_array(nose)[0] - mid_ear[0]) * 100  # normalized to image width %
        )

    # --- Stance width ratio ---
    l_foot = lm.get("left_foot_index")
    r_foot = lm.get("right_foot_index")
    if _vis(l_sh) and _vis(r_sh) and _vis(l_foot) and _vis(r_foot):
        shoulder_width = abs(lm_to_array(r_sh)[0] - lm_to_array(l_sh)[0])
        foot_width = abs(lm_to_array(r_foot)[0] - lm_to_array(l_foot)[0])
        if shoulder_width > 0:
            angles.stance_width_ratio = foot_width / shoulder_width

    # --- Draw length proxy ---
    if _vis(d_wrist) and _vis(b_shoulder):
        angles.draw_wrist_to_bow_shoulder = float(
            np.linalg.norm(lm_to_array(d_wrist)[:2] - lm_to_array(b_shoulder)[:2])
        )

    return angles
