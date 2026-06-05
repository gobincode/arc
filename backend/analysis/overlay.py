"""
Draw skeleton overlays and angle annotations onto video frames.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
from .pose import FramePose, LANDMARK_NAMES
from .angles import ShotAngles
from .baseline import RECURVE_BASELINE

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Severity colors (BGR)
COLOR_OK = (0, 220, 0)
COLOR_MINOR = (0, 200, 255)
COLOR_MODERATE = (0, 140, 255)
COLOR_SIGNIFICANT = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (160, 160, 160)


def _severity_color(deviation: float, tolerance: float):
    ratio = abs(deviation) / (tolerance + 1e-9)
    if ratio < 1.0:
        return COLOR_OK
    elif ratio < 1.5:
        return COLOR_MINOR
    elif ratio < 2.0:
        return COLOR_MODERATE
    return COLOR_SIGNIFICANT


def draw_pose_on_frame(
    frame: np.ndarray,
    frame_pose: FramePose,
    angles: Optional[ShotAngles] = None,
    draw_side: str = "right",
) -> np.ndarray:
    """Draw skeleton + angle annotations on a single frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    if frame_pose.raw is not None:
        mp_drawing.draw_landmarks(
            out,
            frame_pose.raw,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    if angles is None:
        return out

    # Overlay angle values near relevant joints
    lm = frame_pose.landmarks
    bow_side = "left" if draw_side == "right" else "right"

    def put_angle(joint_name: str, value: Optional[float], label: str, field: str):
        if value is None:
            return
        joint = lm.get(joint_name)
        if joint is None:
            return
        px = int(joint.x * w) + 10
        py = int(joint.y * h) - 10
        baseline = RECURVE_BASELINE.get(field)
        if baseline:
            deviation = value - baseline.ideal
            color = _severity_color(deviation, baseline.tolerance)
        else:
            color = COLOR_WHITE
        cv2.putText(out, f"{label}: {value:.0f}", (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    put_angle(f"{draw_side}_elbow", angles.draw_elbow_angle, "DE", "draw_elbow_angle")
    put_angle(f"{bow_side}_elbow", angles.bow_elbow_angle, "BE", "bow_elbow_angle")
    put_angle(f"{draw_side}_shoulder", angles.draw_shoulder_elevation, "DS", "draw_shoulder_elevation")
    put_angle(f"{bow_side}_shoulder", angles.bow_shoulder_depression, "BS", "bow_shoulder_depression")
    put_angle("nose", angles.head_tilt, "HT", "head_tilt")

    # Phase label
    return out


def annotate_video(
    input_path: str,
    output_path: str,
    poses: list[FramePose],
    angles_map: dict[int, ShotAngles],
    phase_map: dict[int, str],
    draw_side: str = "right",
) -> str:
    """Re-encode a video with skeleton and angle overlays."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pose_by_frame = {p.frame_idx: p for p in poses}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fp = pose_by_frame.get(frame_idx)
        if fp is not None:
            ang = angles_map.get(frame_idx)
            frame = draw_pose_on_frame(frame, fp, ang, draw_side)

        phase_label = phase_map.get(frame_idx)
        if phase_label:
            cv2.putText(frame, phase_label.upper(), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3, cv2.LINE_AA)

        out_writer.write(frame)
        frame_idx += 1

    cap.release()
    out_writer.release()
    return output_path
