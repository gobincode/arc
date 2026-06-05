import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FramePose:
    frame_idx: int
    timestamp_ms: float
    landmarks: dict[str, Landmark]
    raw: object = field(repr=False, default=None)


LANDMARK_NAMES = {i: name for i, name in enumerate([
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
])}


def extract_poses(video_path: str, skip_frames: int = 2) -> list[FramePose]:
    """Extract pose landmarks from every nth frame of a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    poses: list[FramePose] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % (skip_frames + 1) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    lms = {}
                    for i, lm in enumerate(result.pose_landmarks.landmark):
                        name = LANDMARK_NAMES.get(i)
                        if name:
                            lms[name] = Landmark(lm.x, lm.y, lm.z, lm.visibility)
                    poses.append(FramePose(
                        frame_idx=frame_idx,
                        timestamp_ms=(frame_idx / fps) * 1000,
                        landmarks=lms,
                        raw=result.pose_landmarks,
                    ))

            frame_idx += 1

    cap.release()
    return poses


def lm_to_array(lm: Landmark) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z])


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info
