"""
Detect individual shot phases within a video using pose motion analysis.
Phases: SETUP -> DRAW -> FULL_DRAW -> RELEASE -> FOLLOW_THROUGH
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from .pose import FramePose, lm_to_array


class Phase(str, Enum):
    SETUP = "setup"
    DRAW = "draw"
    FULL_DRAW = "full_draw"
    RELEASE = "release"
    FOLLOW_THROUGH = "follow_through"
    UNKNOWN = "unknown"


@dataclass
class ShotPhase:
    phase: Phase
    start_frame: int
    end_frame: int
    key_frame: int  # most representative frame for this phase


@dataclass
class DetectedShot:
    shot_id: int
    phases: list[ShotPhase]

    @property
    def full_draw_frame(self) -> Optional[int]:
        for p in self.phases:
            if p.phase == Phase.FULL_DRAW:
                return p.key_frame
        return None

    @property
    def release_frame(self) -> Optional[int]:
        for p in self.phases:
            if p.phase == Phase.RELEASE:
                return p.key_frame
        return None


def _wrist_velocity(poses: list[FramePose], draw_side: str, idx: int) -> float:
    """Compute draw wrist movement speed between adjacent frames."""
    if idx <= 0 or idx >= len(poses):
        return 0.0
    wrist_key = f"{draw_side}_wrist"
    w_curr = poses[idx].landmarks.get(wrist_key)
    w_prev = poses[idx - 1].landmarks.get(wrist_key)
    if w_curr is None or w_prev is None:
        return 0.0
    return float(np.linalg.norm(lm_to_array(w_curr)[:2] - lm_to_array(w_prev)[:2]))


def _draw_extension(poses: list[FramePose], draw_side: str, idx: int) -> float:
    """Proxy for draw length: distance from draw wrist to bow shoulder."""
    bow_side = "left" if draw_side == "right" else "right"
    wrist = poses[idx].landmarks.get(f"{draw_side}_wrist")
    b_shoulder = poses[idx].landmarks.get(f"{bow_side}_shoulder")
    if wrist is None or b_shoulder is None:
        return 0.0
    return float(np.linalg.norm(lm_to_array(wrist)[:2] - lm_to_array(b_shoulder)[:2]))


def detect_shots(poses: list[FramePose], draw_side: str = "right") -> list[DetectedShot]:
    """
    Segment the pose sequence into individual shots and label phases.
    Uses draw wrist velocity and extension distance heuristics.
    """
    if len(poses) < 10:
        return []

    velocities = [_wrist_velocity(poses, draw_side, i) for i in range(len(poses))]
    extensions = [_draw_extension(poses, draw_side, i) for i in range(len(poses))]

    # Smooth
    window = 5
    smooth_vel = np.convolve(velocities, np.ones(window) / window, mode="same")
    smooth_ext = np.convolve(extensions, np.ones(window) / window, mode="same")

    max_ext = max(smooth_ext) if max(smooth_ext) > 0 else 1.0
    norm_ext = smooth_ext / max_ext

    shots: list[DetectedShot] = []
    shot_id = 0
    i = 0
    n = len(poses)

    while i < n:
        # Find start of draw: wrist starts moving + extension increasing
        if smooth_vel[i] > 0.01 and norm_ext[i] < 0.6:
            draw_start = i

            # Find full draw: extension peaks (low velocity + high extension)
            peak_ext_idx = draw_start
            peak_ext_val = 0.0
            j = draw_start
            while j < min(draw_start + 200, n):
                if norm_ext[j] > peak_ext_val:
                    peak_ext_val = norm_ext[j]
                    peak_ext_idx = j
                if norm_ext[j] > 0.85 and smooth_vel[j] < 0.015:
                    peak_ext_idx = j
                    break
                j += 1

            if peak_ext_val < 0.5:
                i += 1
                continue

            full_draw_idx = peak_ext_idx

            # Find release: sudden high velocity spike after full draw
            release_idx = None
            k = full_draw_idx
            while k < min(full_draw_idx + 60, n):
                if smooth_vel[k] > 0.04:
                    release_idx = k
                    break
                k += 1

            if release_idx is None:
                i = full_draw_idx + 1
                continue

            follow_end = min(release_idx + 30, n - 1)

            phases = [
                ShotPhase(Phase.DRAW, draw_start, full_draw_idx, (draw_start + full_draw_idx) // 2),
                ShotPhase(Phase.FULL_DRAW, full_draw_idx, release_idx, full_draw_idx),
                ShotPhase(Phase.RELEASE, release_idx, min(release_idx + 5, n - 1), release_idx),
                ShotPhase(Phase.FOLLOW_THROUGH, release_idx + 5, follow_end, (release_idx + 5 + follow_end) // 2),
            ]

            shots.append(DetectedShot(shot_id=shot_id, phases=phases))
            shot_id += 1
            i = follow_end + 1
        else:
            i += 1

    return shots


def get_key_frames(shot: DetectedShot) -> dict[str, int]:
    """Return a dict of phase -> key frame index for a shot."""
    return {p.phase.value: p.key_frame for p in shot.phases}
