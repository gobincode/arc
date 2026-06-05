"""
Render annotated video using RTMPose 133-keypoint detections.
Run AFTER rtmpose_train_test.py (needs archer_baseline.json).

    python rtmpose_overlay.py
"""

import os, sys, json, time
import cv2
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
import os as _os

BASE_DIR      = _os.path.dirname(_os.path.abspath(__file__))
TEST_VIDEO    = _os.path.join(BASE_DIR, "20250913_180517.mp4")
BASELINE_FILE = _os.path.join(BASE_DIR, "archer_baseline.json")
OUTPUT_VIDEO  = _os.path.join(BASE_DIR, "output", "test_annotated_rtmpose.mp4")
DRAW_SIDE     = "right"
PROCESS_WIDTH = 1280
PANEL_W       = 340
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# ── reuse helpers from rtmpose_train_test ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from rtmpose_train_test import (
    build_inferencer, run_inference, detect_shots,
    compute_angles, get_kp_px, KP, FIELD_LABELS,
    SEVERITY_THRESHOLDS, FEEDBACK,
)

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
C_GREEN  = (50,  220,  50)
C_YELLOW = (0,   200, 255)
C_RED    = (50,   50, 240)
C_WHITE  = (255, 255, 255)
C_GRAY   = (140, 140, 140)
C_BLUE   = (220, 160,  30)
C_CYAN   = (220, 210,  30)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
THICK      = 2


def sev_color(field, value, baseline):
    ref = baseline.get(field)
    if ref is None: return C_WHITE
    dev = abs(value - ref["median"])
    lo, hi = SEVERITY_THRESHOLDS.get(field, (5.0, 10.0))
    if dev < lo:  return C_GREEN
    if dev < hi:  return C_YELLOW
    return C_RED


def text_bg(img, text, pos, scale, color, thick=1, pad=4):
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, thick)
    x, y = pos
    cv2.rectangle(img, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_angle_arc(img, a_px, b_px, c_px, color, r=28):
    a = np.array(a_px, float); b = np.array(b_px, float); c = np.array(c_px, float)
    ba = a - b; bc = c - b
    if np.linalg.norm(ba) < 1 or np.linalg.norm(bc) < 1: return
    s = float(np.degrees(np.arctan2(-ba[1], ba[0])))
    e = float(np.degrees(np.arctan2(-bc[1], bc[0])))
    if s > e: s, e = e, s
    if e - s > 180: s, e = e, s + 360
    cv2.ellipse(img, tuple(b_px), (r, r), 0, s, e, color, 2, cv2.LINE_AA)


# COCO skeleton connections (body only, indices 0-16)
BODY_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11),(6, 12),(11,12),(11,13),(13,15),(12,14),(14,16),
    (0, 5), (0, 6),
]
# Hand connections (relative within hand block)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),   # thumb
    (0,5),(5,6),(6,7),(7,8),   # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
]


def draw_skeleton(canvas, kps_px, scores, W, H, draw_side):
    """Draw body + hand skeletons with confidence-based alpha."""
    # Body connections
    for a, b in BODY_CONNECTIONS:
        if scores[a] > 0.3 and scores[b] > 0.3:
            cv2.line(canvas, kps_px[a], kps_px[b], C_CYAN, 2, cv2.LINE_AA)

    # Body keypoints
    for i in range(17):
        if scores[i] > 0.3:
            cv2.circle(canvas, kps_px[i], 5, C_WHITE, -1)
            cv2.circle(canvas, kps_px[i], 5, C_CYAN, 1)

    # Hands
    for hand_offset in [91, 112]:
        color = C_GREEN if hand_offset == 91 else C_YELLOW
        for a, b in HAND_CONNECTIONS:
            ia, ib = hand_offset + a, hand_offset + b
            if scores[ia] > 0.25 and scores[ib] > 0.25:
                cv2.line(canvas, kps_px[ia], kps_px[ib], color, 1, cv2.LINE_AA)
        for i in range(21):
            idx = hand_offset + i
            if scores[idx] > 0.25:
                cv2.circle(canvas, kps_px[idx], 3, color, -1)


def draw_panel(canvas, angles, baseline, shot_id, phase, out_h):
    W = canvas.shape[1]
    x0 = W - PANEL_W
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, 0), (W, out_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    x = x0 + 10
    y = 32

    if shot_id is not None:
        cv2.putText(canvas, f"SHOT {shot_id+1}", (x, y), FONT, 0.8, C_WHITE, 2, cv2.LINE_AA)
        y += 26
        pc = {"DRAW": C_YELLOW, "FULL DRAW": C_GREEN, "FOLLOW THROUGH": C_BLUE}.get(phase, C_GRAY)
        cv2.putText(canvas, phase or "", (x, y), FONT, 0.5, pc, 1, cv2.LINE_AA)
        y += 22
    else:
        cv2.putText(canvas, "---", (x, y), FONT, 0.5, C_GRAY, 1, cv2.LINE_AA)
        y += 22

    cv2.line(canvas, (x, y), (W - 8, y), C_GRAY, 1)
    y += 14

    SHOW_FIELDS = [
        "draw_elbow_angle", "bow_elbow_angle",
        "draw_shoulder_elevation", "bow_shoulder_depression",
        "torso_lean", "shoulder_tilt_pct",
        "head_tilt_pct", "head_forward_lean",
        "anchor_height_diff", "anchor_to_chin_dist",
        "draw_finger_curl_avg", "bow_index_curl", "bow_thumb_extension",
    ]
    for field in SHOW_FIELDS:
        val = angles.get(field)
        if val is None: continue
        ref  = baseline.get(field)
        col  = sev_color(field, val, baseline)
        lbl  = FIELD_LABELS.get(field, field)
        line = f"{lbl}: {val:.1f}"
        if ref: line += f" ({ref['median']:.1f})"
        dot_col = col
        cv2.circle(canvas, (x + 5, y - 4), 4, dot_col, -1)
        cv2.putText(canvas, line, (x + 14, y), FONT, 0.38, col, 1, cv2.LINE_AA)
        y += 16
        if y > out_h - 20: break


def build_phase_map(shots, n_frames):
    pm = {}
    for sid, s in enumerate(shots):
        for fi in range(s["draw_start"], s["follow_end"] + 1):
            if fi <= s["full_draw"]:  pm[fi] = (sid, "DRAW")
            elif fi <= s["release"]:  pm[fi] = (sid, "FULL DRAW")
            else:                     pm[fi] = (sid, "FOLLOW THROUGH")
    return pm


def render(video_path, baseline, draw_side=DRAW_SIDE, output_path=OUTPUT_VIDEO):
    inferencer = build_inferencer()

    cap    = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    scale  = PROCESS_WIDTH / orig_w
    proc_h = int(orig_h * scale)
    out_w  = PROCESS_WIDTH + PANEL_W
    out_h  = proc_h

    print(f"Input:  {orig_w}x{orig_h} @ {fps:.0f}fps")
    print(f"Output: {out_w}x{out_h} -> {output_path}")

    # Pass 1: get all poses for shot detection
    print("\nPass 1: pose extraction for shot detection...")
    pose_frames, _, _ = run_inference(video_path, inferencer, skip=2, width=PROCESS_WIDTH)
    shots = detect_shots(pose_frames, draw_side)

    # Build lookup: frame_idx -> (norm_kps, scores)
    pose_lookup = {fi: (nk, sc) for fi, nk, sc in pose_frames}

    # phase map indexed by frame_idx in pose_frames list
    pose_fi_list = [fi for fi, _, _ in pose_frames]
    phase_map_pi = build_phase_map(shots, len(pose_frames))
    # convert to real frame_idx
    phase_map = {}
    for pi, (sid, ph) in phase_map_pi.items():
        if pi < len(pose_fi_list):
            phase_map[pose_fi_list[pi]] = (sid, ph)

    # Pass 2: render every frame
    print("\nPass 2: rendering...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    cap = cv2.VideoCapture(video_path)
    t0  = time.time()
    fi  = 0

    # re-run inference per-frame during render (or reuse cached poses)
    from mmpose.apis import MMPoseInferencer
    inf2 = build_inferencer()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        small  = cv2.resize(frame, (PROCESS_WIDTH, proc_h))
        canvas = np.zeros((out_h, out_w, 3), np.uint8)
        canvas[:, :PROCESS_WIDTH] = small

        angles = {}
        shot_id, phase = phase_map.get(fi, (None, None))

        # run pose on this frame
        rgb  = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        preds = list(inf2(rgb, show=False, return_datasamples=False))
        if preds and preds[0].get("predictions"):
            persons = preds[0]["predictions"][0]
            if persons:
                best   = max(persons, key=lambda p: p.get("bbox_score", 0))
                kps    = np.array(best["keypoints"])
                scores = np.array(best["keypoint_scores"])

                # normalize for angle computation
                norm = kps.copy().astype(float)
                norm[:, 0] /= PROCESS_WIDTH
                norm[:, 1] /= proc_h

                angles = compute_angles(norm, scores, draw_side)

                # pixel coords for drawing
                kps_px = [(int(kps[i][0]), int(kps[i][1])) for i in range(len(kps))]

                # Skeleton
                draw_skeleton(canvas[:, :PROCESS_WIDTH], kps_px, scores, PROCESS_WIDTH, proc_h, draw_side)

                bow_side = "left" if draw_side == "right" else "right"

                # Draw elbow angle arc
                ds_px = kps_px[KP[f"{draw_side}_shoulder"]]
                de_px = kps_px[KP[f"{draw_side}_elbow"]]
                dw_px = kps_px[KP[f"{draw_side}_wrist"]]
                if scores[KP[f"{draw_side}_elbow"]] > 0.3 and angles.get("draw_elbow_angle"):
                    col = sev_color("draw_elbow_angle", angles["draw_elbow_angle"], baseline)
                    draw_angle_arc(canvas[:, :PROCESS_WIDTH], ds_px, de_px, dw_px, col)
                    text_bg(canvas[:, :PROCESS_WIDTH],
                             f"{angles['draw_elbow_angle']:.0f}°",
                             (de_px[0]+8, de_px[1]-8), 0.55, col, THICK)

                # Bow elbow angle
                be_px = kps_px[KP[f"{bow_side}_elbow"]]
                if scores[KP[f"{bow_side}_elbow"]] > 0.3 and angles.get("bow_elbow_angle"):
                    col = sev_color("bow_elbow_angle", angles["bow_elbow_angle"], baseline)
                    text_bg(canvas[:, :PROCESS_WIDTH],
                             f"{angles['bow_elbow_angle']:.0f}°",
                             (be_px[0]+8, be_px[1]-8), 0.55, col, THICK)

                # Torso lean line
                ls_px = kps_px[KP["left_shoulder"]]; rs_px = kps_px[KP["right_shoulder"]]
                lh_px = kps_px[KP["left_hip"]];      rh_px = kps_px[KP["right_hip"]]
                if all(scores[KP[k]] > 0.3 for k in ["left_shoulder","right_shoulder","left_hip","right_hip"]):
                    sh_c  = ((ls_px[0]+rs_px[0])//2, (ls_px[1]+rs_px[1])//2)
                    hip_c = ((lh_px[0]+rh_px[0])//2, (lh_px[1]+rh_px[1])//2)
                    if angles.get("torso_lean") is not None:
                        col = sev_color("torso_lean", angles["torso_lean"], baseline)
                        cv2.line(canvas[:, :PROCESS_WIDTH], hip_c, sh_c, col, 3, cv2.LINE_AA)
                        text_bg(canvas[:, :PROCESS_WIDTH],
                                 f"lean:{angles['torso_lean']:.1f}°",
                                 (sh_c[0]+8, sh_c[1]-8), 0.42, col)

                # Draw wrist anchor dot
                if scores[KP[f"{draw_side}_wrist"]] > 0.3:
                    col = sev_color("anchor_height_diff", angles.get("anchor_height_diff", 0), baseline)
                    cv2.circle(canvas[:, :PROCESS_WIDTH], dw_px, 12, col, 3)

                # Draw finger curl indicator on hand
                dh_off = 112 if draw_side == "right" else 91
                if scores[dh_off] > 0.25 and angles.get("draw_finger_curl_avg") is not None:
                    col = sev_color("draw_finger_curl_avg", angles["draw_finger_curl_avg"], baseline)
                    text_bg(canvas[:, :PROCESS_WIDTH],
                             f"grip:{angles['draw_finger_curl_avg']:.0f}°",
                             (kps_px[dh_off][0]+8, kps_px[dh_off][1]-8), 0.42, col)

        # Panel
        draw_panel(canvas, angles, baseline, shot_id, phase, out_h)

        # HUD top-left
        cv2.putText(canvas, f"Frame {fi}  {fi/fps:.1f}s", (10, 22),
                    FONT, 0.48, C_GRAY, 1, cv2.LINE_AA)
        if shot_id is not None:
            pc = {"DRAW": C_YELLOW, "FULL DRAW": C_GREEN, "FOLLOW THROUGH": C_BLUE}.get(phase, C_GRAY)
            cv2.putText(canvas, f"SHOT {shot_id+1}  {phase}", (10, 50),
                        FONT, 0.72, pc, 2, cv2.LINE_AA)

        writer.write(canvas)
        fi += 1
        if fi % 300 == 0:
            eta = ((time.time()-t0)/fi)*(total-fi)
            print(f"  {fi}/{total} ({fi/total*100:.0f}%)  ETA {eta:.0f}s")

    cap.release()
    writer.release()
    print(f"\nDone -> {output_path}")


if __name__ == "__main__":
    if not os.path.exists(BASELINE_FILE):
        print("Run rtmpose_train_test.py first."); sys.exit(1)
    with open(BASELINE_FILE) as f:
        data = json.load(f)
    baseline  = data["angles"]
    DRAW_SIDE = data.get("draw_side", DRAW_SIDE)
    render(TEST_VIDEO, baseline, DRAW_SIDE, OUTPUT_VIDEO)
