"""
Render annotated video with:
  - MediaPipe skeleton overlay
  - Angle values at key joints (green=ok, yellow=moderate, red=significant)
  - Shot number + phase label
  - Live issues panel on the right
  - Per-shot summary at release frame
"""

import sys, os, json, time
import cv2
import numpy as np
import mediapipe as mp

# ── config ────────────────────────────────────────────────────────────────────
TEST_VIDEO    = r"C:\Users\Lenovo\Desktop\arc\20250913_180517.mp4"
BASELINE_FILE = r"C:\Users\Lenovo\Desktop\arc\archer_baseline.json"
OUTPUT_VIDEO  = r"C:\Users\Lenovo\Desktop\arc\output\test_annotated.mp4"
DRAW_SIDE     = "right"
PROCESS_WIDTH = 1280     # output resolution width
SKIP_FRAMES   = 0        # 0 = process every frame for smooth video
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Colors (BGR)
C_GREEN  = (50,  220,  50)
C_YELLOW = (0,   200, 255)
C_RED    = (50,   50, 240)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_GRAY   = (140, 140, 140)
C_BLUE   = (220, 160,  30)
C_BG     = (20,   20,  20)

LANDMARK_IDX = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_foot_index": 31, "right_foot_index": 32,
}

SEVERITY_THRESHOLDS = {
    "draw_elbow_angle":        (8.0,  15.0),
    "bow_elbow_angle":         (8.0,  15.0),
    "shoulder_tilt_pct":       (1.5,   3.0),
    "torso_lean":              (3.0,   7.0),
    "draw_shoulder_elevation": (5.0,  10.0),
    "bow_shoulder_depression": (5.0,  10.0),
    "head_tilt_pct":           (1.5,   3.0),
    "head_forward_lean":       (2.5,   5.0),
    "anchor_height_diff":      (2.0,   4.0),
}

FIELD_LABELS = {
    "draw_elbow_angle":        "Draw elbow",
    "bow_elbow_angle":         "Bow elbow",
    "shoulder_tilt_pct":       "Shoulder level",
    "torso_lean":              "Torso lean",
    "draw_shoulder_elevation": "Draw shoulder",
    "bow_shoulder_depression": "Bow shoulder",
    "head_tilt_pct":           "Head tilt",
    "head_forward_lean":       "Head lean",
    "anchor_height_diff":      "Anchor height",
}


def get_lm_px(landmarks, name, w, h):
    i = LANDMARK_IDX.get(name)
    if i is None: return None
    lm = landmarks.landmark[i]
    if lm.visibility < 0.35: return None
    return (int(lm.x * w), int(lm.y * h))


def get_lm_np(landmarks, name):
    i = LANDMARK_IDX.get(name)
    if i is None: return None
    lm = landmarks.landmark[i]
    if lm.visibility < 0.35: return None
    return np.array([lm.x, lm.y])


def angle3(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def horiz_angle(a, b):
    d = b - a
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-9)))


def compute_angles(landmarks, draw_side):
    bow = "left" if draw_side == "right" else "right"
    D, B = draw_side, bow
    out = {}

    ds = get_lm_np(landmarks, f"{D}_shoulder")
    de = get_lm_np(landmarks, f"{D}_elbow")
    dw = get_lm_np(landmarks, f"{D}_wrist")
    bs = get_lm_np(landmarks, f"{B}_shoulder")
    be = get_lm_np(landmarks, f"{B}_elbow")
    bw = get_lm_np(landmarks, f"{B}_wrist")
    ls = get_lm_np(landmarks, "left_shoulder")
    rs = get_lm_np(landmarks, "right_shoulder")
    lh = get_lm_np(landmarks, "left_hip")
    rh = get_lm_np(landmarks, "right_hip")
    nose = get_lm_np(landmarks, "nose")
    le = get_lm_np(landmarks, "left_ear")
    re = get_lm_np(landmarks, "right_ear")

    if all(x is not None for x in [ds, de, dw]):
        out["draw_elbow_angle"] = angle3(ds, de, dw)
    if all(x is not None for x in [bs, be, bw]):
        out["bow_elbow_angle"] = angle3(bs, be, bw)
    if ls is not None and rs is not None:
        out["shoulder_tilt_pct"] = float((rs[1] - ls[1]) * 100)
    if all(x is not None for x in [ls, rs, lh, rh]):
        sh_mid  = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        d = sh_mid - hip_mid
        out["torso_lean"] = float(np.degrees(np.arctan2(d[0], -d[1] + 1e-9)))
    if ds is not None and de is not None:
        out["draw_shoulder_elevation"] = horiz_angle(ds, de)
    if bs is not None and be is not None:
        out["bow_shoulder_depression"] = horiz_angle(bs, be)
    if le is not None and re is not None:
        out["head_tilt_pct"] = float((re[1] - le[1]) * 100)
        mid_ear = (le + re) / 2
        if nose is not None:
            out["head_forward_lean"] = float((nose[0] - mid_ear[0]) * 100)
    if dw is not None and bs is not None:
        out["draw_extension"] = float(np.linalg.norm(dw - bs))
    if dw is not None and nose is not None:
        out["anchor_height_diff"] = float((dw[1] - nose[1]) * 100)

    return out


def severity_color(field, value, baseline):
    ref = baseline.get(field)
    if ref is None: return C_WHITE
    dev = abs(value - ref["median"])
    lo, hi = SEVERITY_THRESHOLDS.get(field, (5.0, 10.0))
    if dev < lo:  return C_GREEN
    if dev < hi:  return C_YELLOW
    return C_RED


def severity_label(field, value, baseline):
    ref = baseline.get(field)
    if ref is None: return "ok"
    dev = abs(value - ref["median"])
    lo, hi = SEVERITY_THRESHOLDS.get(field, (5.0, 10.0))
    if dev < lo:  return "ok"
    if dev < hi:  return "moderate"
    return "significant"


# ── shot detection (same heuristic as train_and_test.py) ─────────────────────

def detect_shots(ext_series):
    """Returns list of {draw_start, full_draw, release, follow_end} in frame-list indices."""
    w = 7
    smooth = np.convolve(ext_series, np.ones(w) / w, mode="same")
    norm   = smooth / (max(smooth) + 1e-9)
    shots, i, n = [], 0, len(norm)

    while i < n - 5:
        if norm[i] < 0.55:
            i += 1; continue
        peak_i, peak_v = i, 0.0
        j = i
        while j < min(i + 150, n):
            if norm[j] > peak_v:
                peak_v, peak_i = norm[j], j
            j += 1
        if peak_v < 0.82:
            i = j; continue
        release_i = None
        k = peak_i
        while k < min(peak_i + 80, n):
            if norm[k] < 0.45:
                release_i = k; break
            k += 1
        if release_i is None:
            i = peak_i + 1; continue
        draw_start = peak_i
        m = peak_i
        while m > max(0, peak_i - 100):
            if norm[m] < 0.55:
                draw_start = m + 1; break
            m -= 1
        follow_end = min(release_i + 25, n - 1)
        shots.append({"draw_start": draw_start, "full_draw": peak_i,
                      "release": release_i, "follow_end": follow_end})
        i = follow_end + 1
    return shots


def get_phase(fi, shots):
    """Return (shot_id, phase_str) or (None, None)."""
    for sid, s in enumerate(shots):
        ds, fd, rel, fe = s["draw_start"], s["full_draw"], s["release"], s["follow_end"]
        if fi < ds:   continue
        if fi <= fd:  return sid, "DRAW"
        if fi <= rel: return sid, "FULL DRAW"
        if fi <= fe:  return sid, "FOLLOW THROUGH"
    return None, None


# ── drawing helpers ───────────────────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.52
FONT_MED   = 0.70
FONT_LARGE = 1.10
THICK      = 2


def draw_text_bg(img, text, pos, font_scale, color, thickness=1, pad=4):
    """Draw text with a dark background rectangle."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_angle_at(img, pt, value, field, baseline, suffix="°"):
    if pt is None: return
    color = severity_color(field, value, baseline)
    label = f"{value:.0f}{suffix}"
    draw_text_bg(img, label, (pt[0] + 8, pt[1] - 8), FONT_SMALL, color, THICK)


def draw_side_panel(img, angles, baseline, shot_id, phase, h, panel_w=320):
    """Draw a semi-transparent info panel on the right edge."""
    W = img.shape[1]
    overlay = img.copy()
    cv2.rectangle(overlay, (W - panel_w, 0), (W, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    x = W - panel_w + 10
    y = 30

    # Header
    if shot_id is not None:
        header = f"SHOT {shot_id + 1}"
        cv2.putText(img, header, (x, y), FONT, FONT_MED, C_WHITE, 2, cv2.LINE_AA)
        y += 28
        phase_color = {
            "DRAW": C_YELLOW, "FULL DRAW": C_GREEN,
            "FOLLOW THROUGH": C_BLUE,
        }.get(phase, C_GRAY)
        cv2.putText(img, phase or "", (x, y), FONT, FONT_SMALL, phase_color, 1, cv2.LINE_AA)
        y += 24
    else:
        cv2.putText(img, "No shot", (x, y), FONT, FONT_SMALL, C_GRAY, 1, cv2.LINE_AA)
        y += 24

    cv2.line(img, (x, y), (W - 10, y), C_GRAY, 1)
    y += 14

    # Angle rows
    for field in [
        "draw_elbow_angle", "bow_elbow_angle",
        "draw_shoulder_elevation", "bow_shoulder_depression",
        "torso_lean", "shoulder_tilt_pct",
        "head_tilt_pct", "head_forward_lean", "anchor_height_diff",
    ]:
        val = angles.get(field)
        if val is None: continue
        ref = baseline.get(field)
        color = severity_color(field, val, baseline)
        sev   = severity_label(field, val, baseline)
        lbl   = FIELD_LABELS.get(field, field)
        text  = f"{lbl}: {val:.1f}"
        if ref:
            text += f"  (ref {ref['median']:.1f})"
        # severity dot
        dot_color = C_GREEN if sev == "ok" else (C_YELLOW if sev == "moderate" else C_RED)
        cv2.circle(img, (x + 6, y - 5), 5, dot_color, -1)
        cv2.putText(img, text, (x + 16, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        y += 18
        if y > h - 20: break

    return img


# ── main render loop ──────────────────────────────────────────────────────────

def render(video_path, baseline, draw_side=DRAW_SIDE, output_path=OUTPUT_VIDEO):
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale  = PROCESS_WIDTH / orig_w
    proc_h = int(orig_h * scale)
    panel_w = 330
    out_w   = PROCESS_WIDTH + panel_w
    out_h   = proc_h

    print(f"Input:  {orig_w}x{orig_h} @ {fps:.0f}fps  ({total} frames)")
    print(f"Output: {out_w}x{out_h} -> {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # ── Pass 1: extract extension series for shot detection ──
    print("Pass 1: detecting shots...")
    ext_series = []
    frame_indices = []

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fi = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            small = cv2.resize(frame, (PROCESS_WIDTH, proc_h))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res   = pose.process(rgb)
            ext = 0.0
            if res.pose_landmarks:
                bow = "left" if draw_side == "right" else "right"
                dw = get_lm_np(res.pose_landmarks, f"{draw_side}_wrist")
                bs = get_lm_np(res.pose_landmarks, f"{bow}_shoulder")
                if dw is not None and bs is not None:
                    ext = float(np.linalg.norm(dw - bs))
            ext_series.append(ext)
            frame_indices.append(fi)
            fi += 1

    shots = detect_shots(ext_series)
    print(f"  Detected {len(shots)} shots")

    # Build frame_idx -> (shot_id, phase) lookup
    phase_map = {}
    for sid, s in enumerate(shots):
        for fi2 in range(s["draw_start"], s["follow_end"] + 1):
            if fi2 <= s["full_draw"]:  phase_map[fi2] = (sid, "DRAW")
            elif fi2 <= s["release"]:  phase_map[fi2] = (sid, "FULL DRAW")
            else:                      phase_map[fi2] = (sid, "FOLLOW THROUGH")

    # Build per-shot angle snapshots (at full draw) for the summary
    shot_angles = {}
    for sid, s in enumerate(shots):
        # we'll fill this during pass 2 when we hit full_draw frame
        shot_angles[sid] = None

    # ── Pass 2: render ──
    print("Pass 2: rendering overlay...")
    t0 = time.time()

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fi = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            small = cv2.resize(frame, (PROCESS_WIDTH, proc_h))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res   = pose.process(rgb)

            # Canvas: video on left, panel on right
            canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            canvas[:, :PROCESS_WIDTH] = small

            shot_id, phase = phase_map.get(fi, (None, None))
            angles = {}

            if res.pose_landmarks:
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    canvas[:, :PROCESS_WIDTH],
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                angles = compute_angles(res.pose_landmarks, draw_side)

                # Store full-draw snapshot
                if shot_id is not None and phase == "FULL DRAW":
                    shot_angles[shot_id] = angles

                bow = "left" if draw_side == "right" else "right"
                W, H = PROCESS_WIDTH, proc_h

                # Draw elbow angle arc + label
                de_px = get_lm_px(res.pose_landmarks, f"{draw_side}_elbow", W, H)
                ds_px = get_lm_px(res.pose_landmarks, f"{draw_side}_shoulder", W, H)
                dw_px = get_lm_px(res.pose_landmarks, f"{draw_side}_wrist", W, H)
                if de_px and angles.get("draw_elbow_angle"):
                    color = severity_color("draw_elbow_angle", angles["draw_elbow_angle"], baseline)
                    draw_angle_at(canvas[:, :PROCESS_WIDTH], de_px,
                                  angles["draw_elbow_angle"], "draw_elbow_angle", baseline)
                    # draw arc
                    if ds_px and dw_px:
                        _draw_angle_arc(canvas[:, :PROCESS_WIDTH], ds_px, de_px, dw_px, color)

                # Bow elbow angle
                be_px = get_lm_px(res.pose_landmarks, f"{bow}_elbow", W, H)
                if be_px and angles.get("bow_elbow_angle"):
                    color = severity_color("bow_elbow_angle", angles["bow_elbow_angle"], baseline)
                    draw_angle_at(canvas[:, :PROCESS_WIDTH], be_px,
                                  angles["bow_elbow_angle"], "bow_elbow_angle", baseline)

                # Torso lean line
                ls_px = get_lm_px(res.pose_landmarks, "left_shoulder", W, H)
                rs_px = get_lm_px(res.pose_landmarks, "right_shoulder", W, H)
                lh_px = get_lm_px(res.pose_landmarks, "left_hip", W, H)
                rh_px = get_lm_px(res.pose_landmarks, "right_hip", W, H)
                if ls_px and rs_px and lh_px and rh_px and angles.get("torso_lean") is not None:
                    sh_cx  = (ls_px[0] + rs_px[0]) // 2
                    sh_cy  = (ls_px[1] + rs_px[1]) // 2
                    hip_cx = (lh_px[0] + rh_px[0]) // 2
                    hip_cy = (lh_px[1] + rh_px[1]) // 2
                    color  = severity_color("torso_lean", angles["torso_lean"], baseline)
                    cv2.line(canvas[:, :PROCESS_WIDTH],
                             (hip_cx, hip_cy), (sh_cx, sh_cy), color, 3, cv2.LINE_AA)
                    draw_text_bg(canvas[:, :PROCESS_WIDTH],
                                 f"lean:{angles['torso_lean']:.1f}°",
                                 (sh_cx + 8, sh_cy - 8), 0.45, color)

                # Anchor point dot on draw wrist
                dw_px2 = get_lm_px(res.pose_landmarks, f"{draw_side}_wrist", W, H)
                if dw_px2 and angles.get("anchor_height_diff") is not None:
                    color = severity_color("anchor_height_diff", angles["anchor_height_diff"], baseline)
                    cv2.circle(canvas[:, :PROCESS_WIDTH], dw_px2, 10, color, 3)

            # Side panel
            draw_side_panel(canvas, angles, baseline, shot_id, phase, out_h, panel_w)

            # Top-left HUD
            _draw_hud(canvas, fi, fps, shot_id, phase, out_w)

            writer.write(canvas)
            fi += 1

            if fi % 300 == 0:
                elapsed = time.time() - t0
                pct = fi / total * 100
                eta = (elapsed / fi) * (total - fi) if fi > 0 else 0
                print(f"  {fi}/{total} frames ({pct:.0f}%)  ETA {eta:.0f}s")

    cap.release()
    writer.release()
    print(f"\nDone in {time.time()-t0:.1f}s -> {output_path}")


def _draw_angle_arc(img, a_px, b_px, c_px, color, radius=30):
    """Draw a small arc at joint b between arms ba and bc."""
    a = np.array(a_px, dtype=float)
    b = np.array(b_px, dtype=float)
    c = np.array(c_px, dtype=float)
    ba = a - b; bc = c - b
    if np.linalg.norm(ba) < 1 or np.linalg.norm(bc) < 1: return
    ang_start = float(np.degrees(np.arctan2(-ba[1], ba[0])))
    ang_end   = float(np.degrees(np.arctan2(-bc[1], bc[0])))
    if ang_start > ang_end: ang_start, ang_end = ang_end, ang_start
    if ang_end - ang_start > 180: ang_start, ang_end = ang_end, ang_start + 360
    cv2.ellipse(img, b_px, (radius, radius), 0, ang_start, ang_end, color, 2, cv2.LINE_AA)


def _draw_hud(canvas, fi, fps, shot_id, phase, out_w):
    """Top-left: frame counter + shot/phase."""
    ts = f"Frame {fi}  |  {fi/fps:.1f}s"
    cv2.putText(canvas, ts, (10, 25), FONT, FONT_SMALL, C_GRAY, 1, cv2.LINE_AA)
    if shot_id is not None:
        phase_color = {
            "DRAW": C_YELLOW, "FULL DRAW": C_GREEN, "FOLLOW THROUGH": C_BLUE,
        }.get(phase, C_GRAY)
        label = f"SHOT {shot_id + 1}  {phase}"
        cv2.putText(canvas, label, (10, 52), FONT, FONT_MED, phase_color, 2, cv2.LINE_AA)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: baseline not found at {BASELINE_FILE}")
        print("Run train_and_test.py first.")
        sys.exit(1)

    with open(BASELINE_FILE) as f:
        data = json.load(f)
    baseline   = data["angles"]
    DRAW_SIDE  = data.get("draw_side", DRAW_SIDE)
    print(f"Loaded baseline ({data.get('shot_count','?')} training shots, draw_side={DRAW_SIDE})")

    render(TEST_VIDEO, baseline, DRAW_SIDE, OUTPUT_VIDEO)
