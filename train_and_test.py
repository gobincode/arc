"""
Archery Shot Analyzer - Train on video 1, test on video 2.

Usage:
    python train_and_test.py

Steps:
  1. Extract poses from train video (resized for speed)
  2. Detect shots + compute per-archer angle medians -> archer_baseline.json
  3. Extract poses from test video
  4. Compare each test shot against personalized baseline
  5. Compare consecutive test shots against each other (what changed)
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import mediapipe as mp

# ── config ──────────────────────────────────────────────────────────────────
TRAIN_VIDEO = r"C:\Users\Lenovo\Desktop\arc\20250913_175626.mp4"
TEST_VIDEO  = r"C:\Users\Lenovo\Desktop\arc\20250913_180517.mp4"
BASELINE_FILE = r"C:\Users\Lenovo\Desktop\arc\archer_baseline.json"
OUTPUT_DIR = r"C:\Users\Lenovo\Desktop\arc\output"
PROCESS_WIDTH = 960      # resize to this width before pose estimation
SKIP_FRAMES   = 3        # process every Nth frame
DRAW_SIDE     = "right"  # archer's draw hand
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

LANDMARK_IDX = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_foot_index": 31, "right_foot_index": 32,
}


# ── helpers ──────────────────────────────────────────────────────────────────

def angle3(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def horiz_angle(a, b):
    """Angle of line a->b from horizontal, in [0, 90] degrees."""
    d = b - a
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-9)))


def signed_tilt(a, b):
    """
    Signed tilt of the line a->b from horizontal, in [-90, 90].
    Uses abs(dx) so the result is independent of left-right orientation.
    Positive = b is below a in image (image y increases downward).
    """
    d = b - a
    return float(np.degrees(np.arctan2(d[1], abs(d[0]) + 1e-9)))


def extract_frames(video_path, skip=SKIP_FRAMES, width=PROCESS_WIDTH):
    """Yield (frame_idx, resized_rgb_frame, scale_factor)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = width / orig_w
    new_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    print(f"  Video: {orig_w}px -> {width}px, {total} frames @ {fps:.0f}fps, "
          f"processing every {skip+1} frames")
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % (skip + 1) == 0:
            small = cv2.resize(frame, (width, new_h))
            yield idx, cv2.cvtColor(small, cv2.COLOR_BGR2RGB), scale
        idx += 1
    cap.release()


def get_lm(landmarks, name):
    i = LANDMARK_IDX.get(name)
    if i is None:
        return None
    lm = landmarks.landmark[i]
    if lm.visibility < 0.4:
        return None
    return np.array([lm.x, lm.y])


def compute_frame_angles(landmarks, draw_side):
    """
    Side-view camera: archer faces the camera from the side.
    - x-axis = horizontal (left-right in image)
    - y-axis = vertical (up-down in image, y increases downward)
    Both shoulders are at similar x, so shoulder tilt = y-difference.
    """
    bow = "left" if draw_side == "right" else "right"
    D = draw_side; B = bow
    out = {}

    ds = get_lm(landmarks, f"{D}_shoulder")
    de = get_lm(landmarks, f"{D}_elbow")
    dw = get_lm(landmarks, f"{D}_wrist")
    bs = get_lm(landmarks, f"{B}_shoulder")
    be = get_lm(landmarks, f"{B}_elbow")
    bw = get_lm(landmarks, f"{B}_wrist")
    ls = get_lm(landmarks, "left_shoulder")
    rs = get_lm(landmarks, "right_shoulder")
    lh = get_lm(landmarks, "left_hip")
    rh = get_lm(landmarks, "right_hip")
    nose = get_lm(landmarks, "nose")
    le = get_lm(landmarks, "left_ear")
    re = get_lm(landmarks, "right_ear")

    # Draw elbow angle (shoulder-elbow-wrist) — fully reliable from side view
    if all(x is not None for x in [ds, de, dw]):
        out["draw_elbow_angle"] = angle3(ds, de, dw)

    # Bow elbow angle
    if all(x is not None for x in [bs, be, bw]):
        out["bow_elbow_angle"] = angle3(bs, be, bw)

    # Shoulder tilt (side view): y-difference of the two shoulders.
    # In image coords y increases downward, so positive = draw shoulder lower.
    # Multiply by 100 to get a readable percentage of frame height.
    if ls is not None and rs is not None:
        out["shoulder_tilt_pct"] = float((rs[1] - ls[1]) * 100)

    # Torso lean: angle of hip-midpoint -> shoulder-midpoint from vertical.
    # Positive = leaning toward target (forward), negative = leaning back.
    if ls is not None and rs is not None and lh is not None and rh is not None:
        sh_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        d = sh_mid - hip_mid  # upward vector (y negative in image = up)
        # Angle from vertical: arctan2(dx, -dy) — negative dy because up = neg y
        out["torso_lean"] = float(np.degrees(np.arctan2(d[0], -d[1] + 1e-9)))

    # Draw shoulder elevation from horizontal (side view: reliable)
    if ds is not None and de is not None:
        out["draw_shoulder_elevation"] = horiz_angle(ds, de)

    # Bow shoulder elevation
    if bs is not None and be is not None:
        out["bow_shoulder_depression"] = horiz_angle(bs, be)

    # Head tilt (side view): y-difference of ears.
    # Positive = right ear lower (head tilted right in image)
    if le is not None and re is not None:
        out["head_tilt_pct"] = float((re[1] - le[1]) * 100)
        mid_ear = (le + re) / 2
        if nose is not None:
            # Nose x relative to mid-ear x: positive = nose in front (toward target side)
            out["head_forward_lean"] = float((nose[0] - mid_ear[0]) * 100)

    # Draw extension: normalized draw wrist to bow shoulder distance
    if dw is not None and bs is not None:
        out["draw_extension"] = float(np.linalg.norm(dw - bs))

    # Anchor consistency: draw wrist y relative to nose y (should be near 0)
    if dw is not None and nose is not None:
        out["anchor_height_diff"] = float((dw[1] - nose[1]) * 100)

    return out


def run_pose_extraction(video_path, draw_side=DRAW_SIDE, label=""):
    """Run MediaPipe on video, return list of (frame_idx, angles_dict)."""
    print(f"\n[{label}] Extracting poses from {os.path.basename(video_path)}")
    results = []
    t0 = time.time()
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for frame_idx, rgb, scale in extract_frames(video_path):
            res = pose.process(rgb)
            if res.pose_landmarks:
                angles = compute_frame_angles(res.pose_landmarks, draw_side)
                if angles:
                    results.append((frame_idx, angles))

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} frames with pose in {elapsed:.1f}s")
    return results


# ── shot detection ────────────────────────────────────────────────────────────

def detect_shots_from_frames(frame_angles, fps=30.0, skip=SKIP_FRAMES):
    """
    Detect individual shots using draw_extension signal:
      - Extension rises to a peak (full draw)
      - Then drops suddenly (release)
    Returns list of dicts with phase frame indices.
    """
    if not frame_angles:
        return []

    idxs = [f for f, _ in frame_angles]
    exts = [a.get("draw_extension", 0) for _, a in frame_angles]

    # Smooth
    w = 7
    smooth = np.convolve(exts, np.ones(w) / w, mode="same")
    norm   = smooth / (max(smooth) + 1e-9)

    shots = []
    i = 0
    n = len(norm)
    RISE_THRESH  = 0.55   # start of draw
    PEAK_THRESH  = 0.82   # full draw
    DROP_THRESH  = 0.45   # post-release drop

    while i < n - 5:
        # Look for rising phase
        if norm[i] < RISE_THRESH:
            i += 1
            continue

        # Rising: find the peak
        peak_i = i
        j = i
        while j < min(i + 150, n) and norm[j] >= norm[peak_i] - 0.03:
            if norm[j] > norm[peak_i]:
                peak_i = j
            j += 1

        if norm[peak_i] < PEAK_THRESH:
            i = j
            continue

        # Peak found: look for the drop (release)
        release_i = None
        k = peak_i
        while k < min(peak_i + 80, n):
            if norm[k] < DROP_THRESH:
                release_i = k
                break
            k += 1

        if release_i is None:
            i = peak_i + 1
            continue

        # Draw start: go back from peak to where extension first passed RISE_THRESH
        draw_start = peak_i
        m = peak_i
        while m > max(0, peak_i - 100):
            if norm[m] < RISE_THRESH:
                draw_start = m + 1
                break
            m -= 1

        follow_end = min(release_i + 25, n - 1)

        shots.append({
            "draw_start_fi": draw_start,      # index into frame_angles list
            "full_draw_fi":  peak_i,
            "release_fi":    release_i,
            "follow_end_fi": follow_end,
            "draw_start_frame":  idxs[draw_start],
            "full_draw_frame":   idxs[peak_i],
            "release_frame":     idxs[release_i],
        })
        i = follow_end + 1

    print(f"  Detected {len(shots)} shots")
    return shots


def get_full_draw_angles(frame_angles, shot, window=5):
    """Average angles over frames around the full draw peak."""
    peak = shot["full_draw_fi"]
    start = max(0, peak - window)
    end   = min(len(frame_angles) - 1, peak + window)
    subset = [frame_angles[k][1] for k in range(start, end + 1)]
    if not subset:
        return {}
    all_keys = set().union(*subset)
    result = {}
    for k in all_keys:
        vals = [d[k] for d in subset if k in d]
        if vals:
            result[k] = float(np.median(vals))
    return result


# ── calibration (train) ───────────────────────────────────────────────────────

def calibrate(video_path, draw_side=DRAW_SIDE):
    """
    Process training video.
    Returns personalized baseline = median angles across all detected shots.
    """
    frame_angles = run_pose_extraction(video_path, draw_side, label="TRAIN")
    shots = detect_shots_from_frames(frame_angles)

    if not shots:
        print("  WARNING: No shots detected in training video. Using all frames.")
        all_angles = [a for _, a in frame_angles]
        shots_angles = [all_angles] if all_angles else []
    else:
        shots_angles = [get_full_draw_angles(frame_angles, s) for s in shots]

    shots_angles = [a for a in shots_angles if a]
    if not shots_angles:
        print("  ERROR: Could not extract angle data from training video.")
        return None, shots

    print(f"\n  Building personalized baseline from {len(shots_angles)} shots...")
    all_keys = set().union(*shots_angles)
    baseline = {}
    for k in all_keys:
        vals = [d[k] for d in shots_angles if k in d]
        if vals:
            baseline[k] = {
                "mean":   round(float(np.mean(vals)),   2),
                "median": round(float(np.median(vals)), 2),
                "std":    round(float(np.std(vals)),    2),
                "min":    round(float(np.min(vals)),    2),
                "max":    round(float(np.max(vals)),    2),
                "n":      len(vals),
            }
            print(f"    {k:35s} median={baseline[k]['median']:7.2f}  std={baseline[k]['std']:5.2f}  (n={len(vals)})")

    with open(BASELINE_FILE, "w") as f:
        json.dump({"draw_side": draw_side, "angles": baseline, "shot_count": len(shots_angles)}, f, indent=2)
    print(f"\n  Baseline saved -> {BASELINE_FILE}")
    return baseline, shots


# ── testing ───────────────────────────────────────────────────────────────────

SEVERITY_THRESHOLDS = {
    "draw_elbow_angle":        (8.0,  15.0),
    "bow_elbow_angle":         (8.0,  15.0),
    "shoulder_tilt_pct":       (1.5,   3.0),   # % of frame height
    "torso_lean":              (3.0,   7.0),   # degrees
    "draw_shoulder_elevation": (5.0,  10.0),
    "bow_shoulder_depression": (5.0,  10.0),
    "head_tilt_pct":           (1.5,   3.0),
    "head_forward_lean":       (2.5,   5.0),
    "anchor_height_diff":      (2.0,   4.0),
}

FIELD_LABELS = {
    "draw_elbow_angle":        "Draw elbow angle",
    "bow_elbow_angle":         "Bow arm elbow angle",
    "shoulder_tilt_pct":       "Shoulder level",
    "torso_lean":              "Torso lean",
    "draw_shoulder_elevation": "Draw shoulder elevation",
    "bow_shoulder_depression": "Bow shoulder position",
    "head_tilt_pct":           "Head tilt",
    "head_forward_lean":       "Head forward lean",
    "anchor_height_diff":      "Anchor height",
}

FEEDBACK = {
    "draw_elbow_angle": (
        "Draw elbow too bent — use back muscles, not just arm.",
        "Draw elbow hyperextended — engage back muscles and relax arm slightly.",
    ),
    "bow_elbow_angle": (
        "Bow arm too bent — risk of string slap. Extend more.",
        "Bow arm over-extended. A slight soft elbow is safer.",
    ),
    "shoulder_tilt_pct": (
        "Draw-side shoulder too high. Level your shoulders.",
        "Bow-side shoulder too high. Level your shoulders.",
    ),
    "torso_lean": (
        "Leaning back away from target. Stand upright or lean slightly toward target.",
        "Leaning too far forward toward target. Keep torso upright.",
    ),
    "draw_shoulder_elevation": (
        "Draw shoulder too low — may lack power transfer.",
        "Draw shoulder raised / shrugging. Press it down for cleaner form.",
    ),
    "bow_shoulder_depression": (
        "Bow shoulder too low — stabilise the bow arm platform.",
        "Bow shoulder raised. Press it down for a stable bow arm.",
    ),
    "head_tilt_pct": (
        "Head tilted away from target. Keep head upright.",
        "Head tilted toward target. Keep head upright.",
    ),
    "head_forward_lean": (
        "Head leaning back — bring chin forward to meet the string.",
        "Head too far forward. Maintain a natural upright head position.",
    ),
    "anchor_height_diff": (
        "Draw hand anchor is too high relative to face — lower the anchor point.",
        "Draw hand anchor is too low relative to face — raise the anchor point.",
    ),
}


def severity_label(dev, field):
    lo, hi = SEVERITY_THRESHOLDS.get(field, (5.0, 10.0))
    if abs(dev) < lo:   return "ok"
    if abs(dev) < hi:   return "moderate"
    return "significant"


def analyze_test_shot(shot_id, angles, baseline):
    issues = []
    for field, fb in FEEDBACK.items():
        val = angles.get(field)
        ref = baseline.get(field)
        if val is None or ref is None:
            continue
        ideal = ref["median"]
        dev   = val - ideal
        sev   = severity_label(dev, field)
        if sev == "ok":
            continue
        msg = fb[1] if dev > 0 else fb[0]
        issues.append({
            "field":    field,
            "label":    FIELD_LABELS.get(field, field),
            "severity": sev,
            "measured": round(val, 2),
            "ideal":    ideal,
            "deviation":round(dev, 2),
            "feedback": msg,
        })
    issues.sort(key=lambda x: 0 if x["severity"] == "significant" else 1)
    return issues


def compare_two_shots(id_a, angles_a, id_b, angles_b):
    """Diff two shots: what changed?"""
    changed = []
    CHANGE_MIN = {
        "draw_elbow_angle":        5.0,
        "bow_elbow_angle":         5.0,
        "shoulder_tilt_pct":       1.5,
        "torso_lean":              3.0,
        "draw_shoulder_elevation": 4.0,
        "bow_shoulder_depression": 4.0,
        "head_tilt_pct":           1.5,
        "head_forward_lean":       2.5,
        "anchor_height_diff":      2.0,
    }
    for field, thresh in CHANGE_MIN.items():
        a = angles_a.get(field)
        b = angles_b.get(field)
        if a is None or b is None:
            continue
        delta = b - a
        if abs(delta) >= thresh:
            changed.append({
                "field": field,
                "label": FIELD_LABELS.get(field, field),
                "shot_a": round(a, 2),
                "shot_b": round(b, 2),
                "delta":  round(delta, 2),
            })
    changed.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return changed


def run_test(video_path, baseline, draw_side=DRAW_SIDE):
    frame_angles = run_pose_extraction(video_path, draw_side, label="TEST")
    shots = detect_shots_from_frames(frame_angles)

    if not shots:
        print("  WARNING: No shots detected in test video.")
        return

    shot_angle_list = []
    for s in shots:
        a = get_full_draw_angles(frame_angles, s)
        shot_angle_list.append(a)

    print("\n" + "="*70)
    print("  TEST RESULTS")
    print("="*70)

    report = {"shots": [], "comparisons": []}

    for i, (shot, angles) in enumerate(zip(shots, shot_angle_list)):
        issues = analyze_test_shot(i + 1, angles, baseline)
        report["shots"].append({"shot_id": i + 1, "angles": angles, "issues": issues})

        print(f"\n  SHOT {i+1}  (full-draw @ frame {shot['full_draw_frame']})")
        print(f"  {'-'*50}")

        if not issues:
            print("    Form looks consistent with training baseline.")
        for iss in issues:
            sym = "!!" if iss["severity"] == "significant" else " ~"
            print(f"  {sym} [{iss['severity'].upper():11s}] {iss['label']}: "
                  f"{iss['measured']:.1f} (baseline {iss['ideal']:.1f}, Δ{iss['deviation']:+.1f})")
            print(f"       -> {iss['feedback']}")

    # Shot-to-shot comparison
    if len(shot_angle_list) >= 2:
        print(f"\n  SHOT-TO-SHOT CHANGES")
        print(f"  {'-'*50}")
        for i in range(len(shot_angle_list) - 1):
            changed = compare_two_shots(
                i + 1, shot_angle_list[i],
                i + 2, shot_angle_list[i + 1]
            )
            report["comparisons"].append({
                "from": i + 1, "to": i + 2, "changes": changed
            })
            print(f"\n  Shot {i+1} -> Shot {i+2}:")
            if not changed:
                print("    No significant changes.")
            for c in changed:
                arrow = "↑" if c["delta"] > 0 else "↓"
                print(f"    {arrow} {c['label']}: {c['shot_a']:.1f} -> {c['shot_b']:.1f}  (Δ{c['delta']:+.1f})")

    # Save JSON report
    report_path = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved -> {report_path}")

    return report


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*70)
    print("  ARCHERY SHOT ANALYZER")
    print("="*70)

    # Check if baseline already exists (skip retraining)
    if os.path.exists(BASELINE_FILE):
        print(f"\n[TRAIN] Loading existing baseline from {BASELINE_FILE}")
        with open(BASELINE_FILE) as f:
            data = json.load(f)
        baseline = data["angles"]
        DRAW_SIDE = data.get("draw_side", DRAW_SIDE)
        print(f"  Loaded {len(baseline)} angle fields, draw_side={DRAW_SIDE}, "
              f"trained on {data.get('shot_count', '?')} shots")
    else:
        baseline, _ = calibrate(TRAIN_VIDEO, DRAW_SIDE)
        if baseline is None:
            sys.exit(1)

    print("\n" + "="*70)
    print("  TESTING")
    print("="*70)
    run_test(TEST_VIDEO, baseline, DRAW_SIDE)
