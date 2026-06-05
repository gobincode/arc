"""
Archery Shot Analyzer — RTMPose Wholebody (133 keypoints)
Train on video 1, test on video 2.

Run on Colab:
    1. python colab_setup.py
    2. Edit TRAIN_VIDEO / TEST_VIDEO paths below
    3. python rtmpose_train_test.py

RTMPose wholebody gives 133 keypoints:
  0-16  : body  (COCO 17)
  17-22 : feet  (6)
  23-90 : face  (68 — includes jawline, nose tip, eyes, ears)
  91-111: left hand  (21 — wrist + 5 fingers x 4 joints)
  112-132: right hand (21)
"""

import os, sys, json, time
import numpy as np
import cv2

# ── CONFIG ────────────────────────────────────────────────────────────────────
import os as _os

# Base dir: works on Colab (/content/arc) and locally
BASE_DIR      = _os.path.dirname(_os.path.abspath(__file__))
TRAIN_VIDEO   = _os.path.join(BASE_DIR, "20250913_175626.mp4")
TEST_VIDEO    = _os.path.join(BASE_DIR, "20250913_180517.mp4")
BASELINE_FILE = _os.path.join(BASE_DIR, "archer_baseline.json")
OUTPUT_DIR    = _os.path.join(BASE_DIR, "output")
DRAW_SIDE     = "right"    # archer's draw hand
PROCESS_WIDTH = 960
SKIP_FRAMES   = 2          # process every 3rd frame for speed
USE_MOTIONBERT = True      # set False to fall back to 2D-only angles
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── KEYPOINT INDEX MAP ────────────────────────────────────────────────────────

# Body (COCO 17)
KP = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
    # Feet
    "left_big_toe": 17,  "right_big_toe": 20,
    "left_heel": 19,     "right_heel": 22,
    # Face landmarks (subset)
    "chin": 23,          # jawline point 0 (leftmost jaw)
    "chin_center": 30,   # jawline midpoint (chin tip) ~ index 30 in 68-pt face = kp 53
    "face_nose_tip": 53, # nose tip in 68-pt face model
    # Left hand (91-111): wrist=91, thumb=92-95, index=96-99, middle=100-103, ring=104-107, pinky=108-111
    "lh_wrist": 91,
    "lh_thumb_tip": 95,  "lh_index_tip": 99,
    "lh_middle_tip": 103,"lh_ring_tip": 107, "lh_pinky_tip": 111,
    "lh_index_mcp": 96,  "lh_middle_mcp": 100,
    # Right hand (112-132)
    "rh_wrist": 112,
    "rh_thumb_tip": 116, "rh_index_tip": 120,
    "rh_middle_tip": 124,"rh_ring_tip": 128, "rh_pinky_tip": 132,
    "rh_index_mcp": 117, "rh_middle_mcp": 121,
}

# Convenience: draw/bow hand aliases based on DRAW_SIDE
def draw_kp(name):
    prefix = "rh" if DRAW_SIDE == "right" else "lh"
    return KP.get(f"{prefix}_{name}")

def bow_kp(name):
    prefix = "lh" if DRAW_SIDE == "right" else "rh"
    return KP.get(f"{prefix}_{name}")


# ── INFERENCER SETUP ──────────────────────────────────────────────────────────

def _download_models():
    """Download RTMPose wholebody ONNX models from HuggingFace."""
    from huggingface_hub import hf_hub_download
    print("  Downloading models from HuggingFace...")
    det  = hf_hub_download(repo_id="yzd-v/DWPose", filename="yolox_l.onnx")
    pose = hf_hub_download(repo_id="yzd-v/DWPose", filename="dw-ll_ucoco_384.onnx")
    print(f"  det : {det}")
    print(f"  pose: {pose}")
    return det, pose


def build_inferencer():
    from rtmlib import Wholebody
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading RTMPose wholebody model on {device}...")

    det_path, pose_path = _download_models()
    model = Wholebody(
        det=det_path,
        pose=pose_path,
        backend="onnxruntime",
        device=device,
    )
    print("Model loaded.")
    return model


# ── KEYPOINT EXTRACTION ───────────────────────────────────────────────────────

def get_kp(kps, scores, idx, threshold=0.3):
    """Return (x, y) normalized [0,1] for keypoint idx, or None if low confidence."""
    if idx is None or idx >= len(kps):
        return None
    if scores[idx] < threshold:
        return None
    return np.array([kps[idx][0], kps[idx][1]], dtype=float)


def get_kp_px(kps, scores, idx, w, h, threshold=0.3):
    pt = get_kp(kps, scores, idx, threshold)
    if pt is None:
        return None
    return (int(pt[0]), int(pt[1]))   # already in pixel coords from mmpose


def normalize_kps(kps, w, h):
    """Normalize pixel keypoints to [0,1]."""
    norm = kps.copy().astype(float)
    norm[:, 0] /= w
    norm[:, 1] /= h
    return norm


def kps_bbox(kps, scores, threshold=0.3):
    """Get bounding box (x1,y1,x2,y2) from visible keypoints."""
    valid = kps[scores > threshold]
    if len(valid) == 0:
        return None
    return np.array([valid[:,0].min(), valid[:,1].min(),
                     valid[:,0].max(), valid[:,1].max()])


def bbox_iou(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_area(bbox):
    if bbox is None:
        return 0.0
    return max(0, bbox[2]-bbox[0]) * max(0, bbox[3]-bbox[1])


def pick_archer(kps_all, scores_all, prev_bbox=None):
    """
    Pick the archer from multiple detected persons.
    - First frame: pick the person with the largest bounding box.
    - Subsequent frames: pick the person with highest IoU to previous bbox.
      Falls back to largest bbox if IoU is too low (person left frame).
    """
    if kps_all is None or len(kps_all) == 0:
        return None, None, None

    bboxes = [kps_bbox(kps_all[i], scores_all[i]) for i in range(len(kps_all))]

    if prev_bbox is None:
        # First frame: largest person
        idx = max(range(len(bboxes)), key=lambda i: bbox_area(bboxes[i]))
    else:
        ious = [bbox_iou(prev_bbox, b) for b in bboxes]
        best_iou = max(ious)
        if best_iou > 0.15:
            idx = int(np.argmax(ious))
        else:
            # Lost track — fall back to largest
            idx = max(range(len(bboxes)), key=lambda i: bbox_area(bboxes[i]))

    return kps_all[idx], scores_all[idx], bboxes[idx]


# ── ANGLE MATH ────────────────────────────────────────────────────────────────

def angle3(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def horiz_angle(a, b):
    d = b - a
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-9)))

def signed_tilt(a, b):
    """Tilt from horizontal in [-90,90], using abs(dx) for side-view robustness."""
    d = b - a
    return float(np.degrees(np.arctan2(d[1], abs(d[0]) + 1e-9)))

def finger_curl(mcp, tip):
    """Straightness of a finger: angle between MCP-base and MCP-tip vectors.
    0 = fully straight, 90 = fully curled."""
    if mcp is None or tip is None:
        return None
    d = tip - mcp
    # angle from pointing straight (horizontal or downward)
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]) + 1e-9)))


def compute_angles(norm_kps, scores, draw_side=DRAW_SIDE):
    """
    Compute all archery-relevant angles from 133 wholebody keypoints.
    norm_kps: (133, 2) in [0,1] normalized coords.
    """
    bow_side = "left" if draw_side == "right" else "right"
    D = draw_side; B = bow_side
    out = {}

    def g(name):
        return get_kp(norm_kps, scores, KP.get(name))

    # ── Body angles ──────────────────────────────────────────────────────────
    ds = g(f"{D}_shoulder"); de = g(f"{D}_elbow"); dw = g(f"{D}_wrist")
    bs = g(f"{B}_shoulder"); be = g(f"{B}_elbow"); bw = g(f"{B}_wrist")
    ls = g("left_shoulder"); rs = g("right_shoulder")
    lh = g("left_hip");      rh = g("right_hip")
    nose = g("nose")
    le = g("left_ear");      re = g("right_ear")

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

    # ── Anchor precision (face landmarks) ────────────────────────────────────
    chin = g("chin_center")   # nose tip in 68-pt face = kp 53
    if dw is not None and chin is not None:
        out["anchor_to_chin_dist"] = float(np.linalg.norm(dw - chin) * 100)
    if dw is not None and nose is not None:
        out["anchor_height_diff"] = float((dw[1] - nose[1]) * 100)

    # ── Draw hand fingers (string grip) ──────────────────────────────────────
    dh_prefix = "rh" if draw_side == "right" else "lh"

    idx_mcp = g(f"{dh_prefix}_index_mcp");   idx_tip = g(f"{dh_prefix}_index_tip")
    mid_mcp = g(f"{dh_prefix}_middle_mcp");  mid_tip = g(f"{dh_prefix}_middle_tip")

    if idx_mcp is not None and idx_tip is not None:
        out["draw_index_curl"] = finger_curl(idx_mcp, idx_tip)
    if mid_mcp is not None and mid_tip is not None:
        out["draw_middle_curl"] = finger_curl(mid_mcp, mid_tip)

    # Average draw finger curl (how deeply fingers are hooked on string)
    curls = [out.get("draw_index_curl"), out.get("draw_middle_curl")]
    curls = [c for c in curls if c is not None]
    if curls:
        out["draw_finger_curl_avg"] = float(np.mean(curls))

    # ── Bow hand grip ─────────────────────────────────────────────────────────
    bh_prefix = "lh" if draw_side == "right" else "rh"
    bi_mcp = g(f"{bh_prefix}_index_mcp"); bi_tip = g(f"{bh_prefix}_index_tip")
    bm_mcp = g(f"{bh_prefix}_middle_mcp"); bm_tip = g(f"{bh_prefix}_middle_tip")
    bthumb_tip = g(f"{bh_prefix}_thumb_tip")

    if bi_mcp is not None and bi_tip is not None:
        out["bow_index_curl"] = finger_curl(bi_mcp, bi_tip)

    # Bow thumb tension: thumb tip distance from bow wrist
    if bw is not None and bthumb_tip is not None:
        out["bow_thumb_extension"] = float(np.linalg.norm(bthumb_tip - bw) * 100)

    return out


# ── VIDEO FRAME EXTRACTION ────────────────────────────────────────────────────

def iter_frames(video_path, skip=SKIP_FRAMES, width=PROCESS_WIDTH):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = width / orig_w
    new_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    print(f"  {os.path.basename(video_path)}: {orig_w}px -> {width}px, "
          f"{total} frames @ {fps:.0f}fps, every {skip+1} frames")
    cap.release()
    return total, fps, new_h, scale


def run_inference(video_path, inferencer, skip=SKIP_FRAMES, width=PROCESS_WIDTH):
    """
    Run RTMPose on video frames via rtmlib.
    Returns list of (frame_idx, norm_kps (133,2), scores (133,)).
    """
    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    scale  = width / orig_w
    new_h  = int(orig_h * scale)

    results = []
    t0 = time.time()
    fi = 0
    prev_bbox = None   # tracks archer across frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if fi % (skip + 1) == 0:
            small = cv2.resize(frame, (width, new_h))

            kps_all, scores_all = inferencer(small)
            kps, scores, prev_bbox = pick_archer(kps_all, scores_all, prev_bbox)

            if kps is not None:
                norm = kps.copy().astype(float)
                norm[:, 0] /= width
                norm[:, 1] /= new_h
                results.append((fi, norm, scores))

        fi += 1
        if fi % 300 == 0:
            pct = fi / total * 100
            eta = ((time.time() - t0) / fi) * (total - fi)
            print(f"  {fi}/{total} ({pct:.0f}%)  ETA {eta:.0f}s")

    cap.release()
    print(f"  Done: {len(results)} frames with pose in {time.time()-t0:.1f}s")
    results = smooth_keypoints(results)
    return results, fps, new_h


def smooth_keypoints(pose_frames, sigma=3.0):
    """
    Gaussian-smooth each keypoint's x and y trajectory across frames.
    Eliminates per-frame prediction jitter for still subjects.
    sigma=3 means ~7 frame smoothing radius at 30fps.
    """
    from scipy.ndimage import gaussian_filter1d

    if len(pose_frames) < 5:
        return pose_frames

    n   = len(pose_frames)
    fis = [f[0] for f in pose_frames]
    kps = np.array([f[1] for f in pose_frames])    # (N, 133, 2)
    scs = np.array([f[2] for f in pose_frames])    # (N, 133)

    # Smooth x and y for each of the 133 joints independently
    kps_smooth = kps.copy()
    for j in range(kps.shape[1]):
        # Only smooth joints that are consistently visible
        visible = scs[:, j] > 0.3
        if visible.sum() < 5:
            continue
        kps_smooth[:, j, 0] = gaussian_filter1d(kps[:, j, 0], sigma=sigma)
        kps_smooth[:, j, 1] = gaussian_filter1d(kps[:, j, 1], sigma=sigma)

    return [(fis[i], kps_smooth[i], scs[i]) for i in range(n)]


# ── SHOT DETECTION ────────────────────────────────────────────────────────────

def detect_shots(pose_frames, draw_side=DRAW_SIDE):
    bow_side = "left" if draw_side == "right" else "right"
    exts = []
    for fi, norm_kps, scores in pose_frames:
        dw = get_kp(norm_kps, scores, KP.get(f"{draw_side}_wrist"))
        bs = get_kp(norm_kps, scores, KP.get(f"{bow_side}_shoulder"))
        ext = float(np.linalg.norm(dw - bs)) if dw is not None and bs is not None else 0.0
        exts.append(ext)

    w = 7
    smooth = np.convolve(exts, np.ones(w) / w, mode="same")
    norm   = smooth / (max(smooth) + 1e-9)
    shots, i, n = [], 0, len(norm)

    while i < n - 5:
        if norm[i] < 0.55: i += 1; continue
        peak_i, peak_v = i, 0.0
        j = i
        while j < min(i + 150, n):
            if norm[j] > peak_v: peak_v, peak_i = norm[j], j
            j += 1
        if peak_v < 0.82: i = j; continue
        release_i = None
        k = peak_i
        while k < min(peak_i + 80, n):
            if norm[k] < 0.45: release_i = k; break
            k += 1
        if release_i is None: i = peak_i + 1; continue
        draw_start = peak_i
        m = peak_i
        while m > max(0, peak_i - 100):
            if norm[m] < 0.55: draw_start = m + 1; break
            m -= 1
        follow_end = min(release_i + 25, n - 1)
        shots.append({
            "draw_start": draw_start, "full_draw": peak_i,
            "release": release_i,     "follow_end": follow_end,
            "full_draw_frame": pose_frames[peak_i][0],
        })
        i = follow_end + 1

    print(f"  Detected {len(shots)} shots")
    return shots


def get_full_draw_angles(pose_frames, shot, draw_side=DRAW_SIDE, window=20, poses3d=None):
    peak  = shot["full_draw"]
    start = max(0, peak - window)
    end   = min(len(pose_frames) - 1, peak + window)

    angle_list = []
    for k in range(start, end + 1):
        fi, norm_kps, scores = pose_frames[k]

        if USE_MOTIONBERT and poses3d and fi in poses3d:
            # Body angles from 3D (accurate depth)
            from motionbert import compute_3d_angles
            body_a = compute_3d_angles(poses3d[fi], draw_side)
            # Hand/face angles from 2D RTMPose (still reliable from side view)
            hand_a = compute_angles(norm_kps, scores, draw_side)
            # Merge: 3D body overrides 2D body fields
            a = {**hand_a, **body_a}
        else:
            a = compute_angles(norm_kps, scores, draw_side)

        if a:
            angle_list.append(a)

    if not angle_list:
        return {}
    all_keys = set().union(*angle_list)
    return {k: float(np.median([d[k] for d in angle_list if k in d])) for k in all_keys}


# ── CALIBRATION ───────────────────────────────────────────────────────────────

def _lift_3d(pose_frames):
    """Run MotionBERT 3D lifting if enabled."""
    if not USE_MOTIONBERT:
        return None
    try:
        from motionbert import MotionBERTLifter
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lifter = MotionBERTLifter(device=device)
        return lifter.lift(pose_frames)
    except Exception as e:
        print(f"  MotionBERT unavailable ({e}), falling back to 2D angles.")
        return None


def calibrate(video_path, inferencer, draw_side=DRAW_SIDE):
    print(f"\n[TRAIN] {os.path.basename(video_path)}")
    pose_frames, fps, _ = run_inference(video_path, inferencer)
    shots = detect_shots(pose_frames, draw_side)

    if not shots:
        print("  WARNING: no shots detected, using all frames as one shot.")
        mid = len(pose_frames) // 2
        shots = [{"draw_start": 0, "full_draw": mid,
                  "release": mid + 5, "follow_end": len(pose_frames) - 1,
                  "full_draw_frame": pose_frames[mid][0]}]

    poses3d = _lift_3d(pose_frames)
    shots_angles = [get_full_draw_angles(pose_frames, s, draw_side, poses3d=poses3d) for s in shots]
    shots_angles = [a for a in shots_angles if a]

    print(f"\n  Building personalized baseline from {len(shots_angles)} shots...")
    all_keys = set().union(*shots_angles)
    baseline = {}
    for k in sorted(all_keys):
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
            print(f"    {k:35s}  median={baseline[k]['median']:8.2f}  "
                  f"std={baseline[k]['std']:5.2f}  (n={len(vals)})")

    with open(BASELINE_FILE, "w") as f:
        json.dump({"draw_side": draw_side, "angles": baseline,
                   "shot_count": len(shots_angles)}, f, indent=2)
    print(f"\n  Baseline saved -> {BASELINE_FILE}")
    return baseline


# ── FEEDBACK ENGINE ───────────────────────────────────────────────────────────

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
    "anchor_to_chin_dist":     (2.0,   4.0),
    "draw_finger_curl_avg":    (5.0,  10.0),
    "bow_index_curl":          (5.0,  10.0),
    "bow_thumb_extension":     (2.0,   4.0),
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
    "anchor_height_diff":      "Anchor height (nose ref)",
    "anchor_to_chin_dist":     "Anchor-to-chin distance",
    "draw_finger_curl_avg":    "Draw finger curl",
    "bow_index_curl":          "Bow index finger curl",
    "bow_thumb_extension":     "Bow thumb extension",
}

FEEDBACK = {
    "draw_elbow_angle":        ("Draw elbow too bent — use back muscles.",
                                "Draw elbow hyperextended — relax arm, engage back."),
    "bow_elbow_angle":         ("Bow arm too bent — string slap risk. Extend more.",
                                "Bow arm over-extended. Slight soft elbow is safer."),
    "shoulder_tilt_pct":       ("Draw shoulder too high. Level your shoulders.",
                                "Bow shoulder too high. Level your shoulders."),
    "torso_lean":              ("Leaning back. Stand upright or slightly toward target.",
                                "Leaning too far forward. Keep torso upright."),
    "draw_shoulder_elevation": ("Draw shoulder too low.",
                                "Draw shoulder shrugging up — press it down."),
    "bow_shoulder_depression": ("Bow shoulder too low.",
                                "Bow shoulder raised — press it down."),
    "head_tilt_pct":           ("Head tilted away from target.",
                                "Head tilted toward target. Keep it upright."),
    "head_forward_lean":       ("Head leaning back — bring chin to string.",
                                "Head too far forward. Maintain natural position."),
    "anchor_height_diff":      ("Draw hand too high vs face — lower anchor.",
                                "Draw hand too low vs face — raise anchor."),
    "anchor_to_chin_dist":     ("Draw hand not reaching chin — inconsistent anchor.",
                                "Draw hand past chin — over-drawing to face."),
    "draw_finger_curl_avg":    ("Draw fingers too straight — weak string grip.",
                                "Draw fingers over-curled — tension in hand, relax."),
    "bow_index_curl":          ("Bow index too straight — grip may be too loose.",
                                "Bow index over-curled — may torque the bow."),
    "bow_thumb_extension":     ("Bow thumb too close — grip tension risk.",
                                "Bow thumb too extended — low pressure grip."),
}


def severity(field, value, baseline):
    ref = baseline.get(field)
    if ref is None: return "ok"
    dev = abs(value - ref["median"])
    lo, hi = SEVERITY_THRESHOLDS.get(field, (5.0, 10.0))
    if dev < lo:  return "ok"
    if dev < hi:  return "moderate"
    return "significant"


def analyze_shot(shot_id, angles, baseline):
    issues = []
    for field, (fb_low, fb_high) in FEEDBACK.items():
        val = angles.get(field)
        ref = baseline.get(field)
        if val is None or ref is None: continue
        dev = val - ref["median"]
        sev = severity(field, val, baseline)
        if sev == "ok": continue
        issues.append({
            "field": field, "label": FIELD_LABELS.get(field, field),
            "severity": sev, "measured": round(val, 2),
            "ideal": ref["median"], "deviation": round(dev, 2),
            "feedback": fb_high if dev > 0 else fb_low,
        })
    issues.sort(key=lambda x: 0 if x["severity"] == "significant" else 1)
    return issues


def compare_shots(id_a, ang_a, id_b, ang_b):
    CHANGE_MIN = {
        "draw_elbow_angle": 5.0, "bow_elbow_angle": 5.0,
        "shoulder_tilt_pct": 1.5, "torso_lean": 3.0,
        "draw_shoulder_elevation": 4.0, "bow_shoulder_depression": 4.0,
        "head_tilt_pct": 1.5, "head_forward_lean": 2.5,
        "anchor_height_diff": 2.0, "anchor_to_chin_dist": 2.0,
        "draw_finger_curl_avg": 4.0, "bow_index_curl": 4.0,
    }
    changed = []
    for field, thresh in CHANGE_MIN.items():
        a = ang_a.get(field); b = ang_b.get(field)
        if a is None or b is None: continue
        delta = b - a
        if abs(delta) >= thresh:
            changed.append({
                "field": field, "label": FIELD_LABELS.get(field, field),
                "shot_a": round(a, 2), "shot_b": round(b, 2), "delta": round(delta, 2),
            })
    changed.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return changed


# ── TEST RUN ──────────────────────────────────────────────────────────────────

def run_test(video_path, baseline, inferencer, draw_side=DRAW_SIDE):
    print(f"\n[TEST] {os.path.basename(video_path)}")
    pose_frames, fps, _ = run_inference(video_path, inferencer)
    shots = detect_shots(pose_frames, draw_side)

    if not shots:
        print("  No shots detected."); return

    poses3d = _lift_3d(pose_frames)
    shot_angles = [get_full_draw_angles(pose_frames, s, draw_side, poses3d=poses3d) for s in shots]

    print("\n" + "="*70)
    print("  TEST RESULTS")
    print("="*70)

    report = {"shots": [], "comparisons": []}

    for i, (shot, angles) in enumerate(zip(shots, shot_angles)):
        issues = analyze_shot(i + 1, angles, baseline)
        report["shots"].append({"shot_id": i+1, "angles": angles, "issues": issues})

        print(f"\n  SHOT {i+1}  (full-draw @ frame {shot['full_draw_frame']})")
        print(f"  {'-'*55}")
        if not issues:
            print("    Consistent with training baseline.")
        for iss in issues:
            sym = "!!" if iss["severity"] == "significant" else " ~"
            print(f"  {sym} [{iss['severity'].upper():11s}] {iss['label']}: "
                  f"{iss['measured']:.1f}  (baseline {iss['ideal']:.1f}, Δ{iss['deviation']:+.1f})")
            print(f"       -> {iss['feedback']}")

    if len(shot_angles) >= 2:
        print(f"\n  SHOT-TO-SHOT CHANGES")
        print(f"  {'-'*55}")
        for i in range(len(shot_angles) - 1):
            changed = compare_shots(i+1, shot_angles[i], i+2, shot_angles[i+1])
            report["comparisons"].append({"from": i+1, "to": i+2, "changes": changed})
            print(f"\n  Shot {i+1} -> Shot {i+2}:")
            if not changed:
                print("    No significant changes.")
            for c in changed:
                arrow = "↑" if c["delta"] > 0 else "↓"
                print(f"    {arrow} {c['label']:30s} "
                      f"{c['shot_a']:.1f} -> {c['shot_b']:.1f}  (Δ{c['delta']:+.1f})")

    report_path = os.path.join(OUTPUT_DIR, "test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved -> {report_path}")

    # also save pose_frames + shot info for overlay script
    cache = {
        "shot_angles": shot_angles,
        "shots": shots,
        "baseline": baseline,
        "draw_side": draw_side,
        "fps": fps,
    }
    cache_path = os.path.join(OUTPUT_DIR, "test_cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    return report


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    inferencer = build_inferencer()

    if os.path.exists(BASELINE_FILE):
        print(f"\n[TRAIN] Loading existing baseline from {BASELINE_FILE}")
        with open(BASELINE_FILE) as f:
            data = json.load(f)
        baseline  = data["angles"]
        DRAW_SIDE = data.get("draw_side", DRAW_SIDE)
        print(f"  {len(baseline)} fields, {data.get('shot_count','?')} shots, "
              f"draw_side={DRAW_SIDE}")
    else:
        baseline = calibrate(TRAIN_VIDEO, inferencer, DRAW_SIDE)

    run_test(TEST_VIDEO, baseline, inferencer, DRAW_SIDE)
