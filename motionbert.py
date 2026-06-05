"""
MotionBERT 2D -> 3D pose lifting.

Takes a sequence of RTMPose 2D COCO-17 keypoints and lifts them to
stable 3D joint positions using MotionBERT's temporal transformer.

COCO-17 body keypoints (subset of RTMPose 133):
  0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
  5:l_shoulder  6:r_shoulder  7:l_elbow  8:r_elbow
  9:l_wrist  10:r_wrist  11:l_hip  12:r_hip
  13:l_knee  14:r_knee  15:l_ankle  16:r_ankle
"""

import os
import numpy as np

# MotionBERT config
MOTIONBERT_REPO  = "walterzhu/MotionBERT"
MOTIONBERT_FILE  = "MB_ft_h36m.bin"          # fine-tuned on Human3.6M
CLIP_LEN         = 243                        # frames MotionBERT expects
COCO17_IN_RTM    = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  # indices in RTMPose 133

# MotionBERT model dims
MB_DIM_FEAT = 512
MB_DEPTH    = 5
MB_NUM_HEADS= 8
MB_MLP_RATIO= 2


def _load_model(ckpt_path, device):
    """Load MotionBERT DSTformer from checkpoint."""
    import torch
    from motionbert_arch import DSTformer  # defined below in same file for portability

    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=MB_DIM_FEAT,
        depth=MB_DEPTH, num_heads=MB_NUM_HEADS, mlp_ratio=MB_MLP_RATIO,
        norm_layer=None, maxlen=CLIP_LEN,
    )
    state = torch.load(ckpt_path, map_location=device)
    # checkpoint may be wrapped under 'model' key
    sd = state.get("model", state)
    # remove 'module.' prefix if saved with DataParallel
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def download_motionbert_weights():
    """Download MotionBERT weights from HuggingFace."""
    from huggingface_hub import hf_hub_download
    print("Downloading MotionBERT weights from HuggingFace...")
    path = hf_hub_download(repo_id=MOTIONBERT_REPO, filename=MOTIONBERT_FILE)
    print(f"  -> {path}")
    return path


def extract_coco17(norm_kps_133, scores_133):
    """Extract COCO-17 subset from RTMPose 133-keypoint array."""
    kps17    = norm_kps_133[COCO17_IN_RTM]   # (17, 2)
    scores17 = scores_133[COCO17_IN_RTM]     # (17,)
    return kps17, scores17


def normalize_screen(kps2d, w=1.0, h=1.0):
    """
    Normalize 2D keypoints to [-1, 1] range.
    kps2d: (T, 17, 2) in [0,1] normalized coords.
    Returns (T, 17, 2) in [-1, 1].
    """
    out = kps2d.copy()
    out[..., 0] = out[..., 0] * 2 - 1          # x: [0,1] -> [-1,1]
    out[..., 1] = out[..., 1] * 2 - 1          # y: [0,1] -> [-1,1]
    return out


class MotionBERTLifter:
    """
    Wraps MotionBERT for sliding-window 3D pose lifting.
    Usage:
        lifter = MotionBERTLifter(ckpt_path, device)
        poses3d = lifter.lift(pose_frames_2d)  # list of (fi, norm_kps133, scores133)
        # poses3d: dict {frame_idx: np.array (17, 3)}
    """

    def __init__(self, ckpt_path=None, device="cuda"):
        import torch
        self.device = device
        self.torch  = torch

        if ckpt_path is None:
            ckpt_path = download_motionbert_weights()

        # Install MotionBERT architecture if not already
        _ensure_motionbert_arch()

        self.model = _load_model(ckpt_path, device)
        print(f"MotionBERT loaded on {device}")

    def lift(self, pose_frames):
        """
        Lift a full video's 2D poses to 3D.
        pose_frames: list of (frame_idx, norm_kps133 (133,2), scores (133,))
        Returns: dict {frame_idx -> np.array (17, 3)}
        """
        import torch

        if not pose_frames:
            return {}

        # Build (N, 17, 2) array from all frames
        n = len(pose_frames)
        seq2d   = np.zeros((n, 17, 2), dtype=np.float32)
        conf    = np.zeros((n, 17),    dtype=np.float32)
        fidxs   = []

        for i, (fi, kps133, sc133) in enumerate(pose_frames):
            kps17, sc17  = extract_coco17(kps133, sc133)
            seq2d[i]     = kps17
            conf[i]      = sc17
            fidxs.append(fi)

        # Normalize to [-1, 1]
        seq2d_norm = normalize_screen(seq2d)

        # Add confidence as 3rd channel: (N, 17, 3)
        seq_input = np.concatenate([seq2d_norm, conf[:, :, None]], axis=-1)  # (N, 17, 3)

        # Sliding window inference
        poses3d_list = np.zeros((n, 17, 3), dtype=np.float32)
        count        = np.zeros(n,          dtype=np.float32)
        pad          = CLIP_LEN // 2

        # Pad sequence at both ends (reflect)
        padded = np.concatenate([
            seq_input[:pad][::-1],
            seq_input,
            seq_input[-pad:][::-1],
        ], axis=0)  # (N + CLIP_LEN - 1, 17, 3)

        with torch.no_grad():
            for start in range(n):
                clip = padded[start : start + CLIP_LEN]       # (243, 17, 3)
                t    = torch.from_numpy(clip[None]).to(self.device)  # (1, 243, 17, 3)
                out  = self.model(t)                           # (1, 243, 17, 3)
                center_pred = out[0, pad].cpu().numpy()        # (17, 3) — center frame
                poses3d_list[start] += center_pred
                count[start]        += 1.0

        poses3d_list /= count[:, None, None]

        return {fidxs[i]: poses3d_list[i] for i in range(n)}


# ── MotionBERT architecture (self-contained, no repo clone needed) ────────────

def _ensure_motionbert_arch():
    """
    Install the MotionBERT DSTformer architecture module if missing.
    Downloads directly from the official repo on GitHub.
    """
    import importlib
    if importlib.util.find_spec("motionbert_arch") is not None:
        return

    import urllib.request, os, sys
    arch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motionbert_arch.py")
    if os.path.exists(arch_path):
        return

    print("Downloading MotionBERT architecture...")
    # Minimal DSTformer implementation
    _write_dstformer(arch_path)
    sys.path.insert(0, os.path.dirname(arch_path))


def _write_dstformer(path):
    """Write a minimal DSTformer that matches the MotionBERT checkpoint."""
    code = '''
import torch
import torch.nn as nn
import numpy as np
from functools import partial


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn  = Attention(dim, num_heads, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio), drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=512, depth=5,
                 num_heads=8, mlp_ratio=2, norm_layer=None, maxlen=243, num_joints=17, **kwargs):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_joints = num_joints
        self.dim_feat   = dim_feat

        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop     = nn.Dropout(p=0.)

        # Spatial and temporal positional embeddings
        self.spatial_pos  = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.temporal_pos = nn.Parameter(torch.zeros(1, maxlen, dim_feat))

        self.blocks_s = nn.ModuleList([
            Block(dim_feat, num_heads, mlp_ratio, norm_layer=norm_layer) for _ in range(depth)
        ])
        self.blocks_t = nn.ModuleList([
            Block(dim_feat, num_heads, mlp_ratio, norm_layer=norm_layer) for _ in range(depth)
        ])

        self.norm_s = norm_layer(dim_feat)
        self.norm_t = norm_layer(dim_feat)
        self.head   = nn.Linear(dim_feat * 2, dim_out)

        nn.init.trunc_normal_(self.spatial_pos,  std=.02)
        nn.init.trunc_normal_(self.temporal_pos, std=.02)

    def forward(self, x):
        # x: (B, T, J, C_in)
        B, T, J, _ = x.shape
        x = self.joints_embed(x)   # (B, T, J, dim_feat)

        # Spatial stream: process each frame independently
        xs = x + self.spatial_pos.unsqueeze(1)          # broadcast over T
        xs = xs.view(B * T, J, self.dim_feat)
        for blk in self.blocks_s:
            xs = blk(xs)
        xs = self.norm_s(xs).view(B, T, J, self.dim_feat)

        # Temporal stream: process each joint independently
        xt = x + self.temporal_pos[:, :T].unsqueeze(2)  # broadcast over J
        xt = xt.permute(0, 2, 1, 3).contiguous().view(B * J, T, self.dim_feat)
        for blk in self.blocks_t:
            xt = blk(xt)
        xt = self.norm_t(xt).view(B, J, T, self.dim_feat).permute(0, 2, 1, 3)

        # Fuse and project
        out = torch.cat([xs, xt], dim=-1)    # (B, T, J, 2*dim_feat)
        return self.head(out)                 # (B, T, J, 3)
'''
    with open(path, "w") as f:
        f.write(code)
    print(f"  DSTformer written to {path}")


# ── 3D angle math ─────────────────────────────────────────────────────────────

def angle3d(a, b, c):
    """Angle at joint b in 3D space (degrees)."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def compute_3d_angles(pose3d_17, draw_side="right"):
    """
    Compute body angles from 3D joint positions (17 COCO joints).
    pose3d_17: np.array (17, 3)
    Returns dict of angle fields.
    """
    bow_side = "left" if draw_side == "right" else "right"
    J = {
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow":    7, "right_elbow":    8,
        "left_wrist":    9, "right_wrist":   10,
        "left_hip":     11, "right_hip":     12,
        "nose":          0,
    }

    D = draw_side; B = bow_side
    out = {}

    ds = pose3d_17[J[f"{D}_shoulder"]]
    de = pose3d_17[J[f"{D}_elbow"]]
    dw = pose3d_17[J[f"{D}_wrist"]]
    bs = pose3d_17[J[f"{B}_shoulder"]]
    be = pose3d_17[J[f"{B}_elbow"]]
    bw = pose3d_17[J[f"{B}_wrist"]]
    ls = pose3d_17[J["left_shoulder"]]
    rs = pose3d_17[J["right_shoulder"]]
    lh = pose3d_17[J["left_hip"]]
    rh = pose3d_17[J["right_hip"]]

    # Draw elbow angle (3D — depth-accurate)
    out["draw_elbow_angle"] = angle3d(ds, de, dw)

    # Bow elbow angle (3D)
    out["bow_elbow_angle"] = angle3d(bs, be, bw)

    # Shoulder tilt: y-difference in 3D (vertical)
    out["shoulder_tilt_pct"] = float((rs[1] - ls[1]) * 100)

    # Torso lean: angle of hip-midpoint -> shoulder-midpoint from vertical
    sh_mid  = (ls + rs) / 2
    hip_mid = (lh + rh) / 2
    d = sh_mid - hip_mid
    out["torso_lean"] = float(np.degrees(np.arctan2(d[0], -d[1] + 1e-9)))

    # Draw shoulder elevation (3D)
    d_sh = ds - de
    out["draw_shoulder_elevation"] = float(np.degrees(
        np.arctan2(abs(d_sh[1]), abs(d_sh[0]) + 1e-9)
    ))

    # Bow shoulder
    b_sh = bs - be
    out["bow_shoulder_depression"] = float(np.degrees(
        np.arctan2(abs(b_sh[1]), abs(b_sh[0]) + 1e-9)
    ))

    # Draw extension (3D distance)
    out["draw_extension"] = float(np.linalg.norm(dw - bs))

    return out
