"""
Run this first on Colab to install all dependencies.
    !python colab_setup.py

Works on Python 3.12 + CUDA 12.8 (torch 2.11+).
"""
import subprocess, sys

def run(cmd, check=True):
    print(f">> {cmd}")
    # -q -q suppresses WARNING-level pip messages (dependency conflicts from
    # unrelated Colab packages). Real errors still surface.
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r.returncode

# Step 1: pin setuptools to satisfy torch<82 requirement.
# Conflict warnings from openxlab/pymc etc. are safe to ignore — we don't use them.
run("pip install -q -q 'setuptools==75.8.2' pip wheel", check=False)

# Step 2: mmengine
run("pip install -q mmengine")

# Step 3: install mmcv from PyPI (mmcv 2.x on PyPI includes GPU support
# and avoids the openmmlab CDN which times out on Colab)
import torch
torch_ver = torch.__version__.split("+")[0]
cuda_ver  = torch.version.cuda or "unknown"
print(f"\nDetected: torch={torch_ver}, cuda={cuda_ver}")

print("Installing mmcv from PyPI...")
run("pip install -q -q mmcv")

# Step 4: mmdet + mmpose
run("pip install -q mmdet mmpose")

# Step 5: opencv
run("pip install -q opencv-python-headless numpy")

print("\nAll dependencies installed. Verifying...")
run("python -c \"from mmpose.apis import MMPoseInferencer; print('MMPose OK')\"")

