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

# Step 3: detect torch + CUDA, map to nearest available mmcv wheel
import torch

torch_ver   = torch.__version__.split("+")[0]              # "2.11.0"
torch_major = int(torch_ver.split(".")[0])
torch_minor = int(torch_ver.split(".")[1])
torch_short = f"torch{torch_major}.{torch_minor}"          # "torch2.11"

# Get CUDA version from torch (more reliable than nvcc on Colab)
cuda_ver = torch.version.cuda or ""                        # "12.8"
cuda_parts = cuda_ver.replace(".", "")                     # "128"
cuda_tag = f"cu{cuda_parts}"                               # "cu128"

print(f"\nDetected: torch={torch_ver}, cuda={cuda_ver} ({cuda_tag})")

# mmcv available wheels (update this list as openmmlab releases more)
AVAILABLE = ["cu128", "cu126", "cu124", "cu121", "cu120", "cu118", "cu117"]
TORCH_AVAILABLE = [
    f"torch{torch_major}.{torch_minor}",
    f"torch{torch_major}.{torch_minor - 1}" if torch_minor > 0 else None,
    "torch2.3", "torch2.2", "torch2.1", "torch2.0",
]

mmcv_installed = False

# Try exact match first, then fall back to older CUDA tags
for cuda in [cuda_tag] + [c for c in AVAILABLE if c != cuda_tag]:
    for tv in [t for t in TORCH_AVAILABLE if t]:
        url = f"https://download.openmmlab.com/mmcv/dist/{cuda}/{tv}/index.html"
        print(f"  Trying mmcv wheel: {cuda}/{tv} ...", end=" ")
        rc = run(f"pip install -q mmcv -f {url}", check=False)
        if rc == 0:
            # verify it actually imports
            verify = subprocess.run(
                "python -c 'import mmcv; print(mmcv.__version__)'",
                shell=True, capture_output=True, text=True
            )
            if verify.returncode == 0:
                print(f"OK ({verify.stdout.strip()})")
                mmcv_installed = True
                break
            else:
                print("import failed, trying next...")
        else:
            print("not found")
    if mmcv_installed:
        break

if not mmcv_installed:
    # Last resort: PyPI CPU-only mmcv (inference still works, just no CUDA custom ops)
    print("  Falling back to PyPI mmcv (CPU ops only — inference still works)")
    run("pip install -q mmcv")

# Step 4: mmdet + mmpose
run("pip install -q mmdet mmpose")

# Step 5: opencv
run("pip install -q opencv-python-headless numpy")

print("\nAll dependencies installed. Verifying...")
run("python -c \"from mmpose.apis import MMPoseInferencer; print('MMPose OK')\"")

