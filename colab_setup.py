"""
Run this first on Colab to install all dependencies.
    !python colab_setup.py

Works on Python 3.12 (fixes pkgutil.ImpImporter issue with mim).
"""
import subprocess, sys

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Step 1: fix setuptools (Python 3.12 broke pkg_resources in older versions)
run("pip install -q -U setuptools pip wheel")

# Step 2: mmengine (no mim needed)
run("pip install -q mmengine")

# Step 3: detect torch + CUDA version to pick the right mmcv wheel
import torch
torch_ver   = torch.__version__.split("+")[0]             # e.g. "2.3.0"
torch_short = "torch" + ".".join(torch_ver.split(".")[:2])  # "torch2.3"

nvcc = subprocess.run("nvcc --version", shell=True, capture_output=True, text=True).stdout
if "12.1" in nvcc or "12.2" in nvcc or "12.3" in nvcc or "12.4" in nvcc or "12.5" in nvcc:
    cuda_short = "cu121"
elif "12.0" in nvcc:
    cuda_short = "cu120"
elif "11.8" in nvcc:
    cuda_short = "cu118"
elif "11.7" in nvcc:
    cuda_short = "cu117"
else:
    cuda_short = "cu121"   # Colab default as of 2025

mmcv_index = f"https://download.openmmlab.com/mmcv/dist/{cuda_short}/{torch_short}/index.html"
print(f"\nDetected: torch={torch_ver}, cuda={cuda_short}")
print(f"mmcv wheel index: {mmcv_index}\n")

run(f"pip install -q mmcv -f {mmcv_index}")

# Step 4: mmdet + mmpose
run("pip install -q mmdet mmpose")

# Step 5: opencv
run("pip install -q opencv-python-headless numpy")

print("\nAll dependencies installed. Verifying...")
run("python -c \"from mmpose.apis import MMPoseInferencer; print('MMPose OK')\"")

