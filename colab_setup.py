"""
Run this first on Colab to install all dependencies.
    !python colab_setup.py

Uses rtmlib (ONNX Runtime backend) — no mmcv/mmdet build issues.
Works on Python 3.12 + any CUDA version.
"""
import subprocess

def run(cmd, check=True):
    print(f">> {cmd}")
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")
    return r.returncode

# rtmlib: lightweight RTMPose wrapper using ONNX Runtime
# auto-downloads RTMPose wholebody ONNX models on first run
run("pip install -q rtmlib")

# ONNX Runtime GPU (matches CUDA on Colab)
run("pip install -q onnxruntime-gpu", check=False)   # falls back gracefully if CUDA mismatch

# opencv + numpy
run("pip install -q opencv-python-headless numpy")

print("\nAll dependencies installed. Verifying...")
run("python -c \"from rtmlib import Wholebody; print('rtmlib OK')\"")

