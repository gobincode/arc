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

# Pre-download RTMPose wholebody ONNX models from HuggingFace
# (avoids openmmlab CDN which times out on Colab)
print("\nDownloading RTMPose models from HuggingFace...")
run("""python -c "
from huggingface_hub import hf_hub_download
import os, json

det  = hf_hub_download(repo_id='yzd-v/DWPose', filename='yolox_l.onnx')
pose = hf_hub_download(repo_id='yzd-v/DWPose', filename='dw-ll_ucoco_384.onnx')

cfg = {'det': det, 'pose': pose}
out = os.path.join(os.path.dirname(os.path.abspath('.')), 'arc', 'model_paths.json')
with open(out, 'w') as f: json.dump(cfg, f)
print('det :', det)
print('pose:', pose)
print('Saved ->', out)
" """)

