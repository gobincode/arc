"""
Run this first on Colab to install all dependencies.
    !python colab_setup.py
"""
import subprocess, sys

def run(cmd):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

run("pip install -q -U openmim")
run("mim install -q mmengine")
run('mim install -q "mmcv>=2.0.0"')
run('mim install -q "mmdet>=3.0.0"')
run('mim install -q "mmpose>=1.0.0"')
run("pip install -q opencv-python-headless numpy")

print("\nAll dependencies installed.")
