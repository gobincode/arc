"""
Downloads the archery videos from Google Drive into the repo folder.

Usage:
    !python colab_upload_videos.py
"""

import os, subprocess

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEOS = [
    ("11wXKz--HuuIdp_P17sz8wyKX-CraveMv", "20250913_175626.mp4", "TRAIN"),
    ("1ETwfO7p3UY_QbRHhNNRVgocWCrvNlTyn",  "20250913_180517.mp4", "TEST"),
]

# gdown is pre-installed on Colab; install if missing
try:
    import gdown
except ImportError:
    subprocess.run("pip install -q gdown", shell=True)
    import gdown

for file_id, filename, label in VIDEOS:
    dest = os.path.join(REPO_DIR, filename)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"[{label}] {filename} already exists ({size_mb:.0f} MB), skipping.")
        continue
    print(f"[{label}] Downloading {filename} ...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest, quiet=False)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  Done: {size_mb:.0f} MB -> {dest}")

print("\nAll videos ready.")
