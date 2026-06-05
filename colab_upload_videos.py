"""
Run this on Colab to upload your videos into the repo folder.
The videos go into the same folder as the scripts so paths resolve automatically.

Usage in Colab cell:
    !git clone https://github.com/gobincode/arc.git
    %cd arc
    !python colab_upload_videos.py
"""

import os, shutil

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

print("Choose how to get your videos into Colab:\n")
print("  1) Upload from your computer")
print("  2) Copy from Google Drive (mount first)\n")
choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    from google.colab import files
    print("\nA file picker will open. Select BOTH videos.")
    uploaded = files.upload()
    for fname, data in uploaded.items():
        dest = os.path.join(REPO_DIR, fname)
        with open(dest, "wb") as f:
            f.write(data)
        size_mb = len(data) / 1024 / 1024
        print(f"  Saved {fname} ({size_mb:.0f} MB) -> {dest}")

elif choice == "2":
    from google.colab import drive
    drive.mount("/content/drive")
    print("\nEnter the path to each video inside your Drive.")
    print("Example: /content/drive/MyDrive/archery/20250913_175626.mp4\n")
    for label, default_name in [
        ("TRAIN video (20250913_175626.mp4)", "20250913_175626.mp4"),
        ("TEST  video (20250913_180517.mp4)", "20250913_180517.mp4"),
    ]:
        src = input(f"  Path to {label}: ").strip()
        if src and os.path.exists(src):
            dest = os.path.join(REPO_DIR, default_name)
            print(f"  Copying to {dest} ...")
            shutil.copy2(src, dest)
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"  Done ({size_mb:.0f} MB)")
        else:
            print(f"  File not found: {src}")

else:
    print("Invalid choice.")

print("\nVideos in repo folder:")
for f in os.listdir(REPO_DIR):
    if f.endswith(".mp4"):
        size_mb = os.path.getsize(os.path.join(REPO_DIR, f)) / 1024 / 1024
        print(f"  {f}  ({size_mb:.0f} MB)")
