import os
import shutil
from tqdm import tqdm

src_root = r"J:\Capstone Project - IV Year\VIDEO_DFDA\FaceForensics"
dst_root = "Dataset"
os.makedirs(dst_root, exist_ok=True)

# Iterate over class folders
for cls in tqdm(sorted(os.listdir(src_root))):
    src_cls = os.path.join(src_root, cls)
    dst_cls = os.path.join(dst_root, cls)
    if not os.path.isdir(src_cls):
        continue
    os.makedirs(dst_cls, exist_ok=True)

    videos = [f for f in os.listdir(src_cls) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
    videos = sorted(videos)[:200]   # pick only first 200 videos

    for v in videos:
        src_v = os.path.join(src_cls, v)
        dst_v = os.path.join(dst_cls, v)
        try:
            shutil.copy2(src_v, dst_v)
        except Exception as e:
            print(f"❌ Error copying {v}: {e}")

print("\n✅ Dataset reduced successfully.")
