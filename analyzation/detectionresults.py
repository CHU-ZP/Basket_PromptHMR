import os
import joblib
import cv2
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.insert(0, project_root)

from prompt_hmr.utils.visualizer import draw_boxes, draw_masks, draw_coco_kpts
from pipeline.pipeline import Pipeline

# === 路径配置 ===
video_path = "/data/zepeng/PromptHMR/data/examples/10895_11002.mp4"
results_path = "/data/zepeng/PromptHMR/results/10895_11002/results.pkl"
output_dir = "/data/zepeng/PromptHMR/vis_outputs/10895_11002"
os.makedirs(output_dir, exist_ok=True)

# === 加载视频帧 ===
pipeline = Pipeline()
images, _ = pipeline.load_frames(video_path, output_dir)
pipeline.images = images
pipeline.results = joblib.load(results_path)

print("Keys in results.pkl:", pipeline.results.keys())

people = pipeline.results["people"]
# masks = pipeline.results["masks"]
num_frames = len(images)


# === 可视化每一帧 ===
for i in range(num_frames):
    img = images[i].copy()

    # --- 绘制分割掩码 ---
    # if masks is not None and i < len(masks):
    #     img = draw_masks(img, [masks[i]])

    # --- 绘制每个个体的检测框---
    for pid, pdata in people.items():
        frames = pdata["frames"]
        if i in frames:
            idx = np.where(frames == i)[0][0]

            # 画检测框
            bbox = pdata["bboxes"][idx]
            img = draw_boxes(img, [bbox], numbered=True)


    # 保存该帧图像
    out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
    cv2.imwrite(out_path, img)

print(f"✅ 可视化完成，图像保存于：{os.path.abspath(output_dir)}")
