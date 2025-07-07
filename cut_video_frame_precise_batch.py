import cv2
import numpy as np
import os
import subprocess

def detect_shot_boundaries(video_path, threshold=6.0, min_interval=1):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_ids = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = np.mean(mag)

            if mean_mag > threshold and (len(frame_ids) == 0 or frame_idx - frame_ids[-1] > min_interval):
                print(f"Shot boundary at frame {frame_idx}, mag = {mean_mag:.2f}")
                frame_ids.append(frame_idx)

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return frame_ids

def cut_video_by_frames(video_path, frame_ids, output_base_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_ids = [0] + frame_ids + [total_frames]

    clip_idx = 1
    for i in range(len(frame_ids) - 1):
        start_frame = frame_ids[i] + 1  # 避免切换帧
        end_frame = frame_ids[i + 1] - 1

        if end_frame - start_frame < 90:
            continue

        output_file = os.path.join(output_dir, f'segment_{clip_idx:03d}.mp4')

        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=PTS-STARTPTS",
            '-an',
            '-y',
            output_file
        ]

        print('Running:', ' '.join(cmd))
        subprocess.run(cmd)

        clip_idx += 1

def process_all_videos(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_dir, filename)
            print(f"Processing: {video_path}")
            boundaries = detect_shot_boundaries(video_path)
            cut_video_by_frames(video_path, boundaries, output_dir)

if __name__ == "__main__":
    input_videos_dir = '/data/zepeng/PromptHMR/BASKET/NBA19_20'
    output_segments_dir = '/data/zepeng/PromptHMR/data/NBA19_20_cut'

    process_all_videos(input_videos_dir, output_segments_dir)

    print("所有视频处理完成。")
