import cv2
import numpy as np
import os
import subprocess
import argparse
from pathlib import Path


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def detect_shot_boundaries(video_path, threshold=6.0, min_interval=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

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


def cut_video_by_frames(video_path, frame_ids, output_base_dir, min_frames=90):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frame_ids = [0] + frame_ids + [total_frames]

    clip_idx = 1
    for i in range(len(frame_ids) - 1):
        start_frame = frame_ids[i] + 1
        end_frame = frame_ids[i + 1] - 1

        if end_frame - start_frame < min_frames:
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
        subprocess.run(cmd, check=True)

        clip_idx += 1


def process_video(video_path, output_dir, threshold=6.0, min_interval=1, min_frames=90):
    print(f"Processing: {video_path}")
    boundaries = detect_shot_boundaries(str(video_path), threshold=threshold, min_interval=min_interval)
    cut_video_by_frames(str(video_path), boundaries, str(output_dir), min_frames=min_frames)


def iter_videos(input_dir):
    for path in sorted(Path(input_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def process_input(input_path, output_dir, threshold=6.0, min_interval=1, min_frames=90):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        process_video(input_path, output_dir, threshold, min_interval, min_frames)
        return

    if input_path.is_dir():
        videos = list(iter_videos(input_path))
        if not videos:
            print(f"No videos found in {input_path}")
            return

        for video_path in videos:
            process_video(video_path, output_dir, threshold, min_interval, min_frames)
        return

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split long basketball videos into shorter segments using optical-flow shot boundaries."
    )
    parser.add_argument(
        "input_path",
        help="Input video file or directory containing videos.",
    )
    parser.add_argument(
        "output_dir",
        help="Output root directory. Segments are saved under output_dir/<video_name>/segment_*.mp4.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="Optical-flow magnitude threshold for shot boundary detection. Default: 6.0.",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=1,
        help="Minimum frame interval between two detected boundaries. Default: 1.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=90,
        help="Skip output segments shorter than this many frames. Default: 90.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    process_input(
        args.input_path,
        args.output_dir,
        threshold=args.threshold,
        min_interval=args.min_interval,
        min_frames=args.min_frames,
    )

    print("All videos processed.")


if __name__ == "__main__":
    main()
