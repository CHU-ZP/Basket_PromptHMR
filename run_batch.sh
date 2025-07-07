#!/bin/bash

INPUT_DIR="data/NBA19_20_cut/855_19_20"
SUBSAMPLE=1
LOG_DIR="run_batch_logs"

mkdir -p "$LOG_DIR"

for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)
    log_file="$LOG_DIR/${filename}.log"

    echo "Processing $video, logging to $log_file"

    python scripts/demo_video.py \
        --input_video "$video" \
        --viser_subsample "$SUBSAMPLE" \
        --no-run-viser \
        > "$log_file" 2>&1
done
