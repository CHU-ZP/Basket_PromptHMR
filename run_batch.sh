#!/bin/bash

INPUT_DIR="/data/zepeng/PromptHMR/data/NBA19_20_cut"
SUBSAMPLE=1
GPU_ID=${1:-0}  # 默认为GPU 0，可通过第一个参数指定
PROCESSED_FILE="processed_folders.txt"

# 确保记录文件存在
touch "$PROCESSED_FILE"

for subdir in "$INPUT_DIR"/*; do
    if [ -d "$subdir" ]; then
        parent_folder=$(basename "$subdir")
        LOG_PARENT_DIR="run_batch_logs/${parent_folder}"

        # 双重保险：
        # 条件1：processed_folders.txt 已记录
        # 条件2：log 文件夹已存在
        if grep -Fxq "$parent_folder" "$PROCESSED_FILE" || [ -d "$LOG_PARENT_DIR" ]; then
            echo "Skipping $parent_folder: already processed."
            continue
        fi

        # 写入已处理记录
        echo "$parent_folder" >> "$PROCESSED_FILE"

        # 遍历子文件夹下所有 mp4
        for video in "$subdir"/*.mp4; do
            if [ ! -f "$video" ]; then
                continue
            fi

            filename=$(basename "$video" .mp4)
            LOG_DIR="${LOG_PARENT_DIR}/${filename}"
            mkdir -p "$LOG_DIR"

            log_file="${LOG_DIR}/log.txt"
            mem_log_file="${LOG_DIR}/gpu_mem.log"

            echo "Processing $video on GPU $GPU_ID"
            echo "Logs: $log_file"
            echo "GPU Mem Logs: $mem_log_file"

            # 启动 nvidia-smi 监控
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" -l 1 > "$mem_log_file" &
            SMI_PID=$!

            OUTPUT_DIR="results/${parent_folder}/${filename}"  # 注意：不是 LOG_DIR
            mkdir -p "$OUTPUT_DIR"

            # 运行 Python 脚本
            CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/demo_video.py \
                --input_video "$video" \
                --output_dir "$OUTPUT_DIR" \
                --viser_subsample "$SUBSAMPLE" \
                --no-run-viser \
                > "$log_file" 2>&1

            echo "Saving results to: $OUTPUT_DIR" >> "${LOG_DIR}/log.txt"

            # 结束 nvidia-smi
            kill $SMI_PID

            # 计算显存峰值和平均
            peak_mem=$(awk 'BEGIN{max=0} {if($1>max) max=$1} END{print max}' "$mem_log_file")
            avg_mem=$(awk '{sum+=$1; count+=1} END{if(count>0) print int(sum/count); else print 0}' "$mem_log_file")

            echo "GPU Peak Memory: ${peak_mem} MiB" >> "$log_file"
            echo "GPU Average Memory: ${avg_mem} MiB" >> "$log_file"
        done
    fi
done