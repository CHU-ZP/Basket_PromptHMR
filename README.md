# Basket PromptHMR

Basket PromptHMR adapts [PromptHMR](https://github.com/yufu-wang/PromptHMR) for
multi-person 3D human reconstruction in basketball videos.

The repository contains:

- single-image PromptHMR inference
- basketball video preprocessing and shot splitting
- video-to-world-coordinate reconstruction
- SMPL / SMPL-X / MCS / GLB exports
- a minimal static GitHub Pages demo with original video and reconstruction video side by side

## Repository Layout

```text
prompt_hmr/              Core PromptHMR model, SMPL family wrappers, utilities, visualization
pipeline/                Full video pipeline: tracking, masks, camera, video HMR, world export
scripts/                 User-facing scripts for demos, setup, splitting, visualization
data/                    Small tracked examples plus ignored local checkpoints/body models/wheels
assets/                  Small static demo videos used by GitHub Pages
results/                 Local reconstruction outputs, ignored by git
index.html               Static GitHub Pages demo
SERVER_RUN_GUIDE.md      Full server setup and run guide
docs/                    Maintainer notes and repository conventions
```

Third-party research code is vendored under `pipeline/`, especially GVHMR,
DROID-Calib, ViTPose, SAM2, and Metric3D-related components.

## Quick Start

Activate the environment:

```bash
source .venv/bin/activate
```

Check the main runtime dependencies:

```bash
uv run python - <<'PY'
import torch, cv2, smplx, detectron2, pytorch3d, ultralytics, viser
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("env ok")
PY
```

For a full fresh-server setup, follow [SERVER_RUN_GUIDE.md](SERVER_RUN_GUIDE.md).
For repository conventions, see [docs/REPOSITORY_STRUCTURE.md](docs/REPOSITORY_STRUCTURE.md).

## Common Workflows

### 1. Split long basketball videos

```bash
uv run python scripts/split_videos.py \
  /path/to/long/videos \
  data/basketball_cut \
  --threshold 6.0 \
  --min-frames 90
```

`/path/to/long/videos` can be a single video file or a directory containing
videos. Output structure:

```text
data/basketball_cut/
  video_name/
    segment_001.mp4
    segment_002.mp4
```

The legacy root entrypoint `cut_video_frame_precise_batch.py` is kept as a
compatibility wrapper.

### 2. Reconstruct one video segment

```bash
uv run python scripts/demo_video.py \
  --input-video data/basketball_cut/video_name/segment_001.mp4 \
  --output-dir results/video_name/segment_001 \
  --no-run-viser
```

For fixed or near-fixed camera videos:

```bash
uv run python scripts/demo_video.py \
  --input-video data/basketball_cut/video_name/segment_001.mp4 \
  --output-dir results/video_name/segment_001 \
  --static-camera \
  --no-run-viser
```

Main outputs:

```text
results/video_name/segment_001/results.pkl
results/video_name/segment_001/world4d.mcs
results/video_name/segment_001/world4d.glb
results/video_name/segment_001/subject-*.smpl
```

### 3. Visualize an existing reconstruction

```bash
uv run python scripts/visualize_results.py \
  --results-path results/video_name/segment_001/results.pkl
```

This starts a local Viser page and prints a URL such as:

```text
http://localhost:8080
```

The old misspelled entrypoint `scripts/visulization.py` is kept as a
compatibility wrapper.

### 4. Run single-image PromptHMR

```bash
uv run python scripts/demo_phmr.py \
  --image data/examples/example_1.jpg
```

## Static Web Demo

The static demo is served by `index.html` and uses small videos under
`assets/`. It is intentionally simple: original video on the left,
reconstruction video on the right, with synchronized playback controls.

When GitHub Pages is configured to deploy from the repository root, the demo is
available at:

```text
https://<github-user>.github.io/<repo-name>/
```

## Data And Assets

Large or license-restricted files are intentionally ignored:

```text
data/pretrain/
data/body_models/
data/wheels/
results/
run_batch_logs/
.venv/
```

Small example images and the static demo videos are tracked.

## Development Notes

- Keep user-facing commands in `scripts/`.
- Keep generated outputs under `results/`.
- Keep large checkpoints, body models, and wheels under `data/`, but do not
  commit them.
- Prefer adding compatibility wrappers when renaming existing scripts.
