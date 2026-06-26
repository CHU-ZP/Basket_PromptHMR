# Repository Structure

This project combines a compact application layer with several vendored
research dependencies. The top-level layout should stay stable so that scripts,
docs, and GitHub Pages links remain easy to understand.

## Top-Level Directories

```text
prompt_hmr/
```

Core PromptHMR code: model definitions, SMPL/SMPL-X wrappers, inference
helpers, evaluation utilities, and visualization helpers.

```text
pipeline/
```

Full video reconstruction pipeline. This contains the orchestration code plus
vendored third-party research components for detection, tracking, camera
estimation, video HMR, and world-coordinate export.

```text
scripts/
```

User-facing commands. New runnable tools should be added here rather than in
the repository root.

Current primary entries:

- `demo_phmr.py`: single-image PromptHMR demo
- `demo_video.py`: full video reconstruction demo
- `split_videos.py`: split long videos into shot-based short clips
- `visualize_results.py`: open a saved `results.pkl` in Viser
- `fetch_data.sh`: download checkpoints and examples
- `fetch_smplx.sh`: download SMPL/SMPL-X body models

Compatibility wrappers may remain for old names, but documentation should point
to the primary entries above.

```text
data/
```

Local data root. Only tiny examples should be committed. Large files such as
checkpoints, wheels, annotations, and body models must stay ignored.

Expected local-only subdirectories:

- `data/pretrain/`
- `data/body_models/`
- `data/wheels/`
- `data/annotations/`

```text
assets/
```

Small static assets for the GitHub Pages demo. Keep this directory lightweight;
large videos should be hosted elsewhere or compressed before committing.

```text
results/
```

Generated reconstruction outputs. This directory is ignored and should never be
used as a source of committed artifacts.

```text
docs/
```

Maintainer-facing documentation. Use this folder for repository organization,
deployment notes, conventions, and longer explanations.

## Root Files

- `README.md`: project overview and common workflows
- `SERVER_RUN_GUIDE.md`: full server setup and end-to-end run guide
- `index.html`: static GitHub Pages demo
- `requirements.txt`: Python dependency list used with the uv setup guide
- `.gitignore`: protects generated outputs and large local assets

## Conventions

- Add new runnable Python commands under `scripts/`.
- Keep root-level files limited to project metadata, GitHub Pages, and
  compatibility wrappers.
- Do not commit local reconstruction outputs, logs, checkpoints, wheels, or body
  model files.
- Prefer compatibility wrappers when renaming existing scripts.
- Keep GitHub Pages assets small enough for normal GitHub hosting.
