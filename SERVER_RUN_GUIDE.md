# 服务器完整运行指南

这份指南用于在另一台 Linux 服务器上重新搭建 Basket PromptHMR 环境，并完整处理篮球视频。

当前本机验证过的基线是：

- Python 3.11.9，由 `uv` 管理虚拟环境
- PyTorch 2.4.0 + CUDA 12.1 wheels
- NVIDIA GPU，`torch.cuda.is_available()` 为 `True`
- PromptHMR、SMPL/SMPL-X、SAM2、Detectron2、ViTPose、DROID、相机标定等资源放在 `data/` 下

## 1. 克隆仓库

```bash
git clone https://github.com/CHU-ZP/Basket_PromptHMR.git
cd Basket_PromptHMR
```

如果后续代码在私有 fork 或其他分支上，克隆对应仓库和分支即可。

## 2. 检查服务器基础环境

先确认 GPU 可见：

```bash
nvidia-smi
```

Ubuntu 上建议安装这些系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  git wget unzip ffmpeg build-essential ninja-build \
  libgl1 libglib2.0-0 libegl1 libsm6 libxext6 libsuitesparse-dev
```

如果服务器需要从源码编译 `pytorch3d`，还需要有 CUDA toolkit 和 `nvcc`：

```bash
nvcc --version
```

如果没有 `nvcc`，后面 `pytorch3d` 可能会失败。最省心的做法是使用带 CUDA toolkit 的机器镜像，或提前准备好匹配 Python 3.11、PyTorch 2.4、CUDA 12.1 的 `pytorch3d` wheel。

## 3. 安装 uv

如果服务器还没有 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

重新打开 shell，或者 source shell 配置后检查：

```bash
uv --version
```

## 4. 创建 Python 环境

```bash
uv venv --python 3.11.9 .venv
source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel
```

安装 PyTorch 2.4 CUDA 12.1：

```bash
uv pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install torch-scatter \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

uv pip install xformers==0.0.27.post2 \
  --index-url https://download.pytorch.org/whl/cu121 \
  --no-deps
```

先安装普通依赖，跳过 `requirements.txt` 里的 git 源码包：

```bash
uv pip install -r <(grep -v '^git+' requirements.txt)
```

再安装两个需要特殊处理的源码依赖：

```bash
uv pip install --no-build-isolation \
  "git+https://github.com/mattloper/chumpy@9b045ff5d6588a24a0bab52c83f032e2ba433e17"

MAX_JOBS=4 uv pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

这里使用 `--no-build-isolation` 是为了避免源码包构建时找不到当前环境里的 `pip`、`torch` 或 CUDA 相关依赖。

## 5. 安装本地 video pipeline wheels

世界坐标视频链路依赖几组本地 wheel。先下载 wheel 包：

```bash
uvx gdown --folder -O ./data/ \
  "https://drive.google.com/drive/folders/1IXyhVqL25ofI-tYqyUZCqF-h4V20795H?usp=sharing"
```

确认：

```bash
ls data/wheels
```

然后安装：

```bash
uv pip install data/wheels/detectron2-0.8-cp311-cp311-linux_x86_64.whl
uv pip install data/wheels/droid_backends_intr-0.3-cp311-cp311-linux_x86_64.whl
uv pip install data/wheels/lietorch-0.3-cp311-cp311-linux_x86_64.whl
uv pip install data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl
uv pip install data/wheels/gloss-0.5.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

这些 wheel 和 Python、CUDA、PyTorch 版本绑定较强。另一台服务器建议优先使用同样的 Python 3.11、PyTorch 2.4、CUDA 12.1 组合。

## 6. 下载 checkpoint 和预训练资源

```bash
bash scripts/fetch_data.sh
```

下载完成后至少应看到：

```bash
ls data/pretrain/phmr/checkpoint.ckpt
ls data/pretrain/phmr/config.yaml
ls data/pretrain/phmr_vid/prhmr_release_002.ckpt
ls data/pretrain/phmr_vid/prhmr_release_002.yaml
ls data/pretrain/vitpose-h-coco_25.pth
ls data/pretrain/camcalib_sa_biased_l2.ckpt
ls data/pretrain/droidcalib.pth
ls data/pretrain/sam_vit_h_4b8939.pth
ls data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt
ls data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl
```

YOLO 权重不是必须手动放进仓库。如果 `data/pretrain/yolo11x.pt` 或 `data/pretrain/yolov8x.pt` 不存在，代码会回退到 Ultralytics 模型名，并在第一次运行时下载到用户缓存。

如果服务器离线，建议提前把这两个文件放好：

```text
data/pretrain/yolo11x.pt
data/pretrain/yolov8x.pt
```

## 7. 下载 SMPL 和 SMPL-X

先分别注册并接受 license：

- SMPL-X: https://smpl-x.is.tue.mpg.de
- SMPL: https://smpl.is.tue.mpg.de

交互式下载：

```bash
bash scripts/fetch_smplx.sh
```

也可以用环境变量非交互运行：

```bash
SMPLX_USERNAME='your-email@example.com' \
SMPLX_PASSWORD='your-smplx-password' \
SMPL_USERNAME='your-email@example.com' \
SMPL_PASSWORD='your-smpl-password' \
bash scripts/fetch_smplx.sh
```

注意：SMPL 和 SMPL-X 是两个不同网站、两套授权。SMPL-X 能下载成功不代表 SMPL 一定能下载成功；如果 SMPL 返回 `401 Unauthorized`，需要登录 SMPL 网站确认账号已接受对应 license。

下载完成后检查：

```bash
ls data/body_models/smplx/SMPLX_NEUTRAL.npz
ls data/body_models/smplx/SMPLX_MALE.npz
ls data/body_models/smplx/SMPLX_FEMALE.npz
ls data/body_models/smpl/SMPL_NEUTRAL.pkl
ls data/body_models/smpl/SMPL_MALE.pkl
ls data/body_models/smpl/SMPL_FEMALE.pkl
ls data/body_models/smplx2smpl.pkl
ls data/body_models/smplx2smpl_joints.npy
ls data/body_models/J_regressor_h36m.npy
```

这些文件体积大，并且有 license 限制，不要提交到 git。

## 8. 更快的迁移方式：复制 data 目录

如果两台机器都在你的授权范围内，最省时间的方式是直接从当前可运行机器同步资源：

```bash
rsync -avP data/pretrain/ user@server:/path/to/Basket_PromptHMR/data/pretrain/
rsync -avP data/body_models/ user@server:/path/to/Basket_PromptHMR/data/body_models/
rsync -avP data/wheels/ user@server:/path/to/Basket_PromptHMR/data/wheels/
```

这样另一台服务器只需要创建环境、安装依赖和 wheel，不需要重新下载大文件。

## 9. 环境 smoke test

先跑 import test：

```bash
uv run python - <<'PY'
import torch, cv2, smplx, detectron2, pytorch3d, ultralytics, viser
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print("env ok")
PY
```

期望类似：

```text
2.4.0+cu121 True 12.1
env ok
```

再跑 body model test：

```bash
uv run python - <<'PY'
from prompt_hmr.smpl_family import SMPL, SMPLX
from data_config import SMPL_PATH, SMPLX_PATH
smpl = SMPL(SMPL_PATH, gender='neutral')
smplx = SMPLX(SMPLX_PATH, gender='neutral')
print('smpl faces', smpl.faces.shape)
print('smplx faces', smplx.faces.shape)
print('body models ok')
PY
```

再跑一个不依赖 YOLO 检测的最小 PHMR forward：

```bash
uv run python - <<'PY'
import cv2
import torch
from torch.amp import autocast
from prompt_hmr import load_model_from_folder
from prompt_hmr.models.inference import prepare_batch

image = 'data/examples/example_1.jpg'
img = cv2.imread(image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]
boxes = torch.tensor([[0.0, 0.0, float(w - 1), float(h - 1)]])
model = load_model_from_folder('data/pretrain/phmr')
batch = prepare_batch([{'image_cv': img, 'boxes': boxes, 'text': None, 'masks': None}], img_size=896, interaction=False)
with torch.no_grad(), autocast('cuda'):
    out = model(batch, mask_prompt=False, use_mean_hands=True)[0]
print('vertices', tuple(out['vertices'].shape))
print('minimal forward ok')
PY
```

最后跑短视频 smoke test。这个会验证检测、分割、跟踪、相机、PHMR video head、世界坐标导出链路。

先从示例视频截一个很短的片段：

```bash
ffmpeg -y -i data/examples/boxing.mp4 -t 3 -an data/examples/boxing_short.mp4
```

然后运行：

```bash
uv run python - <<'PY'
import os
import shutil
from pipeline import Pipeline

out = 'results/smoke_boxing_short_5f'
if os.path.isdir(out):
    shutil.rmtree(out)

p = Pipeline(static_cam=True)
p.cfg.run_post_opt = False
p.cfg.num_max_people = 2
results = p('data/examples/boxing_short.mp4', out, static_cam=True, save_only_essential=True, max_frame=5)
print('people', list(results['people'].keys()))
print('has_tracks', results['has_tracks'])
print('has_hps_cam', results['has_hps_cam'])
print('has_hps_world', results['has_hps_world'])
print('has_slam', results['has_slam'])
print('video smoke ok')
PY
```

期望输出文件：

```text
results/smoke_boxing_short_5f/results.pkl
results/smoke_boxing_short_5f/world4d.mcs
results/smoke_boxing_short_5f/world4d.glb
```

## 10. 处理单个篮球视频

普通运行：

```bash
uv run python scripts/demo_video.py \
  --input-video /path/to/video.mp4 \
  --output-dir results/my_video \
  --no-run-viser
```

如果视频来自固定机位或近似固定机位，建议加 `--static-camera`：

```bash
uv run python scripts/demo_video.py \
  --input-video /path/to/video.mp4 \
  --output-dir results/my_video_static \
  --static-camera \
  --no-run-viser
```

如果需要打开 Viser 可视化，去掉 `--no-run-viser`。远程服务器上通常需要做 SSH 端口转发，然后在本地浏览器打开对应端口。

主要输出：

```text
results/my_video/results.pkl
results/my_video/world4d.mcs
results/my_video/world4d.glb
results/my_video/subject-*.smpl
```

`.mcs` 可用于 Meshcapade，`.glb` 可导入 Blender 或其他 3D 工具。

## 11. 批量处理篮球视频

如果手里是长视频，可以先切成短片段：

```bash
uv run python cut_video_frame_precise_batch.py \
  /path/to/long/videos \
  data/basketball_cut \
  --threshold 6.0 \
  --min-frames 90
```

`/path/to/long/videos` 可以是一个视频文件，也可以是包含多个视频的目录。输出结构会类似：

```text
data/basketball_cut/
  long_video_1/
    segment_001.mp4
    segment_002.mp4
  long_video_2/
    segment_001.mp4
```

当前 `run_batch.sh` 假设输入目录结构是：

```text
INPUT_DIR/
  game_or_clip_group_1/
    clip_001.mp4
    clip_002.mp4
  game_or_clip_group_2/
    clip_003.mp4
```

先编辑 `run_batch.sh`：

```bash
INPUT_DIR="/path/to/folder/of/video-subfolders"
SUBSAMPLE=1
```

在 GPU 0 上运行：

```bash
bash run_batch.sh 0
```

输出会放在：

```text
results/<parent_folder>/<video_name>/
```

日志会放在：

```text
run_batch_logs/<parent_folder>/<video_name>/log.txt
run_batch_logs/<parent_folder>/<video_name>/gpu_mem.log
```

脚本只有在一个父目录下所有视频都成功处理后，才会把该父目录写入 `processed_folders.txt`。

## 12. 常用配置

主配置文件：

```text
pipeline/config.yaml
```

常看的选项：

- `tracker: sam2`：使用 SAM2 做 tracking/mask
- `num_max_people`：限制最多跟踪人数
- `det_thresh`、`det_score_thresh`、`det_height_thresh`：过滤检测框
- `run_post_opt`：是否开启后优化
- `run_post_opt_cam`：后优化是否优化相机
- `max_height`：读视频时的最大高度，越大越慢
- `max_fps`：读视频时的最大帧率，越大越慢
- `use_floor_rectify`：是否使用地面矫正

调试阶段建议先用较快配置：

```yaml
run_post_opt: false
num_max_people: 2
max_fps: 30
```

等短视频跑通后，再打开更重的阶段。

## 13. 常见问题

### SMPL 下载 401 Unauthorized

通常是 SMPL 网站 license 没接受，或 SMPL 和 SMPL-X 账号权限不一致。分别登录两个网站，确认已经接受对应 license。

### `uv pip install -r requirements.txt` 报 `No module named 'pip'`

不要直接一次性安装带 git 源码包的 `requirements.txt`。按本指南的方式先执行：

```bash
uv pip install -r <(grep -v '^git+' requirements.txt)
```

再用 `--no-build-isolation` 单独安装 `chumpy` 和 `pytorch3d`。

### `pytorch3d` 编译失败

先检查：

```bash
nvcc --version
```

如果没有 CUDA toolkit，安装匹配服务器 driver 和 PyTorch CUDA 12.1 的 toolkit，或者换用预编译 wheel。然后重试：

```bash
MAX_JOBS=4 uv pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### DeepLab 或 YOLO 第一次运行会下载

DeepLabv3 和 YOLO 权重可能在第一次运行时下载到用户缓存。离线服务器要么提前跑一次 warm cache，要么复制缓存/权重文件。

### 视频链路很慢或显存不够

先用：

```bash
--static-camera --no-run-viser
```

并在 `pipeline/config.yaml` 中临时设置：

```yaml
run_post_opt: false
num_max_people: 2
max_fps: 30
```

如果还是显存不够，优先缩短视频、降低 `max_height`、减少 `num_max_people`。

### 处理结果为空或人数不对

优先检查：

- 视频里人物是否太小，必要时降低 `det_height_thresh`
- 检测阈值是否过高，必要时降低 `det_thresh` 或 `det_score_thresh`
- `num_max_people` 是否小于球场上需要保留的人数
- 固定机位视频是否用了 `--static-camera`

## 14. push 前检查

这些路径应保持 ignored，不要提交：

```text
.venv/
data/pretrain/
data/body_models/
data/wheels/
data/annotations/
results/
processed_folders.txt
```

push 前检查：

```bash
git status --short --ignored
git diff --check
```

如果 `git status --short` 里只出现代码、脚本、文档变更，并且 `git diff --check` 没有输出，就可以正常 commit/push。
