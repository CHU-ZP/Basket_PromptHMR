import os
import sys
import cv2
import numpy as np
import torch
import time
import tyro
import joblib
from typing import Optional


sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_human, viser_vis_world4d_nocamera
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline

import numpy as np
import torch

def create_world4d_from_results(results, total=None, step=1):
    n_cam = len(results["camera_world"]["Rwc"])
    total = n_cam if total is None else min(total, n_cam)

    # ① 稳定 id 映射（0..C-1）
    raw_ids = sorted(int(p["track_id"]) for p in results["people"].values())
    id_map = {rid: i for i, rid in enumerate(raw_ids)}

    # ② 自动确定统一的 J / B
    target_J, target_B = None, None
    def fit_J(pose_arr, tgtJ):
        if pose_arr.shape[0] == tgtJ:
            return pose_arr
        out = np.zeros((tgtJ, 3), dtype=np.float32)
        m = min(tgtJ, pose_arr.shape[0])
        out[:m] = pose_arr[:m]
        return out

    world4d = {}
    for i in range(0, total, step):
        pose_list, shape_list, trans_list, track_ids = [], [], [], []

        # 收集这一帧出现的所有人
        for pid, people in results["people"].items():
            frames = np.asarray(people["frames"])
            idx = np.where(frames == i)[0]
            if idx.size != 1:
                continue
            j = int(idx[0])
            smplx_w = people["smplx_world"]

            pose_i  = np.asarray(smplx_w["pose"][j], dtype=np.float32)
            if pose_i.ndim == 1:
                assert pose_i.size % 3 == 0
                pose_i = pose_i.reshape(-1, 3)

            # 首次见到设定 target_J
            if target_J is None:
                target_J = pose_i.shape[0]
            pose_i = fit_J(pose_i, target_J)

            shape_i = np.asarray(smplx_w["shape"][j], dtype=np.float32).reshape(-1)
            if target_B is None:
                target_B = shape_i.shape[0]
            # 若维度不一致，截断/补零
            if shape_i.shape[0] != target_B:
                buf = np.zeros((target_B,), dtype=np.float32)
                m = min(target_B, shape_i.shape[0])
                buf[:m] = shape_i[:m]
                shape_i = buf

            trans_i = np.asarray(smplx_w["trans"][j], dtype=np.float32).reshape(3)

            pose_list.append(pose_i[None, ...])       # (1,J,3)
            shape_list.append(shape_i[None, ...])     # (1,B)
            trans_list.append(trans_i[None, ...])     # (1,3)
            track_ids.append(id_map[int(people["track_id"])])  # 稳定 0..C-1

        # 相机位姿（camera→world）
        Rwc = results["camera_world"]["Rwc"][i]
        Twc = results["camera_world"]["Twc"][i]
        cam = np.eye(4, dtype=np.float32)
        cam[:3, :3] = np.asarray(Rwc, dtype=np.float32)
        cam[:3, 3]  = np.asarray(Twc, dtype=np.float32)

        if track_ids:
            pose  = torch.from_numpy(np.concatenate(pose_list,  axis=0)).float()   # (N,J,3)
            shape = torch.from_numpy(np.concatenate(shape_list, axis=0)).float()   # (N,B)
            trans = torch.from_numpy(np.concatenate(trans_list, axis=0)).float()   # (N,3)
            world4d[i] = {
                "pose":     pose,
                "shape":    shape,
                "trans":    trans,
                "track_id": torch.tensor(track_ids, dtype=torch.long),
                "camera":   cam,
                "orig_frame": i,
            }
        else:
            # 统一类型：torch 0 长度张量
            world4d[i] = {
                "track_id": torch.empty(0, dtype=torch.long),
                "camera":   cam,
                "orig_frame": i,
            }
    return world4d


SMPLX_PATH = "data/body_models/smplx"  # 按你的实际路径设定

def main(
    results_path: str,
    fps: int = 30,
):
    """最简可视化入口：给 results.pkl，启动 viser 播放。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    smplx_layer = SMPLX_Layer(SMPLX_PATH).to(device)

    results = joblib.load(results_path)
    world4d = create_world4d_from_results(results)
    frame_keys = sorted(world4d.keys())
    world4d = {ii: world4d[k] for ii, k in enumerate(frame_keys)}

    all_verts = []
    with torch.no_grad():
        for t, w3d in world4d.items():
            if "track_id" not in w3d or int(w3d["track_id"].numel()) == 0:
                continue

            pose  = w3d["pose"].to(device)    # (N, J, 3) 轴角
            shape = w3d["shape"].to(device)   # (N, B)
            trans = w3d["trans"].to(device)   # (N, 3)

            N, J, _ = pose.shape
            # 轴角 -> 旋转矩阵
            rotmat = axis_angle_to_matrix(pose.reshape(-1, 3)).reshape(N, J, 3, 3)

            if J >= 22:
                global_orient = rotmat[:, :1]        # (N,1,3,3)
                body_pose     = rotmat[:, 1:22]      # (N,21,3,3)
            else:
                pad_needed = 22 - J
                eye = torch.eye(3, device=device).reshape(1,1,3,3)
                pad = eye.repeat(N, max(pad_needed-1, 0), 1, 1)  # -1 因为 global 占了 1
                global_orient = rotmat[:, :1]
                body_exist    = rotmat[:, 1:] if J > 1 else rotmat[:, :0]
                body_pose     = torch.cat([body_exist, pad], dim=1) if pad.numel() else body_exist

            verts = smplx_layer(
                global_orient = global_orient,
                body_pose     = body_pose,
                betas         = shape,
                transl        = trans,
            ).vertices  # (N, V, 3) on device

            w3d["vertices"] = verts.detach().cpu().numpy()
            all_verts.append(verts.detach().float().cpu())

    floor = None
    if len(all_verts) > 0:
        all_verts_cat = torch.cat(all_verts, dim=0)  # (sumN, V, 3), float32
        gv, gf, _ = get_floor_mesh(all_verts_cat, scale=2.0)
        floor = [gv, gf]

    server, gui = viser_vis_world4d_nocamera(
        world4d,
        smplx_layer.faces,
        floor=floor,
        init_fps=fps,
    )

    url = f"http://localhost:{server.get_port()}"
    print(f"Open: {url}")
    print("For longer videos, the page may take a few seconds to load.")

    gui_playing, gui_timestep, gui_framerate, num_frames = gui
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)

if __name__ == "__main__":
    tyro.cli(main)