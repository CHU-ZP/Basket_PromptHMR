import os
import sys
import cv2
import numpy as np
import torch
import time
import tyro
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_human, viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline


def main(input_video='...', 
         static_camera=False,
         run_viser=True,
         viser_total=1500, 
         viser_subsample=1,
         output_dir: Optional[str] = None):
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    if output_dir is None:
        output_folder = 'results/' + os.path.basename(input_video).split('.')[0]
    else:
        output_folder = output_dir

    os.makedirs(output_folder, exist_ok=True)

    pipeline = Pipeline(static_cam=static_camera)
    results = pipeline.__call__(input_video, 
                                output_folder, 
                                save_only_essential=False)
   
    # Downsample for viser visualization
    images = pipeline.images[:viser_total][::viser_subsample]
    world4d = pipeline.create_world4d(step=viser_subsample, total=viser_total)
    world4d = {i:world4d[k] for i,k in enumerate(world4d)}

    # Get vertices
    for k in world4d:
        world3d = world4d[k]
        rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
        verts = smplx(global_orient = rotmat[:,:1].cuda(),
                      body_pose = rotmat[:,1:22].cuda(),
                      betas = world3d['shape'].cuda(),
                      transl = world3d['trans'].cuda()).vertices.cpu().numpy()
        
        world3d['vertices'] = verts
        if k == 0:  # need to compute over the entire sequence
            [gv, gf, gc] = get_floor_mesh(torch.tensor(verts), scale=3)


    # Viser
    if run_viser:
        server, gui = viser_vis_world4d(images, 
                                        world4d, 
                                        smplx.faces, 
                                        floor=[gv, gf],
                                        init_fps=30/viser_subsample)
        
        url = f'https://localhost:{server.get_port()}'
        print(f'Please use this url to view the results: {url}')
        print('For longer video, it will take a few seconds for the webpage to load.')

        gui_playing, gui_timestep, gui_framerate, num_frames = gui
        while True:
            # Update the timestep if we're playing.
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames

            time.sleep(1.0 / gui_framerate.value)
        


if __name__ == '__main__':
    tyro.cli(main)