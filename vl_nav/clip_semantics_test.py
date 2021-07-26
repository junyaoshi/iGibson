import os

import cv2
import numpy as np
from time import time
from tqdm import tqdm, trange

import igibson
import pybullet as p
import logging

from igibson.objects.cube import Cube
from igibson.objects.visual_marker import VisualMarker
from igibson.objects.articulated_object import ArticulatedObject, RBOObject
from igibson.render.profiler import Profiler

from vl_nav.tasks.visual_point_nav_fixed_task import VisualPointNavFixedTask
from vl_nav.tasks.visual_point_nav_random_task import VisualPointNavRandomTask
from vl_nav.tasks.visual_object_nav_task import VisualObjectNavTask
from vl_nav.objects.igibson_object import iGisbonObject
from vl_nav.envs.igibson_env import iGibsonEnv

import clip
import torch
from PIL import Image
from matplotlib import pyplot as plt
import gym
from scipy.stats import entropy
from alf.utils.video_recorder import VideoRecorder

# clip params
scene_id = "Rs_int"
use_colors = False
target_name = 'grey toilet' if use_colors else 'toilet'
plot_file = f'/home/junyaoshi/Desktop/CLIP_semantics_plots/{scene_id}_{target_name.replace(" ", "_")}' \
            f'{"_with_colors_" if use_colors else "_"}0723_3.png'
num_random_pts = 10
delta_angle = np.pi / 8
fps = 10
colors = [
    "blue",
    "yellow",
    "green",
    "white",
    "brown",
    "grey",
    "purple",
    "red",
    "orange",
    "pink",
    "black"
]

# env params
yaml_filename = 'turtlebot_clip.yaml'
mode = 'gui'
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 120.0
device_idx = 1
fov = 75

# keys to action
robot_keys_to_actions = {
    'w': [0.5, 0.5],
    's': [-0.5, -0.5],
    'a': [0.2, -0.2],
    'd': [-0.2, 0.2],
    'f': [0, 0]
}


def main():
    xs, ys, scores = [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    config_filename = os.path.join(igibson.vlnav_config_path, yaml_filename)
    env = iGibsonEnv(
        config_file=config_filename,
        mode=mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx
    )

    classes = list(env.scene.objects_by_category.keys())
    if use_colors:
        classes = [f'{color} {label}' for color in colors for label in classes]
    target_index = classes.index(target_name)
    target_x, target_y = p.getBasePositionAndOrientation(
        env.scene.objects_by_category[target_name.split(' ')[-1]][0].body_id[0]
    )[0][:2]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classes]).to(device)
    # Calculate text features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    def get_robot_cam_frame():
        frames = env.simulator.renderer.render_robot_cameras(modes=('rgb'))
        if len(frames) > 0:
            frame = cv2.cvtColor(np.concatenate(
                frames, axis=1), cv2.COLOR_RGB2BGR)
            return frame
        return None

    def get_z_orn(env):
        return p.getEulerFromQuaternion(env.robots[0].get_orientation())[2]

    def get_xy(env):
        return env.robots[0].get_position()[:2]

    for i in trange(num_random_pts):
        env.reset()
        x, y = get_xy(env)
        xs.append(x)
        ys.append(y)
        orn_cur = get_z_orn(env)
        orn_final = orn_cur + np.pi * 2
        orn_target = orn_cur + delta_angle
        max_score = -np.inf
        while orn_cur < orn_final:
            env.step([0.1, -0.1])
            orn_cur = get_z_orn(env)
            if orn_cur >= orn_target:
                frame = get_robot_cam_frame()
                assert frame is not None
                frame = (frame[:, :, :3] * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame, 'RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)

                # Calculate image features
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Pick the top 5 most similar labels for the image
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                target_score = similarity[:, target_index]
                max_score = max(max_score, target_score)

        scores.append(max_score)

    scores = np.array(scores)
    plt.figure(figsize=(50, 30))
    plt.scatter(xs, ys, s=scores, alpha=0.5)
    plt.plot(target_x, target_y, '*', s=20)
    plt.grid(True)
    plt.title(f'{target_name} heat map in {scene_id}')
    plt.tight_layout()
    plt.savefig(plot_file)


if __name__ == "__main__":
    main()
