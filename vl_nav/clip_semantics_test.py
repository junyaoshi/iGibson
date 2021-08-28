import os

import cv2
import numpy as np
from time import time
from tqdm import tqdm, trange

import igibson
import pybullet as p
import logging
import xml.etree.ElementTree as ET

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
import datetime
import gym
from scipy.stats import entropy
from alf.utils.video_recorder import VideoRecorder

# clip params
scene_id = "Rs_int"
use_colors = False
# note: need to use underscore for category, e.g. towel_rack
target_name = 'grey bed' if use_colors else 'bed'
data_dir = f'/home/junyaoshi/Desktop/CLIP_semantics_plots/{scene_id}_{target_name.replace(" ", "_")}'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
date = str(datetime.datetime.now())
date = date[2:4] + date[5:7] + date[8:10] + '_' + date[11:13] + date[14:16] + date[17:19]
data_dir = os.path.join(data_dir, 'D' + date)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
num_random_pts = 100
delta_angle = np.pi / 2. # how often images are taken for CLIP while rotating the robot
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
mode = 'headless'
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
    target_category = target_name.split(' ')[-1]
    # TO-DO: this only handles one object in the category
    target_x, target_y = p.getBasePositionAndOrientation(
        env.scene.objects_by_category[target_category][0].body_id[0]
    )[0][:2]
    if target_x == 0. and target_y == 0.:
        links = env.scene.scene_tree.findall('link')
        links_with_category = [link for link in links if "category" in link.attrib]
        target_link = [link for link in links_with_category if link.attrib["category"] == target_category][0]
        object_name = target_link.attrib['name']

        # The joint location is given wrt the bounding box center but we need it wrt to the base_link frame
        joint_connecting_embedded_link = \
            [joint for joint in env.scene.scene_tree.findall("joint")
             if joint.find("child").attrib["link"]
             == object_name][0]
        joint_xyz = np.array([float(val) for val in joint_connecting_embedded_link.find(
            "origin").attrib["xyz"].split(" ")])
        target_x, target_y, _ = joint_xyz
    np.save(os.path.join(data_dir, '..', 'target_xy.npy'), np.array([target_x, target_y]))
    # remove underscore
    classes = [(' ').join(c.split('_')) for c in classes]
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

    def get_delta_orn(orn_initial, orn_final):
        if orn_initial > 0. > orn_final:
            return orn_final + 2 * np.pi - orn_initial
        else:
            return orn_final - orn_initial

    for i in trange(num_random_pts):
        print(f'Collecting data point {i}')
        # reset
        env.robots[0].set_position([100.0, 100.0, 100.0])
        env.task.reset_agent(env)
        env.simulator.sync()

        x, y = get_xy(env)
        orn_cur = get_z_orn(env)
        delta_orn_interval = 0.  # how much the robot has rotated in this interval
        delta_orn_total = 0.  # how much the robot has rotated overall
        max_score = -np.inf
        steps = 0

        # TO-DO: simplify code here, merge it with while loop
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
        target_score = similarity[:, target_index].item()
        max_score = max(max_score, target_score)

        print(f'Evaluated CLIP at angle: {orn_cur}, score: {target_score}')
        delta_orn_interval = 0.

        while delta_orn_total < np.pi * 2:
            env.step([-0.2, 0.2])
            steps += 1
            # stuck on obstacle
            if steps > 100:
                print('Stuck at obstacle... discarding this data point')
                break

            orn_prev = orn_cur
            orn_cur = get_z_orn(env)
            delta_orn = get_delta_orn(orn_prev, orn_cur)

            delta_orn_interval += delta_orn
            delta_orn_total += delta_orn

            if delta_orn_interval >= delta_angle:
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
                target_score = similarity[:, target_index].item()
                max_score = max(max_score, target_score)

                print(f'Evaluated CLIP at angle: {orn_cur}, score: {target_score}')
                delta_orn_interval = 0.
        if steps <= 100:
            print(f'Finished collecting data point {i}, max score: {max_score}')
            xs.append(x)
            ys.append(y)
            scores.append(max_score)

    np.save(os.path.join(data_dir, 'xs.npy'), np.array(xs))
    np.save(os.path.join(data_dir, 'ys.npy'), np.array(ys))
    np.save(os.path.join(data_dir, 'scores.npy'), np.array(scores))

if __name__ == "__main__":
    main()
