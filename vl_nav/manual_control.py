import os

import cv2
import numpy as np
from time import time

import gibson2
import pybullet as p
import logging

from gibson2.objects.cube import Cube
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.articulated_object import ArticulatedObject, RBOObject
from gibson2.render.profiler import Profiler

from vl_nav.tasks.visual_point_nav_fixed_task import VisualPointNavFixedTask
from vl_nav.tasks.visual_point_nav_random_task import VisualPointNavRandomTask
from vl_nav.tasks.visual_object_nav_task import VisualObjectNavTask
from vl_nav.objects.igibson_object import iGisbonObject
from vl_nav.envs.igibson_env import iGibsonEnv

# env params
yaml_filename = 'turtlebot_clip.yaml'
mode = 'gui'
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 120.0
device_idx = 1
fov = 75

# data saving
trial_name = '0630_1'
data_dir = os.path.join(gibson2.vlnav_path, 'manual_control_images', trial_name)
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# keys to action
turtlebot_keys_to_actions = {
    'w': [0.5, 0.5],
    's': [-0.5, -0.5],
    'a': [0.2, -0.2],
    'd': [-0.2, 0.2],
    'f': [0, 0]
}


def main():
    config_filename = os.path.join(gibson2.vlnav_config_path, yaml_filename)
    env = iGibsonEnv(
        config_file=config_filename,
        mode=mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx
    )
    env.reset()
    robot = env.config.get('robot')
    if robot == 'Turtlebot':
        keys_to_actions = turtlebot_keys_to_actions
    elif robot == 'Locobot':
        keys_to_actions = turtlebot_keys_to_actions
    else:
        raise ValueError(f'Unknown robot: {robot}')

    categories = list(env.scene.objects_by_category.keys())
    categories_fname = os.path.join(data_dir, f'{env.config.get("scene_id")}_categories.txt')
    with open(categories_fname, 'w') as output:
        for name in categories:
            output.write(name + '\n')

    def get_key_pressed():
        pressed_keys = []
        events = p.getKeyboardEvents()
        key_codes = events.keys()
        for key in key_codes:
            pressed_keys.append(key)
        return pressed_keys

    def get_robot_cam_frame():
        frames = env.simulator.renderer.render_robot_cameras(modes=('rgb'))
        if len(frames) > 0:
            frame = cv2.cvtColor(np.concatenate(
                frames, axis=1), cv2.COLOR_RGB2BGR)
            return frame
        return None

    running = True
    while running:
        # detect pressed keys
        pressed_keys = get_key_pressed()
        if ord('r') in pressed_keys:
            print('Reset the environment...')
            env.reset()
            pressed_keys = []
        if ord('p') in pressed_keys:
            print('Shutting down the environment...')
            env.close()
            running = False
            pressed_keys = []
        if ord('c') in pressed_keys:
            print('Saving an image from RobotView...')
            img_path = os.path.join(data_dir, f'img_{int(time())}.png')
            frame = get_robot_cam_frame()
            assert frame is not None
            frame = (frame[:, :, :3] * 255).astype(np.uint8)
            cv2.imwrite(img_path, frame)
            pressed_keys = []
        if ord('8') in pressed_keys:
            print('Forward...')
            env.step(keys_to_actions['w'])
            # pressed_keys = []
        if ord('5') in pressed_keys:
            print('Backward...')
            env.step(keys_to_actions['s'])
            # pressed_keys = []
        if ord('6') in pressed_keys:
            print('Left...')
            env.step(keys_to_actions['a'])
            # pressed_keys = []
        if ord('4') in pressed_keys:
            print('Right...')
            env.step(keys_to_actions['d'])
            # pressed_keys = []
        if ord('2') in pressed_keys:
            print('Staying still...')
            env.step(keys_to_actions['f'])
            pressed_keys = []


if __name__ == "__main__":
    main()
