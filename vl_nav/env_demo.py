import os
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
yaml_filename = 'turtlebot_object_nav_stadium.yaml'
mode = 'gui'
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 40.0
device_idx = 1

# object params
visual_shape = p.GEOM_CYLINDER
cyl_length = 1.5
cyl_radius = 0.2
rgba_color = [0, 0, 1, 1.0]
initial_offset = [0, 0, cyl_length / 2.0]

# task params
task = 'visual_object_nav'
target_pos = [5, 5, 0]
fov = 60


def main():
    config_filename = os.path.join(gibson2.vlnav_config_path, yaml_filename)
    env = iGibsonEnv(
        config_file=config_filename,
        mode=mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx
    )
    # vis_obj = VisualMarker(
    #     visual_shape=visual_shape,
    #     rgba_color=rgba_color,
    #     radius=cyl_radius,
    #     length=cyl_length,
    #     initial_offset=initial_offset
    # )
    # vis_obj = RBOObject(name='tripod')
    # vis_obj = ArticulatedObject(
    #     filename=os.path.join(
    #         gibson2.ig_dataset_path, 'objects', 'basket',
    #         '2ae75c0f4bf43142e76bc197b3a3ffc0', '2ae75c0f4bf43142e76bc197b3a3ffc0.urdf'
    #     )
    # )
    vis_obj = iGisbonObject(name='speaker_system')
    if task == 'visual_point_nav_fixed':
        env_task = VisualPointNavFixedTask(
            env=env,
            target_pos=target_pos,
            target_pos_vis_obj=vis_obj
        )
        env.task = env_task
    elif task == 'visual_point_nav_random':
        env_task = VisualPointNavRandomTask(
            env=env,
            target_pos_vis_obj=vis_obj
        )
        env.task = env_task
    elif task == 'visual_object_nav':
        pass
    else:
        raise ValueError(f'Unrecoganized task: {task}')
    env.simulator.renderer.set_fov(fov)

    for j in range(20):
        env.reset()
        for i in range(500):
            with Profiler('Environment action step'):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)
                if reward < -0.05:
                    logging.info(f"episode: {env.current_episode} | timestep: {env.current_step} | "
                                 f"reward: {reward} | done: {done} | info: {info}")
                if done:
                    logging.info(
                        "Episode finished after {} timesteps".format(i + 1))
                    break
    env.close()


if __name__ == "__main__":
    main()
