from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import gibson2
from gibson2.objects.cube import Cube
from gibson2.objects.visual_marker import VisualMarker
import os
import pybullet as p
from gibson2.render.profiler import Profiler
import logging
import numpy as np
from vl_nav.tasks.visual_point_nav_fixed_task import VisualPointNavFixedTask
from vl_nav.tasks.visual_point_nav_random_task import VisualPointNavRandomTask

# params
yaml_filename = 'turtlebot_point_nav_stadium.yaml'
mode = 'gui'
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 120.0


def main():
    config_filename = os.path.join(gibson2.vlnav_config_path, yaml_filename)
    env = iGibsonEnv(
        config_file=config_filename,
        mode=mode,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
    )
    cyl_length = 1.5
    cyl_radius = 0.5
    vis_obj = VisualMarker(
        visual_shape=p.GEOM_CYLINDER,
        rgba_color=[0, 0, 1, 1.0],
        radius=cyl_radius,
        length=cyl_length,
        initial_offset=[0, 0, cyl_length / 2.0]
    )
    task = VisualPointNavRandomTask(
        env=env,
        target_pos=[5, 5, 0],
        target_pos_vis_obj=vis_obj,
    )
    env.task = task

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
