from gibson2.tasks.task_base import BaseTask
import pybullet as p
import os
import gibson2
import logging

from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.reward_functions.potential_reward import PotentialReward
from gibson2.reward_functions.collision_reward import CollisionReward
from gibson2.reward_functions.point_goal_reward import PointGoalReward

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.articulated_object import ArticulatedObject
from vl_nav.objects.igibson_object import IGisbonObject

import numpy as np


class VisualObjectNavTask(BaseTask):
    """
    Object Navigation Task
    The goal is to navigate to one of the many loaded objects given object name
    """
    def __init__(self, env, num_objects=5, object_randomization_freq=None):
        """
        :param num_objects: number of objects in the environment
        """
        super(VisualObjectNavTask, self).__init__(env)

        # minimum distance between object and initial robot position
        self.object_dist_min = self.config.get('object_dist_min', 1.0)
        # maximum distance between object and initial robot position
        self.object_dist_max = self.config.get('object_dist_max', 10.0)
        # minimum distance between objects
        self.object_dist_keepout = self.config.get('object_dist_keepout', 3.0)

        self.reward_type = self.config.get('reward_type', 'l2')
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))
        self.goal_format = self.config.get('goal_format', 'polar')
        self.dist_tol = self.termination_conditions[-1].dist_tol

        self.visual_object_at_initial_pos = self.config.get(
            'visual_object_at_initial_pos', True
        )
        self.floor_num = 0

        self.num_objects = num_objects
        self.object_randomization_freq = object_randomization_freq
        self.initialize_scene_objects()
        self.load_visualization(env)

    def initialize_scene_objects(self):
        all_object_names = [
            os.path.basename(f.path)
            for f in os.scandir(os.path.join(gibson2.ig_dataset_path, 'objects')) if f.is_dir()
        ]
        assert self.num_objects <= len(all_object_names)
        if self.object_randomization_freq is not None:
            self.object_names = np.random.choice(all_object_names, self.num_objects)
        else:
            # hardcoding case
            if self.num_objects == 5:
                self.object_names = ['standing_tv', 'piano', 'mirror', 'fridge', 'chair']
            else:
                self.object_names = all_object_names[-self.num_objects:]

        self.object_dict = {}
        self.object_z_offset_dict = {}
        self.object_pos_dict = {}
        for object_name in self.object_names:
            self.object_dict[object_name] = IGisbonObject(name=object_name)
            self.env.simulator.import_object(self.object_dict[object_name])
            object_zmin = p.getAABB(self.object_dict[object_name].body_id)[0][2]
            if object_zmin < 0:
                object_z_offset = -object_zmin
                x, y, z = self.object_dict[object_name].get_position()
                new_pos = [x, y, z + object_z_offset]
                self.object_dict[object_name].set_position(new_pos)
                self.object_z_offset_dict[object_name] = object_z_offset
                self.object_pos_dict[object_name] = new_pos

    def sample_initial_pose_and_object_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        object_pos_dict ={}

        def placement_is_valid(pos, initial_pos):
            dist = l2_distance(pos, initial_pos)
            if dist < self.object_dist_min or dist > self.object_dist_max:
                return False
            for object_name, object_pos in object_pos_dict.items():
                dist = l2_distance(pos, object_pos)
                if dist < self.object_dist_keepout:
                    return False
            return True

        _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        max_trials = 500
        dist = 0.0
        for object_name in self.object_names:
            for _ in range(max_trials):
                _, object_pos = env.scene.get_random_point(floor=self.floor_num)
                if placement_is_valid(object_pos, initial_pos):
                    x, y, z = object_pos
                    new_object_pos = np.array([x, y, z + self.object_z_offset_dict[object_name]])
                    object_pos_dict[object_name] = new_object_pos
                    break
            if not placement_is_valid(object_pos, initial_pos):
                print(f"WARNING: Failed to sample valid position for {object_name}")

        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, object_pos_dict

    def reset_scene(self, env):
        """
        Task-specific scene reset: get a random floor number first

        :param env: environment instance
        """
        self.floor_num = env.scene.get_random_floor()
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
                Reset robot initial pose.
                Sample initial pose and target position, check validity, and land it.

                :param env: environment instance
                """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, object_pos_dict = self.sample_initial_pose_and_object_pos(env)
            reset_success = env.test_valid_position(env.robots[0], initial_pos, initial_orn)
            for object_name, object in self.object_dict.items():
                reset_success &= env.test_valid_position(object, object_pos_dict[object_name])
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.object_pos_dict = object_pos_dict
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn

        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        for object_name, object in self.object_dict.items():
            self.object_dict[object_name].set_position(self.object_pos_dict[object_name])
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])

        if self.target_visual_object_visible_to_agent:
            env.simulator.import_object(self.target_pos_vis_obj)
        else:
            self.target_pos_vis_obj.load()
        self.initial_pos_vis_obj.load()

        if env.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=[0, 1, 0, 0.3],
                radius=0.1,
                length=cyl_length,
                initial_offset=[0, 0, cyl_length / 2.0])
                for _ in range(self.num_waypoints_vis)]
            for waypoint in self.waypoints_vis:
                waypoint.load()

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        for object_name, object in self.object_dict.items():
            self.object_dict[object_name].set_position(self.object_pos_dict[object_name])

        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([shortest_path[i][0],
                                  shortest_path[i][1],
                                  floor_height]))
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(
                    pos=np.array([0.0, 0.0, 100.0]))