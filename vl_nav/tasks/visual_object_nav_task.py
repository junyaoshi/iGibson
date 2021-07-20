from igibson.tasks.task_base import BaseTask
import pybullet as p
import os
import igibson
import logging

from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.timeout import Timeout
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.point_goal import PointGoal
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from igibson.objects.visual_marker import VisualMarker

# from vl_nav.reward_functions.collision_reward import CollisionReward
from vl_nav.objects.igibson_object import iGisbonObject

import numpy as np
import time


class VisualObjectNavTask(BaseTask):
    """
    Object Navigation Task
    The goal is to navigate to one of the many loaded objects given object name
    """
    def __init__(self, env):
        """
        :param num_objects: number of objects in the environment
        """
        super(VisualObjectNavTask, self).__init__(env)
        self.env = env

        # minimum distance between object and initial robot position
        self.object_dist_min = self.config.get('object_dist_min', 1.0)
        # maximum distance between object and initial robot position
        self.object_dist_max = self.config.get('object_dist_max', 10.0)
        # minimum distance between objects
        # self.object_dist_keepout = self.config.get('object_dist_keepout', 3.0)

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
        self.goal_buffer_dist = self.config.get('goal_buffer_dist', 0.5)
        self.object_keepout_buffer_dist = self.config.get('object_keepout_buffer_dist', 0.5)

        self.visual_object_at_initial_pos = self.config.get(
            'visual_object_at_initial_pos', True
        )
        self.floor_num = 0

        self.num_objects = self.config.get('num_objects', 5)
        self.object_randomization_freq = self.config.get('object_randomization_freq', None)
        self.initialize_scene_objects()
        self.load_visualization(env)

        self.reset_time_vars()

    def reset_time_vars(self):
        self.start_time = time.time()
        self.reset_time = time.time()
        self.episode_time = time.time()

    def initialize_scene_objects(self):
        all_object_names = [
            os.path.basename(f.path)
            for f in os.scandir(os.path.join(igibson.ig_dataset_path, 'objects')) if f.is_dir()
        ]
        assert self.num_objects <= len(all_object_names)
        if self.object_randomization_freq is not None:
            self.object_names = np.random.choice(all_object_names, self.num_objects)
        else:
            # hardcoding case
            if self.num_objects == 5:
                self.object_names = ['standing_tv', 'piano', 'office_chair', 'toilet', 'speaker_system']
            else:
                self.object_names = all_object_names[-self.num_objects:]

        max_radius = 0.0
        self.object_dict = {}
        self.object_pos_dict = {}
        self.object_orn_dict = {}
        self.object_id_dict = {}
        for object_name in self.object_names:
            self.object_dict[object_name] = iGisbonObject(name=object_name)
            pybullet_id = self.env.simulator.import_object(self.object_dict[object_name])
            self.object_pos_dict[object_name] = self.object_dict[object_name].get_position()
            self.object_orn_dict[object_name] = self.object_dict[object_name].get_orientation()
            self.object_id_dict[object_name] = pybullet_id
            (xmax, ymax, _), (xmin, ymin, _) = p.getAABB(pybullet_id)
            object_max_radius = max(abs(xmax - xmin) / 2., abs(ymax - ymin) / 2.)
            max_radius = max(object_max_radius, max_radius)
        self.max_radius = max_radius
        self.dist_tol = self.max_radius + self.goal_buffer_dist
        self.termination_conditions[-1].dist_tol = self.dist_tol
        self.reward_functions[-1].dist_tol = self.dist_tol
        self.object_dist_keepout = self.max_radius * 2 + self.object_keepout_buffer_dist
        self.sample_goal_object()

    def sample_goal_object(self):
        goal_object_idx = np.random.randint(self.num_objects)
        self.target_name = self.object_names[goal_object_idx]
        self.target_object = self.object_dict[self.target_name]
        self.target_pos = self.object_pos_dict[self.target_name]
        self.target_orn = self.object_orn_dict[self.target_name]
        # one-hot encoding
        self.target_obs = np.eye(self.num_objects)[goal_object_idx]

    def sample_initial_pose_and_object_pos(self, env):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        object_pos_dict ={}
        object_orn_dict = {}

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
        for object_name in self.object_names:
            object_orn_dict[object_name] = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
            for _ in range(max_trials):
                _, object_pos = env.scene.get_random_point(floor=self.floor_num)
                valid_pos = placement_is_valid(object_pos, initial_pos)
                if valid_pos:
                    object_pos_dict[object_name] = object_pos
                    break
            if not valid_pos:
                print(f"WARNING: Failed to sample valid position for {object_name}")

        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, object_pos_dict, object_orn_dict

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])
        self.initial_pos_vis_obj.load()
        x, y, z = self.target_pos
        p.addUserDebugText(
            text=f'Target: {self.target_name}',
            textPosition=[x, y, z + 2],
            textSize=2
        )

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

    def get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path(env)
        return geodesic_dist

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        # print(f'positions: {env.robots[0].get_position()[:2]}, {self.target_pos[:2]}')
        return l2_distance(env.robots[0].get_position()[:2],
                           self.target_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        if self.reward_type == 'l2':
            return self.get_l2_potential(env)
        else:
            raise ValueError(f'Invalid reward type: {self.reward_type}')

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
        # time
        self.episode_time = time.time()
        logging.info(f'Episode time: {self.episode_time - self.start_time:.5f} | '
                     f'Reset time: {self.reset_time - self.start_time:.5f}')
        self.reset_time_vars()

        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            initial_pos, initial_orn, object_pos_dict, object_orn_dict = self.sample_initial_pose_and_object_pos(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn
            )
            for object_name, object in self.object_dict.items():
                reset_success &= env.test_valid_position(
                    object, object_pos_dict[object_name], object_orn_dict[object_name]
                )
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        p.removeState(state_id)

        self.object_pos_dict = object_pos_dict
        self.object_orn_dict = object_orn_dict
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn

        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        for object_name, object in self.object_dict.items():
            env.land(object, self.object_pos_dict[object_name], self.object_orn_dict[object_name])
        self.geodesic_dist = self.get_geodesic_potential(env)
        self.sample_goal_object()
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

        if env.mode == 'gui':
            p.removeAllUserDebugItems()
            x, y, z = self.target_pos
            p.addUserDebugText(
                text=f'Target: {self.target_name}',
                textPosition=[x, y, z + 2],
                textSize=2
            )

        self.reset_time = time.time()

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(VisualObjectNavTask, self).get_termination(
            env, collision_links, action, info)

        info['path_length'] = self.path_length
        if done:
            info['spl'] = float(info['success']) * \
                min(1.0, self.geodesic_dist / self.path_length)
        else:
            info['spl'] = 0.0

        return done, info

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        return self.target_obs

    def get_shortest_path(self,
                          env,
                          from_initial_pos=False,
                          entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return env.scene.get_shortest_path(
            self.floor_num, source, target, entire_path=entire_path)

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)

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

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
