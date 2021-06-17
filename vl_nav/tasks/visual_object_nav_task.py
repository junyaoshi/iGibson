from gibson2.tasks.task_base import BaseTask
import pybullet as p
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

import numpy as np


class VisualObjectNavTask(BaseTask):
    """
    Object Navigation Task
    The goal is to navigate to one of the many loaded objects given object name
    """
    def __init__(self, env):
        """
        :param target_pos: [x, y, z]
        :param target_pos_vis_obj: an instance of gibson2.objects
        """
        super(VisualObjectNavTask, self).__init__(env)
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

        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', True
        )
        self.floor_num = 0

        self.load_visualization(env)

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