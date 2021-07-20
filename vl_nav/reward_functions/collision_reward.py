from igibson.reward_functions.collision_reward import CollisionReward


class VLNCollisionReward(CollisionReward):
    """
    Collision reward
    Penalize robot collision. Typically collision_reward_weight is negative.
    """

    def __init__(self, config):
        super(CollisionReward, self).__init__(config)
        self.collision_ignore_steps = self.config.get(
            'collision_ignore_steps', 0
        )

    def get_reward(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """

        has_collision = float(len(env.collision_links) > 0) if env.current_step > self.collision_ignore_steps else 0.
        return has_collision * self.collision_reward_weight
