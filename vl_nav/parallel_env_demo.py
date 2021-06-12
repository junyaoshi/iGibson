from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.envs.parallel_env import ParallelNavEnv
from time import time
import functools
import gibson2
import os
from gibson2.render.profiler import Profiler
import logging


def main():
    config_filename = os.path.join(gibson2.vlnav_config_path, 'turtlebot_point_nav_stadium.yaml')
    env_num = 64

    def load_env():
        return iGibsonEnv(config_file=config_filename, mode='headless')

    parallel_env = ParallelNavEnv([load_env] * env_num, blocking=False)

    from time import time
    for episode in range(5):
        start = time()
        print("episode {}".format(episode))
        parallel_env.reset()
        for i in range(300):
            res = parallel_env.step([[0.5, 0.5] for _ in range(env_num)])
            state, reward, done, _ = res[0]
            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                break
        print("{} elapsed".format(time() - start))


if __name__ == "__main__":
    main()
