import random
import gym
import gym.spaces
import numpy as np

class SeedWrapper(gym.Wrapper):
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

