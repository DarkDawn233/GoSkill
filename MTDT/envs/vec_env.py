from gym.vector.async_vector_env import AsyncVectorEnv

class VecEnv(AsyncVectorEnv):
    def _check_spaces(self):
        # Not check the state and action space
        return