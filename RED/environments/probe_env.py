import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionRewardEnv(gym.Env):
    """Gymnasium interface definition of the probe environment: Two actions,
    one observation for one timestep, action-dependent +1/-1 reward. If the
    agent can't find the best action, something must be wrong.
    """

    def __init__(self):
        super().__init__()
        # Observation is fixed to 0
        self._observation_space = spaces.Discrete(1)
        self._action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.reward_range = (-1, 1)

        # Random sample the correct action
        self.correct_action = random.choice([-1, 1])
        self.done = False

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed=None, options=None):
        self.done = False
        return np.array([0]), {}

    def step(self, action):
        assert self.done == False, "Reset the environment episode has ended."

        reward = -1
        # If action is in the same interval [1, 0) or (0, 1] as correct action
        # return a positive reward.
        if self.correct_action > 0 and action > 0:
            reward = 1
        if self.correct_action < 0 and action < 0:
            reward = 1

        self.done = True

        # Passing done twice for terminated and truncated
        return np.array([0]), reward, self.done, self.done, {}

    def render(self):
        pass

    def close(self):
        pass


class ActionObsRewardEnv(gym.Env):
    """Gymnasium interface definition of the probe environment: Two actions,
    two observation for one timestep, action-and observation dependent
    +1/-1 reward. If the agent can't find the best action, something must
    be wrong.
    """

    def __init__(self):
        super().__init__()
        self._observation_space = spaces.Discrete(2)
        self._action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.reward_range = (-1, 1)
        self.done = False

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self, seed=None, options=None):
        self.done = False
        # Correct action maps from observation,
        # Observation of +1 implies action should be +1
        self.correct_action = np.array([random.choice([-1, 1])])
        return self.correct_action, {}

    def step(self, action):
        assert self.done == False, "Reset the environment episode has ended."

        reward = -1
        # If action is in the same interval [1, 0) or (0, 1] as correct action
        # return a positive reward.
        if self.correct_action > 0 and action > 0:
            reward = 1
        if self.correct_action < 0 and action < 0:
            reward = 1

        self.correct_action = np.array([random.choice([-1, 1])])
        self.done = True

        # Passing done twice for terminated and truncated
        return self.correct_action, reward, self.done, self.done, {}

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = ActionRewardEnv()
    print("Action dependent reward")
    obs, _ = env.reset()
    print(f"Action taken: {env.correct_action=}")
    next_obs, reward, done, _, info = env.step(env.correct_action)
    print(f"{next_obs=}, {reward=}, {done=}\n")

    obs, _ = env.reset()
    print(f"Action taken:{-env.correct_action=}")
    next_obs, reward, done, _, info = env.step(-env.correct_action)
    print(f"{next_obs=}, {reward=}, {done=}\n")

    env = ActionObsRewardEnv()
    print("Observation-Action dependent reward")
    obs, _ = env.reset()
    print(f"Correct action: {obs=}")
    next_obs, reward, done, _, info = env.step(obs)
    print(f"{next_obs=}, {reward=}, {done=}")
