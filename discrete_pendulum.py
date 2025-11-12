import gymnasium as gym
import numpy as np

class DiscretePendulum(gym.ActionWrapper):
    
    def __init__(self, env, num_actions=5):
        super().__init__(env)
        self.num_actions = num_actions
        self.action_space = gym.spaces.Discrete(num_actions)
        self.action_map = np.linspace(-2, 2, num_actions)

    def action(self, action):
        continuous_action = self.action_map[action]
        return np.array([continuous_action], dtype=np.float32)

    def reverse_action(self, action):
        idx = np.argmin(np.abs(self.action_map - action[0]))
        return idx

def make_pendulum(num_discrete_actions=5, render_mode=None):
    env = gym.make('Pendulum-v1', render_mode=render_mode)
    env = DiscretePendulum(env, num_actions=num_discrete_actions)
    return env