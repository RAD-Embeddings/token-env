import gymnasium as gym
from token_env import TokenEnv
from pettingzoo.utils import ParallelEnv


def gym2zoo(env: TokenEnv):
    return ParallelTokenEnv(env)

class ParallelTokenEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "parallel_token_env"}

    def __init__(self, env: TokenEnv):
        self._env = env

        # Copy over agent ordering and spaces
        self.possible_agents = self._env.possible_agents[:]
        self.agents = self.possible_agents[:]
        self.action_spaces = self._env.action_space
        self.observation_spaces = self._env.observation_space

    def reset(self, seed=None, options=None):
        obs, infos = self._env.reset(seed=seed, options=options)
        self.agents = self._env.agents[:]
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self._env.step(actions)
        self.agents = self._env.agents[:]
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

