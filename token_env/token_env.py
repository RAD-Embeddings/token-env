import numpy as np
import gymnasium as gym
from gymnasium import spaces

from typing import Any

class TokenEnv(gym.Env):
    def __init__(
        self,
        n_tokens: int = 10,
        size: tuple[int, int] = (7, 7),
        timeout: int = 100,
        use_fixed_map: bool = False
    ):
        super().__init__()
        assert size[0] % 2 == 1 and size[1] % 2 == 1
        self.n_tokens = n_tokens
        self.size = size
        self.timeout = timeout
        self.use_fixed_map = use_fixed_map
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_tokens, *self.size), dtype=np.uint8)
        self.t = 0
        self.agent = None
        self.action_map = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        if self.use_fixed_map:
            np.random.seed(42)
            _, self.tokens = self._sample_map()
            np.random.seed(None)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        np.random.seed(seed)
        if self.use_fixed_map:
            self.agent = (0, 0)
        else:
            self.agent, self.tokens = self._sample_map()
        self.t = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        dx, dy = self.action_map[action]
        agent_x = (self.agent[0] + dx + self.size[0]) % self.size[0]
        agent_y = (self.agent[1] + dy + self.size[1]) % self.size[1]
        self.agent = (agent_x, agent_y)
        obs = self._get_obs()
        reward = 0
        self.t += 1
        done = self.t >= self.timeout
        return obs, reward, done, False, {}

    def _get_obs(self) -> np.ndarray:
        center_x = self.size[0] // 2
        center_y = self.size[1] // 2
        delta = center_x - self.agent[0], center_y - self.agent[1]
        obs = np.zeros(shape=(self.n_tokens, *self.size), dtype=np.uint8)
        for i, xy in self.tokens:
            rel_xy = (xy + delta + self.size) % self.size
            obs[i, *rel_xy] = 1
        return obs

    def _sample_map(self):
        x = np.arange(self.size[0])
        y = np.arange(self.size[1])
        xx, yy = np.meshgrid(x, y)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        indices = np.random.choice(grid.shape[0], 2 * self.n_tokens + 1, replace=False)
        samples = grid[indices]
        agent = samples[0]
        samples = samples[1:]
        np.random.shuffle(samples)
        tokens = [(i % self.n_tokens, samples[i]) for i in range(2 * self.n_tokens)]
        return agent, tokens

    @staticmethod
    def label_f(obs: np.ndarray) -> int | None:
        token = np.where(obs[:, obs.shape[1] // 2, obs.shape[2] // 2] == 1)[0]
        assert token.size < 2
        if token.size == 1:
            return token.item()
        return None
