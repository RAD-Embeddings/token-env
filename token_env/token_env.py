import numpy as np
import gymnasium as gym
from typing import Any, Sequence, Tuple, Dict, Union

COLLISION_REWARD = -1e2

class TokenEnv(gym.Env):
    metadata = {"render_modes": [], "name": "token_env"}

    def __init__(
        self,
        n_agents: int = 1,
        n_tokens: int = 10,
        n_token_repeat: int = 2,
        size: Tuple[int, int] = (7, 7),
        timeout: int = 100,
        use_fixed_map: bool = False,
        slip_prob: Tuple[float, float] = (0.0, 0.0),
        render_mode: str | None = None
    ):
        super().__init__()
        assert size[0] % 2 == 1 and size[1] % 2 == 1, "Grid size must be odd"
        assert n_tokens * n_token_repeat <= size[0] * size[1], "Grid size is not large enough"

        self.n_agents = n_agents
        self.n_tokens = n_tokens
        self.n_token_repeat = n_token_repeat
        self.size = size
        self.timeout = timeout
        self.use_fixed_map = use_fixed_map
        self.slip_prob = slip_prob
        self.render_mode = render_mode

        # Agent identifiers
        self.possible_agents = [f"A_{i}" for i in range(self.n_agents)]

        # Internal state
        self.t = 0
        self.action_map = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0)}
        self.action_parser = {0: "DOWN", 1: "RIGHT", 2: "UP", 3: "LEFT", 4: "NOOP"}
        self.agent_positions: Dict[str, np.ndarray] = {}
        self.token_positions: Dict[int, Sequence[np.ndarray]] = {}

        self.action_space: Union[gym.spaces.Space, Dict[str, gym.spaces.Space]] = gym.spaces.Discrete(len(self.action_map)) if self.n_agents == 1 else gym.spaces.Dict({
            agent: gym.spaces.Discrete(len(self.action_map)) for agent in self.possible_agents
        })
        self.observation_space: Union[gym.spaces.Space, Dict[str, gym.spaces.Space]] = gym.spaces.Box(low=0, high=1, shape=(self.n_tokens, *self.size), dtype=np.uint8) if self.n_agents == 1 else gym.spaces.Dict({
            agent: gym.spaces.Box(
                low=0, high=1, shape=(self.n_tokens + self.n_agents - 1, *self.size), dtype=np.uint8
            ) for agent in self.possible_agents
        })
        # self.observation_space: Union[gym.spaces.Space, Dict[str, gym.spaces.Space]] = gym.spaces.Box(low=0, high=1, shape=(self.n_tokens, *self.size), dtype=np.uint8) if self.n_agents == 1 else gym.spaces.Dict({
        #     agent: gym.spaces.Box(
        #         low=0, high=1, shape=(self.n_tokens, *self.size), dtype=np.uint8
        #     ) for agent in self.possible_agents
        # })

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> Dict[str, np.ndarray]:
        np.random.seed(seed)

        self.agents = self.possible_agents.copy() # Do this before calling _sample_map
        self.agent_positions, self.token_positions = self._sample_map()

        self.t = 0
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        return (list(observations.values())[0] if self.n_agents == 1 else observations,
                list(infos.values())[0] if self.n_agents == 1 else infos)

    def step(
        self,
        actions: Union[int, Dict[str, int]]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, Union[int, float, None]],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]]
    ]:
        # Update positions
        for agent in self.agents:
            act = actions if self.n_agents == 1 else actions[agent]

            dx, dy = self.action_map[int(act)]
            dx_slip = int(np.sign(self.slip_prob[0])) if np.random.random() < abs(self.slip_prob[0]) else 0
            dy_slip = int(np.sign(self.slip_prob[1])) if np.random.random() < abs(self.slip_prob[1]) else 0

            pos = self.agent_positions[agent]
            self.agent_positions[agent][0] = (pos[0] + dx + dx_slip) % self.size[0]
            self.agent_positions[agent][1] = (pos[1] + dy + dy_slip) % self.size[1]

        observations = self._get_obs()

        rewards = {}
        for agent in self.agents:
            if all(other_agent == agent or any(self.agent_positions[agent] != self.agent_positions[other_agent]) for other_agent in self.possible_agents):
                rewards[agent] = 0
            else:
                rewards[agent] = COLLISION_REWARD # Collision with another agent!

        terminations = {agent: self.t >= self.timeout or rewards[agent] == COLLISION_REWARD for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.agents = [agent for agent in self.agents if not terminations[agent] and not truncations[agent]]

        self.t += 1

        return (list(observations.values())[0] if self.n_agents == 1 else observations,
                list(rewards.values())[0] if self.n_agents == 1 else rewards,
                list(terminations.values())[0] if self.n_agents == 1 else terminations,
                list(truncations.values())[0] if self.n_agents == 1 else truncations,
                list(infos.values())[0] if self.n_agents == 1 else infos)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        center = np.array([s // 2 for s in self.size])
        observations: Dict[str, np.ndarray] = {}

        for agent in self.agents:
            obs = np.zeros((self.n_tokens + self.n_agents - 1, *self.size), dtype=np.uint8)
            delta = center - self.agent_positions[agent]
            for token in self.token_positions:
                for xy in self.token_positions[token]:
                    rel = (xy + delta) % self.size
                    obs[token, rel[0], rel[1]] = 1
            for idx, other_agent in enumerate(filter(lambda a: a != agent, self.agents)):
                rel = (self.agent_positions[other_agent] + delta) % self.size
                obs[self.n_tokens + idx, rel[0], rel[1]] = 1
            observations[agent] = obs
        return observations

    def _sample_map(
        self
    ) -> Tuple[Sequence[np.ndarray], Sequence[Tuple[int, np.ndarray]]]:
        if self.use_fixed_map:
            old_state = np.random.get_state()
            np.random.seed(42)

        total = self.n_agents + self.n_token_repeat * self.n_tokens
        x = np.arange(self.size[0])
        y = np.arange(self.size[1])
        xx, yy = np.meshgrid(x, y)
        grid = np.column_stack((xx.ravel(), yy.ravel()))

        indices = np.random.choice(grid.shape[0], total, replace=False)
        samples = grid[indices]

        agents = {agent: samples[i] for i, agent in enumerate(self.agents)}
        token_samples = samples[self.n_agents:]
        np.random.shuffle(token_samples)
        tokens = {token: [] for token in range(self.n_tokens)}
        for i in range(self.n_token_repeat * self.n_tokens):
            tokens[i % self.n_tokens].append(token_samples[i])

        if self.use_fixed_map:
            np.random.set_state(old_state)

        return agents, tokens

    def render(self):
        empty_cell = "."
        grid = np.full(self.size, empty_cell, dtype=object)

        # Place tokens
        for token, positions in self.token_positions.items():
            for pos in positions:
                grid[pos[0], pos[1]] = f"{token}"

        # Place agents
        for agent in self.possible_agents:
            pos = self.agent_positions[agent]
            current = grid[pos[0], pos[1]]
            if current == empty_cell:
                grid[pos[0], pos[1]] = agent
            else:
                grid[pos[0], pos[1]] = f"{agent},{current}"

        # Compute the max width of any cell
        max_width = max(len(str(cell)) for row in grid for cell in row)

        # Render the grid
        h_line = "+" + "+".join(["-" * (max_width + 2) for _ in range(self.size[1])]) + "+"
        print(h_line)
        for row in grid:
            row_str = "| " + " | ".join(f"{str(cell):<{max_width}}" for cell in row) + " |"
            print(row_str)
            print(h_line)
        print()

    @staticmethod
    def label_f(obs: np.ndarray, n_tokens: int) -> int:
        if isinstance(obs, np.ndarray):
            layer = obs[:n_tokens, obs.shape[1] // 2, obs.shape[2] // 2]
            token = np.where(layer == 1)[0]
            assert token.size < 2
            return int(token[0]) if token.size == 1 else None

    @staticmethod
    def r_agg_f(token_env_reward, dfa_wrapper_reward) -> int:
        if token_env_reward == COLLISION_REWARD:
            return COLLISION_REWARD
        else:
            return dfa_wrapper_reward


