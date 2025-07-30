from typing import Any, Sequence, Tuple, Dict, Union

import jax
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


COLLISION_REWARD = -1e2
# ACTION_MAP = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0)}
ACTION_MAP = jnp.array([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])


@struct.dataclass
class EnvParams(environment.EnvParams):
    n_agents: int = 3
    n_tokens: int = 5
    n_token_repeat: int = 2
    size: Tuple[int, int] = (5, 5)
    max_steps_in_episode: int = 100
    use_fixed_map: bool = False


@struct.dataclass
class EnvState(environment.EnvState):
    agent_positions: jax.Array
    token_positions: jax.Array
    time: int


class TokenEnvJax(environment.Environment[EnvState, EnvParams]):

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        actions: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        
        size = jnp.array(params.size)

        def move_agent(pos, a, subkey):
            return (pos + ACTION_MAP[a]) % size

        # split keys per agent
        keys = jax.random.split(key, params.n_agents)
        new_positions = jax.vmap(move_agent)(state.agent_positions, actions, keys)

        # Build a [n_agents, n_agents] equality matrix
        eq = (new_positions[:, None, :] == new_positions[None, :, :]).all(axis=-1)
        # Zero out self-comparisons
        eq = eq.at[jnp.diag_indices(params.n_agents)].set(False)
        collisions = jnp.any(eq, axis=1)

        rewards = jnp.where(collisions, COLLISION_REWARD, 0.0)

        new_state = EnvState(agent_positions=new_positions,
                             token_positions=state.token_positions,
                             time=state.time + 1)

        dones = jnp.logical_or(collisions, new_state.time >= params.max_steps_in_episode)
        #### Gymnax is giving the following error because dones is a vector
        # jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
        # The above exception was the direct cause of the following exception:
        # Traceback (most recent call last):
        #   File "/Users/beyazit/Documents/token-env/test_jax.py", line 36, in <module>
        #     test(env=token_env.TokenEnvJax())
        #   File "/Users/beyazit/Documents/token-env/test_jax.py", line 22, in test
        #     temp = env.step(action=action, state=state, key=key)
        #   File "/Users/beyazit/Documents/token-env/.venv/lib/python3.10/site-packages/gymnax/environments/environment.py", line 55, in step
        #     state = jax.tree.map(
        #   File "/Users/beyazit/Documents/token-env/.venv/lib/python3.10/site-packages/jax/_src/tree.py", line 155, in map
        #     return tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)
        #   File "/Users/beyazit/Documents/token-env/.venv/lib/python3.10/site-packages/gymnax/environments/environment.py", line 56, in <lambda>
        #     lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        # TypeError: select `which` must be scalar or have the same shape as cases, got `which` shape (3,) but case shape ().
        # So here you can either (i) return dones.all() and do some bookkeeping for alive agents and implement black death(ish) or
        # (ii) you can migrate to JaxMARL but JaxMARL is a lowkey mess (reset vs reset_env etc.) so prob do (i).
        # Doing (i) makes sure that this is a single agent env-alike but need to do stuff in the algorithm which is fine because
        # we need to copy-past-and-modify the algorithm anyway. In the alg, prob just running the policy on non-zero obs is enough.

        obs = self.get_obs(new_state, params, key=None)
        info = {}

        return obs, new_state, rewards, dones, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        key, sub = jax.random.split(key)
        total = params.n_agents + params.n_tokens * params.n_token_repeat
        grid_points = jnp.stack(jnp.meshgrid(jnp.arange(params.size[0]), jnp.arange(params.size[1])), -1)
        grid_flat = grid_points.reshape(-1,2)
        idx = jax.random.choice(sub, grid_flat.shape[0], (total,), replace=False)
        locs = grid_flat[idx]
        agent_pos = locs[:params.n_agents]
        token_flat = locs[params.n_agents:]
        token_pos = token_flat.reshape(params.n_tokens, params.n_token_repeat, 2)
        state = EnvState(agent_positions=agent_pos,
                         token_positions=token_pos,
                         time=jnp.array(0))
        obs = self.get_obs(state, params, key=None)
        return obs, state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        size = jnp.array(params.size)
        center = size // 2

        def obs_for_agent(i):
            base = jnp.zeros((params.n_tokens + params.n_agents - 1, *params.size), dtype=jnp.uint8)
            offset = center - state.agent_positions[i]

            def place_token(token_idx, val):
                rel = (state.token_positions[token_idx] + offset) % size
                return val.at[token_idx, rel[:, 0], rel[:, 1]].set(1)
            b1 = jax.lax.fori_loop(0, params.n_tokens, place_token, base)

            def place_other(other_idx, val):
                rel = (state.agent_positions[other_idx + (other_idx >= i)] + offset) % size
                return val.at[params.n_tokens + other_idx, rel[0], rel[1]].set(1)
            b2 = jax.lax.fori_loop(0, params.n_agents - 1, place_other, b1)

            return b2

        return jax.vmap(obs_for_agent)(jnp.arange(params.n_agents))

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        pass

    @property
    def name(self) -> str:
        """Environment name."""
        return "TokenEnv-Jax"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(ACTION_MAP)

    def action_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Box(
            low=0, high=len(ACTION_MAP), shape=(params.n_agents,), dtype=jnp.uint8
        )

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Box(
            low=0, high=1, shape=(params.n_agents, params.n_tokens + params.n_agents - 1, *params.size), dtype=jnp.uint8
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "agent_positions": spaces.Box(
                    low=jnp.array([0, 0]),
                    high=jnp.array(params.size),
                    shape=jnp.array(params.n_agents, 2),
                    dtype=jnp.int32
                ),
                "token_positions": spaces.Box(
                    low=jnp.array([0, 0]),
                    high=jnp.array(params.size),
                    shape=jnp.array(params.n_tokens, 2),
                    dtype=jnp.int32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode)
            }
        )

####

# COLLISION_REWARD = -1e2

# class TokenEnvJax(gym.Env):
#     metadata = {"render_modes": ["human", "ansi"], "name": "token_env"}

#     def __init__(
#         self,
#         n_agents: int = 1,
#         n_tokens: int = 10,
#         n_token_repeat: int = 2,
#         size: Tuple[int, int] = (7, 7),
#         timeout: int = 100,
#         use_fixed_map: bool = False,
#         slip_prob: Tuple[float, float] = (0.0, 0.0),
#         render_mode: str = "human"
#     ):
#         super().__init__()
#         assert size[0] % 2 == 1 and size[1] % 2 == 1, "Grid size must be odd"
#         assert n_tokens * n_token_repeat <= size[0] * size[1], "Grid size is not large enough"
#         assert render_mode in self.metadata["render_modes"]

#         self.n_agents = n_agents
#         self.n_tokens = n_tokens
#         self.n_token_repeat = n_token_repeat
#         self.size = size
#         self.timeout = timeout
#         self.use_fixed_map = use_fixed_map
#         self.slip_prob = slip_prob
#         self.render_mode = render_mode

#         self.possible_agents = [f"A_{i}" for i in range(self.n_agents)]

#         self.t = 0
#         self.action_map = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0)}
#         self.action_parser = {0: "DOWN", 1: "RIGHT", 2: "UP", 3: "LEFT", 4: "NOOP"}
#         self.agent_positions: Dict[str, np.ndarray] = {}
#         self.token_positions: Dict[int, Sequence[np.ndarray]] = {}

#         self.action_space: Dict[str, gym.spaces.Space] = gym.spaces.Dict({
#             agent: gym.spaces.Discrete(len(self.action_map)) for agent in self.possible_agents
#         })
#         self.observation_space: Dict[str, gym.spaces.Space] = gym.spaces.Dict({
#             agent: gym.spaces.Box(
#                 low=0, high=1, shape=(self.n_tokens + self.n_agents - 1, *self.size), dtype=np.uint8
#             ) for agent in self.possible_agents
#         })

#     def reset(
#         self,
#         seed: int | None = None,
#         options: dict[str, Any] | None = None
#     ) -> Dict[str, np.ndarray]:
#         np.random.seed(seed)

#         self.agents = self.possible_agents.copy() # Do this before calling _sample_map
#         self.agent_positions, self.token_positions = self._sample_map()

#         self.t = 0
#         observations = self._get_obs()
#         infos = {agent: {} for agent in self.agents}

#         return observations, infos

#     def step(
#         self,
#         actions: Dict[str, int]
#     ) -> Tuple[
#         Dict[str, np.ndarray],
#         Dict[str, Union[int, float]],
#         Dict[str, bool],
#         Dict[str, bool],
#         Dict[str, Dict[str, Any]]
#     ]:
#         # Update positions
#         for agent in self.agents:
#             act = actions[agent]

#             dx, dy = self.action_map[int(act)]
#             dx_slip = int(np.sign(self.slip_prob[0])) if np.random.random() < abs(self.slip_prob[0]) else 0
#             dy_slip = int(np.sign(self.slip_prob[1])) if np.random.random() < abs(self.slip_prob[1]) else 0

#             pos = self.agent_positions[agent]
#             self.agent_positions[agent][0] = (pos[0] + dx + dx_slip) % self.size[0]
#             self.agent_positions[agent][1] = (pos[1] + dy + dy_slip) % self.size[1]

#         observations = self._get_obs()

#         rewards = {}
#         for agent in self.agents:
#             if all(other_agent == agent or any(self.agent_positions[agent] != self.agent_positions[other_agent]) for other_agent in self.possible_agents):
#                 rewards[agent] = 0
#             else:
#                 rewards[agent] = COLLISION_REWARD # Collision with another agent!

#         terminations = {agent: self.t >= self.timeout or rewards[agent] == COLLISION_REWARD for agent in self.agents}
#         truncations = {agent: False for agent in self.agents}
#         infos = {agent: {} for agent in self.agents}

#         self.agents = [agent for agent in self.agents if not terminations[agent] and not truncations[agent]]

#         self.t += 1

#         return observations, rewards, terminations, truncations, infos

#     def _get_obs(self) -> Dict[str, np.ndarray]:
#         center = np.array([s // 2 for s in self.size])
#         observations: Dict[str, np.ndarray] = {}

#         for agent in self.agents:
#             obs = np.zeros((self.n_tokens + self.n_agents - 1, *self.size), dtype=np.uint8)
#             delta = center - self.agent_positions[agent]
#             for token in self.token_positions:
#                 for xy in self.token_positions[token]:
#                     rel = (xy + delta) % self.size
#                     obs[token, rel[0], rel[1]] = 1
#             for idx, other_agent in enumerate(filter(lambda a: a != agent, self.agents)):
#                 rel = (self.agent_positions[other_agent] + delta) % self.size
#                 obs[self.n_tokens + idx, rel[0], rel[1]] = 1
#             observations[agent] = obs
#         return observations

#     def _sample_map(
#         self
#     ) -> Tuple[Sequence[np.ndarray], Sequence[Tuple[int, np.ndarray]]]:
#         if self.use_fixed_map:
#             old_state = np.random.get_state()
#             np.random.seed(42)

#         total = self.n_agents + self.n_token_repeat * self.n_tokens
#         x = np.arange(self.size[0])
#         y = np.arange(self.size[1])
#         xx, yy = np.meshgrid(x, y)
#         grid = np.column_stack((xx.ravel(), yy.ravel()))

#         indices = np.random.choice(grid.shape[0], total, replace=False)
#         samples = grid[indices]

#         agents = {agent: samples[i] for i, agent in enumerate(self.agents)}
#         token_samples = samples[self.n_agents:]
#         np.random.shuffle(token_samples)
#         tokens = {token: [] for token in range(self.n_tokens)}
#         for i in range(self.n_token_repeat * self.n_tokens):
#             tokens[i % self.n_tokens].append(token_samples[i])

#         if self.use_fixed_map:
#             np.random.set_state(old_state)

#         return agents, tokens

#     def render(self):
#         empty_cell = "."
#         grid = np.full(self.size, empty_cell, dtype=object)

#         for token, positions in self.token_positions.items():
#             for pos in positions:
#                 grid[pos[0], pos[1]] = f"{token}"

#         for agent in self.possible_agents:
#             pos = self.agent_positions[agent]
#             current = grid[pos[0], pos[1]]
#             if current == empty_cell:
#                 grid[pos[0], pos[1]] = agent
#             else:
#                 grid[pos[0], pos[1]] = f"{agent},{current}"

#         max_width = max(len(str(cell)) for row in grid for cell in row)

#         out = ""
#         h_line = "+" + "+".join(["-" * (max_width + 2) for _ in range(self.size[1])]) + "+"
#         out += h_line + "\n"
#         for row in grid:
#             row_str = "| " + " | ".join(f"{str(cell):<{max_width}}" for cell in row) + " |"
#             out += row_str + "\n"
#             out += h_line + "\n"

#         if self.render_mode == "human":
#             print(out)
#         else:
#             return out

#     @staticmethod
#     def label_f(obs: np.ndarray, n_tokens: int) -> int:
#         if isinstance(obs, np.ndarray):
#             layer = obs[:n_tokens, obs.shape[1] // 2, obs.shape[2] // 2]
#             token = np.where(layer == 1)[0]
#             assert token.size < 2
#             return int(token[0]) if token.size == 1 else None

#     @staticmethod
#     def r_agg_f(token_env_reward, dfa_wrapper_reward) -> int:
#         if token_env_reward == COLLISION_REWARD:
#             return COLLISION_REWARD
#         else:
#             return dfa_wrapper_reward

