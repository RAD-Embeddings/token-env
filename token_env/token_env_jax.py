from typing import Any, Sequence, Tuple, Dict, Union

import jax
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces

# ACTION_MAP = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0)}
ACTION_MAP = jnp.array([(1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)])


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100
    collision_reward = -1e2
    black_death = True


@struct.dataclass
class EnvState(environment.EnvState):
    agent_positions: jax.Array
    token_positions: jax.Array
    is_alive: jax.Array
    time: int


class TokenEnvJax(environment.Environment[EnvState, EnvParams]):

    def __init__(
        self,
        n_agents: int = 3,
        n_tokens: int = 5,
        n_token_repeat: int = 2,
        grid_shape: Tuple[int, int] = (5, 5),
        use_fixed_map: bool = False
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_tokens = n_tokens
        self.n_token_repeat = n_token_repeat
        self.grid_shape = grid_shape
        self.grid_shape_arr = jnp.array(self.grid_shape)
        self.use_fixed_map = use_fixed_map

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

        def move_agent(pos, a, is_agent_alive):
            return jnp.where(
                is_agent_alive,
                (pos + jnp.array(ACTION_MAP[a])) % self.grid_shape_arr,
                pos
            )

        # new_positions = jax.vmap(move_agent, in_axes=(0, None, 0))(state.agent_positions, actions, state.is_alive)
        new_positions = jax.vmap(move_agent, in_axes=(0, 0, 0))(state.agent_positions, actions, state.is_alive)

        eq = (new_positions[:, None, :] == new_positions[None, :, :]).all(axis=-1)
        eq = eq.at[jnp.diag_indices(self.n_agents)].set(False)
        collisions = jnp.any(eq, axis=1)

        rewards = jnp.where(jnp.logical_and(state.is_alive, collisions), params.collision_reward, 0.0)

        new_state = EnvState(agent_positions=new_positions,
                             token_positions=state.token_positions,
                             is_alive=jnp.logical_and(state.is_alive, jnp.logical_not(collisions)),
                             time=state.time + 1)

        dones = jnp.logical_or(collisions, new_state.time >= params.max_steps_in_episode)

        done = dones.all()

        obs = self.get_obs(state=new_state, params=params)
        info = {}

        # return obs, new_state, 0, done, info
        return obs, new_state, rewards, done, info

    def reset_env(
        self,
        key: jax.Array,
        params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        key, subkey = jax.random.split(key)
        total = self.n_agents + self.n_tokens * self.n_token_repeat
        grid_points = jnp.stack(jnp.meshgrid(jnp.arange(self.grid_shape[0]), jnp.arange(self.grid_shape[1])), -1)
        grid_flat = grid_points.reshape(-1,2)
        idx = jax.random.choice(subkey, grid_flat.shape[0], (total,), replace=False)
        locs = grid_flat[idx]
        agent_pos = locs[:self.n_agents]
        token_flat = locs[self.n_agents:]
        token_pos = token_flat.reshape(self.n_tokens, self.n_token_repeat, 2)
        state = EnvState(agent_positions=agent_pos,
                         token_positions=token_pos,
                         is_alive=jnp.array([True for _ in jnp.arange(self.n_agents)]),
                         time=0)
        obs = self.get_obs(state=state, params=params)
        return obs, state

    def get_obs(
        self,
        state: EnvState,
        params: EnvParams
    ) -> jax.Array:

        def obs_for_agent(i):
            base = jnp.zeros((self.n_tokens + self.n_agents - 1, *self.grid_shape), dtype=jnp.uint8)
            offset = (self.grid_shape_arr // 2) - state.agent_positions[i]

            def place_token(token_idx, val):
                rel = (state.token_positions[token_idx] + offset) % self.grid_shape_arr
                return val.at[token_idx, rel[:, 0], rel[:, 1]].set(1)
            b1 = jax.lax.fori_loop(0, self.n_tokens, place_token, base)

            def place_other(other_idx, val):
                rel = (state.agent_positions[other_idx + (other_idx >= i)] + offset) % self.grid_shape_arr
                return val.at[self.n_tokens + other_idx, rel[0], rel[1]].set(1)
            b2 = jax.lax.fori_loop(0, self.n_agents - 1, place_other, b1)

            return jnp.where(jnp.logical_or(jnp.logical_not(params.black_death), state.is_alive[i]), b2, base)

        return jax.vmap(obs_for_agent)(jnp.arange(self.n_agents))

    @property
    def name(self) -> str:
        return "TokenEnv-Jax"

    @property
    def num_actions(self) -> int:
        return len(ACTION_MAP)

    def action_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Box(
            low=0, high=len(ACTION_MAP), shape=(self.n_agents,), dtype=jnp.uint8
        )

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Box(
            low=0, high=1, shape=(self.n_agents, self.n_tokens + self.n_agents - 1, *self.grid_shape), dtype=jnp.uint8
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "agent_positions": spaces.Box(
                    low=jnp.array([0, 0]),
                    high=self.grid_shape_arr,
                    shape=jnp.array(self.n_agents, 2),
                    dtype=jnp.int32
                ),
                "token_positions": spaces.Box(
                    low=jnp.array([0, 0]),
                    high=self.grid_shape_arr,
                    shape=jnp.array(self.n_tokens, 2),
                    dtype=jnp.int32
                ),
                "time": spaces.Discrete(params.max_steps_in_episode)
            }
        )
