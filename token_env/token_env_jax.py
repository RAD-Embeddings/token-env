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
    black_death: bool = True


@struct.dataclass
class EnvState(environment.EnvState):
    agent_positions: jax.Array
    token_positions: jax.Array
    is_alive: jax.Array
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

        # def move_agent(pos, a):
        #     # TODO: Modify the following like so that it is `(pos + ACTION_MAP[a]) % size if is_alive else pos`
        #     return (pos + ACTION_MAP[a]) % size

        # new_positions = jax.vmap(move_agent)(state.agent_positions, actions)

        def move_agent(pos, a, is_agent_alive):
            return jnp.where(
                is_agent_alive,
                (pos + jnp.array(ACTION_MAP[a])) % size,
                pos
            )

        new_positions = jax.vmap(move_agent)(state.agent_positions, actions, state.is_alive)

        # Build a [n_agents, n_agents] equality matrix
        eq = (new_positions[:, None, :] == new_positions[None, :, :]).all(axis=-1)
        # Zero out self-comparisons
        eq = eq.at[jnp.diag_indices(params.n_agents)].set(False)
        collisions = jnp.any(eq, axis=1)

        rewards = jnp.where(jnp.logical_and(state.is_alive, collisions), COLLISION_REWARD, 0.0)

        new_state = EnvState(agent_positions=new_positions,
                             token_positions=state.token_positions,
                             is_alive=jnp.logical_and(state.is_alive, jnp.logical_not(collisions)),
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
        # Going with (i).

        done = dones.all()

        obs = self.get_obs(state=new_state, params=params)
        info = {}

        return obs, new_state, rewards, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        key, subkey = jax.random.split(key)
        total = params.n_agents + params.n_tokens * params.n_token_repeat
        grid_points = jnp.stack(jnp.meshgrid(jnp.arange(params.size[0]), jnp.arange(params.size[1])), -1)
        grid_flat = grid_points.reshape(-1,2)
        idx = jax.random.choice(subkey, grid_flat.shape[0], (total,), replace=False)
        locs = grid_flat[idx]
        agent_pos = locs[:params.n_agents]
        token_flat = locs[params.n_agents:]
        token_pos = token_flat.reshape(params.n_tokens, params.n_token_repeat, 2)
        state = EnvState(agent_positions=agent_pos,
                         token_positions=token_pos,
                         is_alive=jnp.array([True for _ in jnp.arange(params.n_agents)]),
                         time=0)
        obs = self.get_obs(state=state, params=params)
        return obs, state

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
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

            return jnp.where(jnp.logical_or(jnp.logical_not(params.black_death), state.is_alive[i]), b2, base)

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
