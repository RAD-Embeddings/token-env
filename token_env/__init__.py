from token_env.token_env import *
from token_env.utils import *
from gymnasium.envs.registration import register

register(
    id = "TokenEnv-v0",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_agents": 1, "n_tokens": 12, "size": (7, 7), "timeout": 75}
)

register(
    id = "TokenEnv-fixed-v0",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_agents": 1, "n_tokens": 12, "size": (7, 7), "timeout": 75, "use_fixed_map": True}
)

register(
    id = "TokenEnv-v1",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_agents": 1}
)

register(
    id = "TokenEnv-fixed-v1",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_agents": 1, "use_fixed_map": True}
)

register(
    id = "TokenEnv-2-agents-v1",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_agents": 2}
)

register(
    id = "TokenEnv-2-agents-fixed-v1",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_agents": 2, "use_fixed_map": True}
)
