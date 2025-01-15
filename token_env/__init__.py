from token_env.token_env import *
from gymnasium.envs.registration import register

register(
    id = "TokenEnv-v0",
    entry_point = "token_env.token_env:TokenEnv",
    kwargs = {"n_tokens": 12, "size": (7, 7), "timeout": 75, "use_fixed_map": True}
)

register(
    id = "TokenEnv-v1",
    entry_point = "token_env.token_env:TokenEnv"
)
