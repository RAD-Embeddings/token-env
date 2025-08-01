import jax
import token_env
import gymnasium as gym

# import warnings
# warnings.simplefilter('error')
def test(env):
    key = jax.random.PRNGKey(30)
    # Reset env
    key, subkey = jax.random.split(key)
    obs, state = env.reset(key=subkey)
    print("initial obs:", obs)
    print("initial state:", state)
    done = False
    steps = 0


    while not done:
        # Sample actions
        # TODO: Key split???
        keys = jax.random.split(key, env.n_agents + 1)
        key, subkeys = keys[0], keys[1:]
        actions = {agent: env.action_space(agent).sample(subkeys[i]) for i, agent in enumerate(env.agents)}
        print("actions:", actions)
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(actions=actions, state=state, key=subkey)
        print("step:", steps)
        print("obs:", obs)
        print("state:", state)
        print("rewards:", rewards)
        print("dones:", dones)
        input(">>")
        # temp = env.render()
        steps += 1

if __name__ == '__main__':
    test(env=token_env.TokenEnvJax())



    