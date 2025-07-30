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
    print("Initial State:", state)
    done = False
    steps = 0


    while not done:
        # Sample actions
        # TODO: Key split???
        key, subkey = jax.random.split(key)
        action = env.action_space(params=env.default_params).sample(key=subkey)
        # print({agent: env.unwrapped.action_parser[action[agent]] for agent in action})
        print(action)
        key, subkey = jax.random.split(key)
        obs, state, rewards, done, info = env.step(action=action, state=state, key=subkey)
        print("Step:", steps)
        print(obs)
        print(state)
        print(rewards)
        print(done)
        input(">>")
        # temp = env.render()
        steps += 1

    env.close()

if __name__ == '__main__':
    test(env=token_env.TokenEnvJax())



    