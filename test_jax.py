import jax
import token_env
import gymnasium as gym

# import warnings
# warnings.simplefilter('error')
def test(env):
    key = jax.random.PRNGKey(30)
    # Reset env
    obs, state = env.reset(key=key)
    print("Initial State:", state)
    done = False
    steps = 0


    while not done:
        # Sample actions
        # Key split???
        action = env.action_space(params=env.default_params).sample(key=key)
        # print({agent: env.unwrapped.action_parser[action[agent]] for agent in action})
        print(action)
        temp = env.step(action=action, state=state, key=key)
        print(temp[1])
        print(temp[2])
        input(">>")
        temp = env.render()
        print(steps, reward, terminated, truncated, info)

        done = ((all(terminated.values()) if isinstance(terminated, dict) else terminated)
             or (all(truncated.values()) if isinstance(truncated, dict) else truncated))
        steps += 1

    env.close()

if __name__ == '__main__':
    test(env=token_env.TokenEnvJax())



    