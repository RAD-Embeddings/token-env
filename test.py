import token_env
import gymnasium as gym

def test(env):
    # Initialize environment
    if isinstance(env, str):
        env = gym.make(env)
    # Reset env
    obs = env.reset()
    env.render()
    done = False
    steps = 0

    while not done:
        # Sample actions
        action = env.action_space.sample()
        print({agent: env.unwrapped.action_parser[action[agent]] for agent in action})
        obs, reward, terminated, truncated, info = env.step(action)
        temp = env.render()
        print(steps, reward, terminated, truncated, info)

        done = ((all(terminated.values()) if isinstance(terminated, dict) else terminated)
             or (all(truncated.values()) if isinstance(truncated, dict) else truncated))
        steps += 1

    env.close()

if __name__ == '__main__':
    test(env="TokenEnv-v1")
    test(env="TokenEnv-fixed-v1")
    test(env="TokenEnv-2-agents-v1")
    test(env="TokenEnv-2-agents-fixed-v1")
    test(env=token_env.TokenEnv(n_agents=3, n_tokens=3, size=(5, 5), use_fixed_map=True))
    print("Tests completed.")



    