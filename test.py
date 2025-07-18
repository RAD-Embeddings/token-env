import token_env
import gymnasium as gym
import supersuit as ss

def test(env_id):
    # Initialize environment
    env = gym.make(env_id)
    # env = token_env.TokenEnv(n_agents=1, n_tokens=3, size=(5, 5), use_fixed_map=True)
    # Reset env
    obs = env.reset()
    env.render()
    done = False
    steps = 0

    while not done:
        steps += 1
        # Sample actions
        action = env.action_space.sample()
        print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        # input(">>")

        done = ((all(terminated.values()) if isinstance(terminated, dict) else terminated)
             or (all(truncated.values()) if isinstance(truncated, dict) else truncated))

    print(f"Test completed in {steps} steps in {env_id}.")
    env.close()

if __name__ == '__main__':
    # test(env_id="TokenEnv-v1")
    # test(env_id="TokenEnv-fixed-v1")
    test(env_id="TokenEnv-2-agents-v1")
    # test(env_id="TokenEnv-2-agents-fixed-v1")

    