import token_env
import gymnasium as gym

def test(env_id: str):
    env = gym.make(env_id)
    obs, info = env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

if __name__ == '__main__':
    test("TokenEnv-v0")
    test("TokenEnv-v1")
