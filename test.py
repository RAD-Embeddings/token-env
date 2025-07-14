import token_env
import gymnasium as gym
import supersuit as ss
from pettingzoo.test import parallel_api_test

def test(env_id):
    # Initialize environment
    env = gym.make(env_id)
    # env = token_env.TokenEnv(n_agents=1, n_tokens=3, size=(5, 5), use_fixed_map=True)
    # Reset env
    obs = env.reset()
    done = False
    steps = 0

    while not done:
        steps += 1
        # Sample actions
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = ((all(terminated.values()) if isinstance(terminated, dict) else terminated)
             or (all(truncated.values()) if isinstance(truncated, dict) else truncated))

    print(f"Test completed in {steps} steps in {env_id}.")
    env.close()

if __name__ == '__main__':
    test(env_id="TokenEnv-v1")
    test(env_id="TokenEnv-fixed-v1")
    test(env_id="TokenEnv-2-agents-v1")
    test(env_id="TokenEnv-2-agents-fixed-v1")

    env = token_env.TokenEnv(
        n_agents=2,
        n_tokens=5,
        n_token_repeat=2,
        size=(7,7),
        timeout=100,
        use_fixed_map=True,
        slip_prob=(0.0, 0.0)
    )
    env = token_env.gym2zoo(env)

    parallel_api_test(env)

    env = ss.black_death_v3(env)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    # env = VecMonitor(env)


    # Sanity test: run one episode with random actions
    obs, info = env.reset()

    done = False

    step = 0
    while not done:
        actions = {a: gym.spaces.Discrete(4).sample() for a in info}
        # actions = env.action_space.sample()
        obs, rewards, terms, truncs, infos = env.step(actions)
        print(f"\nStep {step}")
        for a in rewards:
            print(f" {a}: reward={rewards[a]}, done={terms[a] or truncs[a]}")
        done = all(terms.values())
        step += 1

    env.close()
    print("\nEpisode finished.")
