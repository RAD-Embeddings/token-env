# token-env

This repo implements a single agent Gymnasium environment with tokens.

## Installation

```
pip install token-env
```

## Usage

```
import token_env
import gymnasium as gym

if __name__ == '__main__':
    env = gym.make("TokenEnv-v0")
    obs, info = env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()
```
