# git clone https://github.com/Farama-Foundation/gym-examples
# cd gym-examples
# python -m venv .env
# source .env/bin/activate
# pip install -e .

import gymnasium as gym
import gym_examples


env = gym.make('gym_examples/GridWorld-v0', size=10)
# an action space is the set of all possible actions that an agent can take in a given environment. It defines the range of actions that can be performed.
# setting the seed for the random number generator associated with the action space.
env.action_space.seed(42)
observation, info = env.reset(seed=42)

for _ in range(100):
    # env.action_space.sample(): gen random sample from the action space of the environment.
    # env.step(): called on the environment to take a step in the simulation by providing an action
    # observation: Represents the new state or observation of the environment after taking the specified action.
    # teminated: true or false
    # truncated: true or false
    # info: metadata
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
