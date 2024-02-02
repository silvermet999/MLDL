import gymnasium as gym
import gym_examples


env = gym.make('gym_examples/GridWorld-v0', size=10)
# an action space is the set of all possible actions that an agent can take in a given environment. It defines the range of actions that can be performed.
# setting the seed for the random number generator associated with the action space.
env.action_space.seed(42)

eps = 5
for _ in range(1, eps+1):
    observation, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    # env.action_space.sample(): gen random sample from the action space of the environment.
    # env.step(): called on the environment to take a step in the simulation by providing an action
    # observation: Represents the new state or observation of the environment after taking the specified action.
    # teminated: true or false
    # truncated: true or false
    # info: metadata
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()
    print('Episode:{} Score:{}'.format(_, score))
