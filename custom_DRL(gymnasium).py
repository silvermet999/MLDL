# ------------------------------------------------------main-----------------------------------------------

import gymnasium as gym
import gym_examples


env = gym.make('gym_examples/GridWorld-v0', size=10)
env.observation_space.shape
wrapped_env = FlattenObservation(env)
wrapped_env.observation_space.shape
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



# ------------------------------------------------------DRL-----------------------------------------------

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import main

states = main.observation
actions = main.action

def GW_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(actions, activation="softmax"))

GW_model = GW_model(states, actions)
GW_model.summary()



# ------------------------------------------------------viz-----------------------------------------------

from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger
import gymnasium as gym

experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="YOUR_PROJECT_NAME",
    workspace="YOUR_WORKSPACE",
)

env = gym.make('Acrobot-v1', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, 'test')
env = CometLogger(env, experiment)

for x in range(20):
    observation, info = env.reset()
    truncated = False
    terminated = False
    while not (truncated or terminated):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        env.render()

env.close() #Uploads video folder 'test' to Comet

