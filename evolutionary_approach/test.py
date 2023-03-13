import numpy as np
import gym

best_individual = np.load('population.npy', allow_pickle=True)
env = gym.make("CarRacing-v2", continuous=False, render_mode ='human')
env.reset(seed=27)
print(best_individual)
for step in best_individual:
    env.step(step)
