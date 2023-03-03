import gc
import matplotlib.pyplot as plt
import numpy as np
import CarRacingDQN as DQN
import gym
import requests

max_steps_per_episode = 64
threshold = 5
env = gym.make("CarRacing-v2", continuous = False)
token = '5619479679:AAE2zq66Opudre_W0DhCeDFup_N7BszXIQg'
chat_id = '563209854'
template = "https://api.telegram.org/bot" + token + "/sendMessage" + "?chat_id=" + chat_id + "&text="
update_after_actions = 8
agent = DQN.DQNAgent((96, 96, 3), 5, 5048, batchsize=32)
episode_count = 0
reward_hist = []
for _ in range(1024):  # Run until solved
    episode_count +=1
    frame_count = 0
    state = env.reset()[0]
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):
        frame_count += 1
        action = agent.act(state)
        state_next, reward, done, _, _ = env.step(action)
        if action == 3:
            reward += .25

        episode_reward += reward
        agent.add_to_memory(state, action, state_next, reward, done)

        if frame_count % update_after_actions == 0:
            agent.train()
    print(f'episode {episode_count} done with episode reward {episode_reward:.2f}')
    if episode_reward > threshold:
        threshold *= 3 / 2
        max_steps_per_episode *= 2
        print(f'new_threshold: {threshold}, new maximum steps per episode: {max_steps_per_episode}')
    if episode_count % 100 == 0:
        agent.save(name=f'models/model_{episode_count}.h5')
        agent.update_target_model()
        requests.get(
            template + f'episode {episode_count} done with episode reward {episode_reward:.2f} ')
    reward_hist.append(episode_reward)
    gc.collect()
fig, ax = plt.subplots()
ax.plot(np.linspace(1, len(reward_hist), len(reward_hist)), reward_hist)
ax.set_xlabel('epoch')
ax.set_ylabel('reward')
plt.show()
