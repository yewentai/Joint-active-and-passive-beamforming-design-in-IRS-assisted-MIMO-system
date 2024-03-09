import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import *
from agent import *
import cmath
def R2P(x):
    return abs(x), np.angle(x)
def Phi(angles):
    angles_ = []
    for i in range(len(angles)):
        angles_.append(cmath.exp(1j * angles[i]))
    shifts = np.diag(angles_)
    return shifts
def H_sum(hr, shifts, g, hd):
    reflect = np.matmul(np.matmul(hr.getH(), shifts), g)
    direct = hd
    sum = reflect + direct
    return sum
def SNR(hr, shifts, g, hd, Pmax, sigma):
    reflect = np.matmul(np.matmul(hr.getH(), shifts), g)
    direct = hd
    sum = reflect + direct
    w = np.sqrt(Pmax) * sum.getH() / np.sqrt(np.matmul(sum, sum.getH()).item())
    snr = (1 / sigma) * np.abs(np.matmul(sum, w)) ** 2
    return snr.item()
# load data
with open('pathloss.pickle', 'rb') as f:
    Hd, G, Hr, FSPL_indir = pickle.load(f)
num_rx = np.size(Hd, 1)
num_tx = np.size(Hd, 2)
num_ris = np.size(G, 0)
Power = 1  # Transmit power in Watts
N0_dB = -120.0              # Noise power in dB
N0 = 10 ** (N0_dB/10)      # SNR
hd = R2P(Hd)
hr = R2P(Hr)
g = R2P(G)
h = np.concatenate((np.array(hr).reshape((2, -1)), np.array(
    hd).reshape((2, -1)), np.array(g).reshape((2, -1))), axis=1)
print(np.shape(h))
num_cell = 1
num_base = 1
num_irs = 1
num_user = 1
num_antenna_base = 10
num_element_irs = 5*10
num_antenna_user = 1
num_slot = 1000
Pmax_base_w = 5
dist_base_irs_m = 48
dist_base_user_m = 50
dist_irs_user_m = 5
sigma_linear = 8e-11
num_actions = num_element_irs
num_states = num_element_irs + 1
batch_size = 300
rewards = []
avg_rewards = []
agent = DDPGagent(num_actions, num_states, actor_learning_rate=1e-4,
                  critic_learning_rate=1e-3, disc_fact=0.95, tau=0.005, max_memory_size=50000)
noise = OUNoise(num_actions)
noise.reset()
'''
state:shifts
action:shifts
'''
for episode in range(100):
    snr_ = []
    for _ in range(100):
        angles = 2*np.pi*np.random.rand(num_actions)
        shifts = Phi(angles)
        snr_.append(SNR(hr, shifts, g, hd,  Pmax_base_w, sigma_linear))
    if episode % 10 == 0:
        print(episode)
    # generate initail phase-shift on irs randomly
    state = 2*np.pi*np.random.rand(num_actions)
    shifts = Phi(state)
    snr = SNR(hr, shifts, g, hd,  Pmax_base_w, sigma_linear)
    state = np.concatenate((state, snr), axis=None).reshape(1, num_states)
    episode_reward = 0
    for step in range(num_slot):
        action = agent.get_action(state)
        action = noise.get_action(action)
        # noise = np.random.normal(loc=0, scale=np.sqrt(10), size=(1, num_actions))
        # action = action + noise
        shifts = Phi(action.squeeze())
        snr = SNR(hr, shifts, g, hd,  Pmax_base_w, sigma_linear)
        next_state = np.concatenate(
            (action, snr), axis=None).reshape(1, num_states)
        reward = next_state[:, -1].item() - np.mean(snr)
        agent.memory.push(state, action, reward, next_state)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = next_state
        episode_reward += reward
    rewards.append(episode_reward/num_slot)
    avg_rewards.append(np.mean(rewards[-10:]))
'''
shate:h
action:shifts
'''
for episode in range(10):
    if episode % 1 == 0:
        print(episode)
    # generate initail phase-shift on irs randomly
    angles = 2*np.pi*np.random.rand(num_actions)
    shifts = Phi(angles)
    snr = SNR(hr, shifts, g, hd,  Pmax_base_w, sigma_linear)
    episode_reward = 0
    for step in range(num_slot):
        state = h.reshape(1, -1)
        # state = np.concatenate((state, snr), axis=None).reshape(1, num_states)
        action = agent.get_action(state)
        action = noise.get_action(action)
        # noise = np.random.normal(loc=0, scale=np.sqrt(10), size=(1, num_actions))
        # action = action + noise
        shifts = Phi(action.squeeze())
        snr = SNR(hr, shifts, g, hd,  Pmax_base_w, sigma_linear)
        next_state = state
        # next_state = np.concatenate((h.reshape(1,-1), snr), axis=None).reshape(1, num_states)
        reward = next_state[:, -1].item()
        agent.memory.push(state, action, reward, next_state)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = next_state
        episode_reward += reward
    rewards.append(episode_reward/num_slot)
    avg_rewards.append(np.mean(rewards[-10:]))
plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
plt.show()