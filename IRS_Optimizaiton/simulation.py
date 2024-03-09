import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import *
import scipy.io as scio
# load data
with open('pathloss.pickle', 'rb') as f:
    Hd, G, Hr = pickle.load(f)
num_rx = np.size(Hd, 1)
num_tx = np.size(Hd, 2)
num_ris = np.size(G, 0)
Power = 1  # Transmit power in Watts
N0_dB = -80.0              # Noise power in dB
N0 = 10 ** (N0_dB/10)      # SNR
# calculate the capacity without direct link
# Q = np.eye(num_tx) * Power/num_tx
# theta = np.linspace(0, 2*np.pi, num_ris)
# Hd_ = np.zeros((np.shape(Hd[0])))
# C_withoutdir = capacity(Hd_, G, Hr, theta, Q, N0)
# print(C_withoutdir)
# # calculate the capacity without IRS
# rate_withoutIRS = []
# Q = np.eye(num_tx) * Power/num_tx
# Hd_T = np.array(np.matrix(Hd[0]).T.conjugate())
# for i in range(501):
#     rate_withoutIRS.append(np.real(np.log2(np.linalg.det(
#         np.eye(num_rx)+np.dot(Hd[0], np.dot(Q, Hd_T))/N0))))
# # calculate the capacity with random IRS shifts
# rate_ranshift = []
# for i in range(501):
#     theta = np.random.uniform(0, 2*np.pi, num_ris)
#     Q = np.eye(num_tx) * Power/num_tx
#     rate_ranshift.append(capacity(Hd, G, Hr, theta, Q, N0))
# # calculate the capacity with random beamforming vector
# rate_rancov = []
# for i in range(501):
#     theta = np.linspace(0, 2*np.pi, num_ris)
#     Q = np.diag(np.random.dirichlet(np.ones(num_tx))) * Power
#     rate_rancov.append(capacity(Hd, G, Hr, theta, Q, N0))
data5 = scio.loadmat('B5.mat')
plt.semilogx(b5, label='Nt=5')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Achievable Rate(bit/s)')
plt.show()