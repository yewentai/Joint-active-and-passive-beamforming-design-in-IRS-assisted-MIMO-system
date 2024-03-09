import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import *
# load data
with open('pathloss.pickle', 'rb') as f:
    Hd, G, Hr = pickle.load(f)
def grad_Q(Hd, G, Hr, theta, Q):
    # Calculation of gradient w.r.t. Q
    H = Hd[0] + np.dot(Hr[0], np.dot(np.diag(theta), G))
    H_T = np.array(np.matrix(H).T.conjugate())
    grad_Q = H_T.dot(np.linalg.inv(
        (np.eye(num_rx)+H.dot(Q.dot(H_T))))).dot(H)
    return grad_Q
def grad_theta(Hd, G, Hr, theta, Q):
    # Calculation of gradient w.r.t. theta
    H = Hd[0] + np.dot(Hr[0], np.dot(np.diag(theta), G))
    H_T = np.array(np.matrix(H).T.conjugate())
    Hr_T = np.array(np.matrix(Hr[0]).T.conjugate())
    G_T = np.array(np.matrix(G).T.conjugate())
    grad_theta = np.diag(Hr_T.dot(np.linalg.inv(
        np.eye(num_rx)+H.dot(Q.dot(H_T))).dot(H).dot(Q).dot(G_T)))
    return grad_theta
def proj_Q(Power, Q):
    # Projection of Q onto the feasible set
    [D, U] = np.linalg.eig(Q)
    D_ = water_filling(Power, np.real(D))
    U_T = np.array(np.matrix(U).T.conjugate())
    Q_ = U_T.dot(np.diag(D_)).dot(U)
    return Q_
def proj_theta(theta):
    # Projection of theta onto the feasible set
    return theta/np.abs(theta)
# initiate
num_rx = np.size(Hd, 1)
num_tx = np.size(Hd, 2)
num_ris = np.size(G, 0)
Power = 1  # Transmit power in Watts
N0_dB = -120.0              # Noise power in dB
N0 = 10 ** (N0_dB/10)       # Noise level
SNR = Power/N0  # SNR
theta = np.random.uniform(0, 2*np.pi, num_ris)
Q = np.eye(num_tx) * Power/num_tx
# C1 = capacity(Hd, G, Hr, FSPL_indir, theta, Q, N0)
# Scaling
c = np.sqrt(np.linalg.norm(
    Hd[0])/np.linalg.norm(np.dot(Hr[0], G))) * max(np.sqrt(Power), 1)/np.sqrt(Power)*10
Hd = Hd*np.sqrt(SNR)/c
G = G*np.sqrt(SNR)
Q = Q*c**2
theta = theta/c
# C2 = capacity(Hd, G, Hr, FSPL_indir, theta, Q, N0)
# Line-search
delta = 1e-5
rho = 0.5
iIter = 0
maxIter = 500
stepsize = 100
Cout = capacity(Hd, G, Hr, theta, Q, N0)
Cprev = Cout
out = [Cout]
while iIter < maxIter:
    iIter += 1
    Q_grad = grad_Q(Hd, G, Hr, theta, Q)
    theta_grad = grad_theta(Hd, G, Hr, theta, Q)
    for iLineSearch in range(30):
        Q_new = Q + stepsize*Q_grad
        Q_new = proj_Q(Power*c**2, Q_new)
        theta_new = theta + stepsize*theta_grad
        theta_new = proj_theta(theta_new)/c
        Cnew = capacity(Hd, G, Hr, theta_new, Q_new, N0)
        if Cnew - Cprev >= delta * np.linalg.norm(Q_new - Q)**2 + np.linalg.norm(theta_new - theta)**2 or stepsize < 1e-4:
            theta = theta_new
            Q = Q_new
            Cprev = Cnew
            break
        else:
            stepsize = rho * stepsize
    out.append(Cnew)  
plt.plot(out)