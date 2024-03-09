import numpy as np
def capacity(Hd, G, Hr, theta, Q, N0):
    # Calculation of the capacity
    H = Hd[0] + np.dot(Hr[0], np.dot(np.diag(theta), G))
    H_T = np.array(np.matrix(H).T.conjugate())
    rate = np.real(np.log2(np.linalg.det(
        np.eye(np.size(H, 0))+np.dot(H, np.dot(Q, H_T))/N0)))
    return rate
def water_filling(Power, vec):
    # water filling algorithm
    sortindex = np.argsort(vec)[::-1]
    sortval = vec[sortindex]
    for n in range(len(vec))[::-1]:
        water_level = (sum(sortval[0:n+1]) - Power) / (n+1)
        di = sortval[0:n+1] - water_level
        if all in di > 0:
            break
    out = np.zeros(len(vec))
    out[sortindex[0:len(di)]] = di
    return out
# b = water_filling(20, np.array([50, 10, 1, 1]))