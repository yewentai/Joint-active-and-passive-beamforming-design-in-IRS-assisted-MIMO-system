import numpy as np
import matplotlib.pyplot as plt
import pickle
# initial parameters
# position of base: abscissa, ordinate, height
positon_base = np.array([100., 100., 30.])
radius_cell = 100  # radius of the cell
num_user = 5  # number of users
positon_user = np.zeros((num_user, 3))  # location of the users
# randomly generate the location of the users
for i in range(num_user):
    radius = radius_cell * (0.8 + np.random.rand(1, 1) * 0.2)
    theta = np.random.rand(1, 1) * 2 * np.pi / 10
    p_x = positon_base[0] + radius * np.cos(theta)
    p_y = positon_base[1] + radius * np.sin(theta)
    p_z = 0
    positon_user[i, :] = np.array([p_x, p_y, p_z])
# deploy the IRS
position_irs = np.mean(positon_user, axis=0)
# radius = radius_cell * (0.1 + np.random.rand(1, 1) * 0.1)
# theta = np.random.rand(1, 1) * 2 * np.pi / 10
# p_x = positon_base[0] + radius * np.cos(theta)
# p_y = positon_base[1] + radius * np.sin(theta)
# p_z = 10.
# position_irs = np.array([p_x, p_y, p_z])
# save data
with open('position.pickle', 'wb') as f:
    pickle.dump([positon_base, positon_user, position_irs], f)
# plot
plt.scatter(positon_base[0], positon_base[1], c='r')
plt.scatter(positon_user[:, 0], positon_user[:, 1], c='b')
plt.scatter(position_irs[0], position_irs[1], c='g')
plt.xlim(positon_base[0] - radius_cell, positon_base[0] + radius_cell)
plt.ylim(positon_base[1] - radius_cell, positon_base[1] + radius_cell)
theta = np.linspace(0, 1, 100) * 2 * np.pi
plt.plot(positon_base[0] + radius_cell * np.cos(theta),
         positon_base[1] + radius_cell * np.sin(theta))
plt.show()