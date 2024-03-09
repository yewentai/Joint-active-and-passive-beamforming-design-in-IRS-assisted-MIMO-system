import pickle
import numpy as np
import scipy.constants as sc
from matplotlib import pyplot as plt
# load data
with open('position.pickle', 'rb') as f:
    position_base, position_user, position_irs = pickle.load(f)
radius_cell = 100  # radius of the cell
# System configuration
frequency = 2e9  # Frequency
wavelength = sc.c / frequency  # Wavelength
wavenumber = 2 * np.pi / wavelength  # Wavenumber
space_antenna_tx = wavelength / 2  # TX antenna space
space_antenna_rx = wavelength / 2  # RX antenna space
space_element_irs = wavelength / 2  # RIS element space
num_antenna_tx = 8  # Number of TX antennas
num_antenna_rx = 2  # Number of RX antennas
num_element_irs_x = 6  # Number of RIS elements in the x-direction
num_element_irs_y = 6  # Number of RIS elements in the y-direction
rician_factor = 1  # Rician factor
num_user = np.size(position_user, 0)  # Number of users
ple_tx_rx = 2.0  # Path loss exponent of TX and RX
# generate the position of the antennas of the bs, users, and the irs
tx_array = np.tile(position_base, [num_antenna_tx, 1])
tx_array[:, 2] = np.linspace(
    position_base[2], position_base[2] + num_antenna_tx * space_antenna_tx, num_antenna_tx)
rx_array = np.tile(position_user, [num_antenna_rx, 1,  1]).reshape(
    num_user, num_antenna_rx, 3)
for i in range(num_user):
    rx_array[i, :, 2] = np.linspace(
        position_user[i, 2], position_user[i, 2] + num_antenna_rx * space_antenna_rx, num_antenna_rx)
irs_array = np.tile(position_irs, [num_element_irs_x, num_element_irs_y, 1])
for i in range(num_element_irs_x):
    irs_array[i, :, 1] = np.linspace(
        position_irs[1], position_irs[1]+num_element_irs_x*space_element_irs, num_element_irs_x)
for i in range(num_element_irs_x):
    irs_array[:, i, 2] = np.linspace(
        position_irs[2], position_irs[2]+num_element_irs_y*space_element_irs, num_element_irs_y)
'''
generate the channel between BS and Users: Hd
'''
# Free Space Path Loss
dist_tx_rx = np.linalg.norm(position_user - position_base, axis=1)
FSPL_dir = (wavelength/(4*np.pi)) ** 2/dist_tx_rx ** ple_tx_rx
# Line of Sight
distance_tx_rx = np.zeros((num_user, num_antenna_rx,  num_antenna_tx))
for i in range(num_user):
    for j in range(num_antenna_rx):
        distance_tx_rx[i, j, :] = np.linalg.norm(
            rx_array[i, j, :] - tx_array, axis=1)
Hd_los = np.exp(-1j*wavenumber*distance_tx_rx)
# Non-Line of Sight
Hd_nlos = np.random.normal(0, 1, size=(
    num_user, num_antenna_rx, num_antenna_tx))
for i in range(num_user):
    Hd_nlos[i, :] = np.sqrt(1/2)*(np.random.normal(0, 1, size=(num_antenna_rx, num_antenna_tx)
                                                   )+1j*np.random.normal(0, 1, size=(num_antenna_rx, num_antenna_tx)))
# Hd
Hd = np.zeros((np.shape(Hd_los)))
for i in range(num_user):
    Hd[i] = np.sqrt(FSPL_dir[i]) * (Hd_los[i] + Hd_nlos[i])
'''
generate the cascaded channel FSPL
'''
# generate the cascaded channel free space path loss
dist_tx_irs = np.linalg.norm(position_irs - position_base)
dist_irs_rx = np.linalg.norm(position_user - position_irs, axis=1)
ple_tx_irs = 2.0
ple_irs_rx = 2.0
FSPL_indir = np.zeros(num_user)
for i in range(num_user):
    FSPL_indir[i] = wavelength**4 / (16*np.pi)**2 / dist_tx_irs**ple_tx_irs / dist_irs_rx[i]**ple_irs_rx * (np.linalg.norm(
        (position_irs-position_base)[0:2])/dist_irs_rx[i] + np.linalg.norm(
        (position_user-position_irs)[i, 0:2])/dist_tx_irs) ** 2
# generate the channel G
distance_tx_irs = np.zeros(
    (num_element_irs_x, num_element_irs_y, num_antenna_tx))
for i in range(num_element_irs_x):
    for j in range(num_element_irs_y):
        distance_tx_irs[i, j, :] = np.linalg.norm(
            irs_array[i, j, :] - tx_array, axis=1)
G_los = np.exp(-1j*wavenumber*distance_tx_irs).reshape(num_element_irs_x *
                                                       num_element_irs_y, num_antenna_tx)
G_nlos = np.random.normal(0, 1, size=(num_element_irs_x * num_element_irs_y, num_antenna_tx)) + \
    1j*np.random.normal(0, 1, size=(num_element_irs_x *
                        num_element_irs_y, num_antenna_tx))
G = np.sqrt(FSPL_indir[0])*np.sqrt(1/(rician_factor + 1)
                                   )*(np.sqrt(rician_factor)*G_los+G_nlos)
distance_irs_rx = np.zeros(
    (num_user, num_antenna_rx, num_element_irs_x, num_element_irs_y))
for i in range(num_user):
    for j in range(num_antenna_rx):
        for k in range(num_element_irs_x):
            for l in range(num_element_irs_y):
                distance_irs_rx[i, j, k, l] = np.linalg.norm(
                    rx_array[i, j, :] - irs_array[k, l, :], axis=0)
Hr_los = np.exp(-1j*wavenumber*distance_irs_rx).reshape(num_user,
                                                        num_antenna_rx, num_element_irs_x*num_element_irs_y)
Hr_nlos = np.random.normal(0, 1, size=(num_user, num_antenna_rx, num_element_irs_x * num_element_irs_y)) + \
    1j*np.random.normal(0, 1, size=(num_user, num_antenna_rx,
                        num_element_irs_x * num_element_irs_y))
Hr = np.sqrt(1/(rician_factor + 1))*(np.sqrt(rician_factor)*Hr_los+Hr_nlos)
# plot
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter3D(tx_array[:, 0], tx_array[:, 1],

              tx_array[:, 2], cmap='Blues')
ax1.scatter3D(rx_array[:, :, 0], rx_array[:, :, 1],
              rx_array[:, :, 2], cmap='Reds')
ax1.scatter3D(irs_array[:, :, 0], irs_array[:, :, 1],
              irs_array[:, :, 2], cmap='Greens')
theta = np.linspace(0, 1, 100) * 2 * np.pi
plt.plot(position_base[0] + radius_cell * np.cos(theta),
         position_base[1] + radius_cell * np.sin(theta),  'gray')
plt.show()
# save data
with open('pathloss.pickle', 'wb') as f:
    pickle.dump([Hd, G, Hr], f)


import scipy.io as scio
scio.savemat('data.mat', {'A':[Hd[0], G, Hr[0]]})