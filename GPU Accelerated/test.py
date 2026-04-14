# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import NewHelpers as nh
# import NewInterfaceAnalysis as nia
# import NewAlgorithm as na
import time
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import ObjectiveFunctions as of
from MaterialClass import Material
import RayTracing as rt


# # --------------------------------------------------------------------------------------------------------------- #

# Testing variance function in tensorflow:
wvG1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 16.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0], [25.0, 26.0, 27.0], [28.0, 29.0, 30.0]]
variance = tf.math.reduce_variance(wvG1, axis=0)
pass

# ### Testing out quasi-uniform sampling over a sphere:

# # Inputs:
# N = tf.constant(100.0) # Number of rays
# theta_max_deg = tf.constant(5.0) # Cone angle in degrees
# theta_target_deg = tf.constant(30.0) # Target angle in degrees

# positions, wave_vectors, PoyntingMag, alive, Efields, ordinary, material_IDs, ray_ordinary_consts, ray_extraordinary_consts, ray_director_consts = nh.createIsotropicRays(N, theta_max_deg, theta_target_deg, [0.0, 0.0, 0.0], [.57735, .57735, .57735], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], 3)

# ### Plotting/Visualization:

# dirs_np = wave_vectors.numpy()

# x = dirs_np[:, 0]
# y = dirs_np[:, 1]
# z = dirs_np[:, 2]

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection='3d')

# ax.scatter(x, y, z, s=5)

# # Make axes equal:
# ax.set_box_aspect([1, 1, 1])

# ax.set_xlim(-1, 1)
# ax.set_ylim(-1, 1)
# ax.set_zlim(-1, 1)

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# plt.title("Fibonacci Spherical Cap Sampling")
# plt.show()






