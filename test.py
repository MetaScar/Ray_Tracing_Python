# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import ObjectiveFunctions as of
from RayClass import Ray
from MaterialClass import Material
import RayTracing as rt

# Define a rectangular prism bounding box:
boundingBox = Material([], "rect", [-0.20, 0.20, -0.20, 0.20, 0.0, 0.50], [], [], [], [], [], [], [], [], [])

# Air in front of the lens:
a0_1 = tf.constant(1.0)
a1_1 = tf.constant(0.0)
a2_1 = tf.constant(0.0)

# GRIN Lens material:
a0_2 = tf.constant(16.0)
a1_2 = tf.constant(0.0)
a2_2 = tf.constant(5.19337)

b0_2 = tf.constant(2.25)
b1_2 = tf.constant(0.0)
b2_2 = tf.constant(4.44926)

c0_2 = tf.constant(0.0)
c1_2 = tf.constant(0.0)
c2_2 = tf.constant(0.0)

mat1 = Material(True, "rect", [-0.20, 0.20, -0.05, 0.05, 0.0, 0.05], a0_1, a1_1, a2_1, [], [], [], [], [], [])
mat2 = Material(False, "rect", [-0.20, 0.20, -0.05, 0.05, 0.05, 0.35], a0_2, a1_2, a2_2, b0_2, b1_2, b2_2, c0_2, c1_2, c2_2)
mat3 = Material(True, "rect", [-0.20, 0.20, -0.05, 0.05, 0.35, 0.5], a0_1, a1_1, a2_1, [], [], [], [], [], [])

# Create list of materials:
matList = [mat1, mat2, mat3]

# Create a list of initial rays to be launched into the system:
# Define the incident rays:
ipRays = []

# Incident angle:
theta = tf.constant(15.0)
theta = 0.0174532925*theta

# Calculate incident positions such that all rays reach the lens at the same point, irrespective of incidence angle:
delta_x = tf.constant(0.05)*tf.math.tan(theta)
pos1 = tf.constant(-0.03) - delta_x
pos2 = tf.constant(-0.02) - delta_x
pos3 = tf.constant(-0.01) - delta_x
pos4 = tf.constant(0.01) - delta_x
pos5 = tf.constant(0.02) - delta_x
pos6 = tf.constant(0.03) - delta_x

z_pos = tf.constant(0.0000000000001)


# Y-polarized rays:
incident_position = [tf.constant(pos1), tf.constant(0.0), tf.constant(z_pos)]
pi = [tf.math.sin(theta), tf.constant(0.0), tf.math.cos(theta)]
Ei = [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
Ei_xpol = [tf.constant(-0.866, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64), tf.constant(0.5, dtype=tf.complex64)]
incident_material = hp.getMaterialAtCoordinate(matList, incident_position)
incident_ray = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
ipRays.append(incident_ray)

incident_position = [tf.constant(pos2), tf.constant(0.0), tf.constant(z_pos)]
incident_ray2 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
ipRays.append(incident_ray2)

incident_position = [tf.constant(pos3), tf.constant(0.0), tf.constant(z_pos)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos4), tf.constant(0.0), tf.constant(z_pos)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos5), tf.constant(0.0), tf.constant(z_pos)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos6), tf.constant(0.0), tf.constant(z_pos)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
ipRays.append(incident_ray3)

# X-polarized rays:
incident_position = [tf.constant(pos1), tf.constant(0.0), tf.constant(0.0000001)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos2), tf.constant(0.0), tf.constant(0.0000001)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos3), tf.constant(0.0), tf.constant(0.0000001)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos4), tf.constant(0.0), tf.constant(0.0000001)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos5), tf.constant(0.0), tf.constant(0.0000001)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
ipRays.append(incident_ray3)

incident_position = [tf.constant(pos6), tf.constant(0.0), tf.constant(0.0000001)]
incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
ipRays.append(incident_ray3)

initialRays = ipRays

# Initialize the DISTANCE step and tolerance for bisection algorithm:
d = tf.constant(0.005)
tol = tf.constant(0.001)

############ Call the ray-tracing algorithm: ###############
cpRays = rt.RayTrace(boundingBox, matList, initialRays, d, tol)



################ Plotting: ##############################

plt.figure(1, figsize=(6,6))

for i in range(len(cpRays)):
    if not(cpRays[i].ordinary):
        plt.plot(cpRays[i].rz, cpRays[i].rx, color='red')
    else:
        plt.plot(cpRays[i].rz, cpRays[i].rx, color='blue')



# Plot a vertical line to indicate the start of the lens:
plt.axvline(x=0.05, color='k', linestyle='-', linewidth=2)

# Plot a vertical line to indicate the end of the lens:
plt.axvline(x=0.35, color='k', linestyle='-', linewidth=2)

plt.title("Dual Focal Point Lens")
plt.xlabel("Z-Axis")
plt.ylabel("X-Axis")
plt.xlim(0.0, 0.5)
plt.ylim(-0.05, 0.05)
plt.grid(True)

plt.show()