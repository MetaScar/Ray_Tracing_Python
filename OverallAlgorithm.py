import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
from RayClass import Ray
from MaterialClass import Material

# This is the main algorithm for the ray-tracing code.

## Section 1 - Define System Parameters and Initial Ray ##

# Define a rectangular prism bounding box:
boundingBox = Material([], 0, 7, 0, 5, 0, 5, [], [], [], [], [])

# Define system materials:
no1 = tf.constant(1.0)
no2 = tf.constant(1.2)
ne2 = tf.constant(1.9)
c0 = tf.constant(.7854)
c1 = tf.constant(1.0)
c2 = tf.constant(0.0)
no3 = tf.constant(1.0)

with tf.GradientTape(persistent=True) as tape:

    tape.watch(c0)
    tape.watch(c1)

    mat1 = Material(True, 0, 7, 0, 5, 0, 2, no1, [], [], [], [])
    mat2 = Material(False, 0, 7, 0, 5, 2, 3, no2, ne2, c0, c1, c2)
    mat3 = Material(True, 0, 7, 0, 5, 3, 5, no3, [], [], [], [])

    # Create list of materials:
    matList = [mat1, mat2, mat3]

    # Create a list of completed rays and in progress rays:
    cpRays = []
    ipRays = []

    # Define the incident ray (or multiple rays):
    incident_position = [tf.constant(0.5), tf.constant(2.5), tf.constant(0.0)]
    pi = [tf.constant(.7071), tf.constant(0.0), tf.constant(.7071)]
    Ei = [tf.constant(-0.57735, dtype=tf.complex64), tf.constant(0.57735, dtype=tf.complex64), tf.constant(0.57735, dtype=tf.complex64)]
    incident_material = hp.getMaterialAtCoordinate(matList, incident_position)
    incident_ray = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray)

    # Initialize a boolean variable to check if the ray is in the bounding box:
    inBounds = True

    # Initialize a variable to keep track of the current material:
    currentMat = incident_material

    # Initialize the time step:
    t = tf.constant(0.001) # Arbitrarily chosen for now.

    ## Section 2 - Ray-Tracing Algorithm!! ##
    while(ipRays):
        while(True):
            # Perform a single step of propagation:
            ipRays[-1].propagation_step(t)

            # Check if ray is out of bounds:
            inBounds = hp.checkBoundary(boundingBox, [ipRays[-1].rx[-1], ipRays[-1].ry[-1], ipRays[-1].rz[-1]])
            if not(inBounds):
                cpRays.append(ipRays.pop())
                break

            # Check if a new material has been reached:
            currentMat = hp.getMaterialAtCoordinate(matList, [ipRays[-1].rx[-1], ipRays[-1].ry[-1], ipRays[-1].rz[-1]])
            if not(ipRays[-1].Mat == currentMat):
                finishedRay = ipRays[-1]
                cpRays.append(ipRays.pop())
                newRays = finishedRay.initialize_new_rays(currentMat)
                for ray in newRays:
                    ipRays.append(ray)

#gradient1 = tape.gradient(cpRays[14].rx[-1], c0)
#gradient2 = tape.gradient(cpRays[14].rx[-1], c1)

# print(gradient1.numpy())   

## Section 3 - Plotting ##
for i in range(len(cpRays)):
    if not(cpRays[i].ordinary):
        plt.plot(cpRays[i].rz, cpRays[i].rx, color='red')
    else:
        plt.plot(cpRays[i].rz, cpRays[i].rx, color='blue')

plt.title("2D Ray Propagation")
plt.xlabel("Z-Axis")
plt.ylabel("X-Axis")
plt.xlim(boundingBox.zmin, boundingBox.zmax)
plt.ylim(boundingBox.xmin, boundingBox.xmax)

plt.savefig("Test_plot.png")
