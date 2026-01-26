import tensorflow as tf
import time
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import ObjectiveFunctions as of
from RayClass import Ray
from MaterialClass import Material
import json

# Start timer:
start_time = time.perf_counter()

# This is the main algorithm for the ray-tracing code.

## Section 1 - Define System Parameters and Initial Ray ##

# Define a rectangular prism bounding box:
boundingBox = Material([], "rect", [-0.20, 0.20, -0.20, 0.20, 0.0, 0.50], [], [], [], [], [], [], [], [], [])

# Define system materials:

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

# Luneburg lens:
# a0 = tf.constant(2.0)
# a2 = tf.constant(0.0)

# Code that GradientTape needs to watch when optimizing:
with tf.GradientTape() as tape:

    mat1 = Material(True, "rect", [-0.20, 0.20, -0.05, 0.05, 0.0, 0.05], a0_1, a1_1, a2_1, [], [], [], [], [], [])
    mat2 = Material(False, "rect", [-0.20, 0.20, -0.05, 0.05, 0.05, 0.35], a0_2, a1_2, a2_2, b0_2, b1_2, b2_2, c0_2, c1_2, c2_2)
    mat3 = Material(True, "rect", [-0.20, 0.20, -0.05, 0.05, 0.35, 0.5], a0_1, a1_1, a2_1, [], [], [], [], [], [])

    #mat1 = Material(True, "sphere", [0.0, 0.0, 0.0, 0.0, 4.0], a0, 0.0, a2, [], [], [], [], [], [])
    #mat2 = Material(True, "sphere", [0.0, 0.0, 0.0, 4.0, 7.0], 1.0, 0.0, 0.0, [], [], [], [], [], [])

    # Create list of materials:
    matList = [mat1, mat2, mat3]

    # Create a list of completed rays and in progress rays:
    cpRays = []
    ipRays = []

    # Define the incident rays:

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

    # Initialize a boolean variable to check if the ray is in the bounding box:
    inBounds = True

    # Initialize a variable to keep track of the current material:
    currentMat = incident_material

    # Initialize the DISTANCE step and tolerance for bisection algorithm:
    d = tf.constant(0.005)
    tol = tf.constant(0.001)

    ## Section 2 - Ray-Tracing Algorithm!! ##
    while(ipRays):
        while(True):
            # Perform a single step of propagation:
            ipRays[-1].propagation_step(d)

            # Check if ray is out of bounds:
            inBounds = hp.checkBoundary(boundingBox, [ipRays[-1].rx[-1], ipRays[-1].ry[-1], ipRays[-1].rz[-1]])
            if not(inBounds):
                cpRays.append(ipRays.pop())
                break

            # Check if a new material has been reached:
            currentMat = hp.getMaterialAtCoordinate(matList, [ipRays[-1].rx[-1], ipRays[-1].ry[-1], ipRays[-1].rz[-1]])
            if not(ipRays[-1].Mat == currentMat):
                ipRays[-1].bisection(d, tol, matList)
                finishedRay = ipRays[-1]
                cpRays.append(ipRays.pop())
                newRays = finishedRay.initialize_new_rays(currentMat)
                for ray in newRays:
                    ipRays.append(ray)

    # Section 3 - Calculation of Objective (or Cost) Function:
    #f = of.focusObjective(cpRays, 0.45, 0.0, b0_2, b2_2)
    #print("Value of objective function:")
    #print(f.numpy())

# test_grad1 = tape.gradient(f, a0_2)
# test_grad2 = tape.gradient(f, a2_2)
# print(test_grad1.numpy())
# print(test_grad2.numpy())

# print(gradient1.numpy()) 

# End timer:
end_time = time.perf_counter()

# Calculate and print runtime:
duration = end_time - start_time
print(f"Execution took {duration:.4f} seconds")


## Section 3 - Plotting ##
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

# Plot dashed vertical lines to indicate the desired focal points:
#plt.axvline(x=0.30, color='g', linestyle='dashed', linewidth=2)
#plt.axvline(x=0.45, color='g', linestyle='dashed', linewidth=2)


#Plot a circle of radius 4 to indicate where the Luneburg Lens is:
#theta = tf.linspace(0.0, 2.0*3.14159, 100)
#x = 4.0*tf.math.cos(theta)
#y = 4.0*tf.math.sin(theta)

#plt.plot(x, y, color='black')

plt.title("2D Ray Propagation")
plt.xlabel("Z-Axis")
plt.ylabel("X-Axis")
plt.xlim(0.0, 0.5)
plt.ylim(-0.05, 0.05)
plt.grid(True)

plt.show()


# Saving z-position and x-position of rays to a JSON file:

filtered_data = []
for ray in cpRays:
    ray_dictionary = {
        "x": tf.convert_to_tensor(ray.rx).numpy().tolist(),
        "z": tf.convert_to_tensor(ray.rz).numpy().tolist()
    }
    filtered_data.append(ray_dictionary)

with open("rays.json", "w") as f:
    json.dump(filtered_data, f, indent=4)
