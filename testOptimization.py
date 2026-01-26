import tensorflow as tf
import time
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import ObjectiveFunctions as of
from RayClass import Ray
from MaterialClass import Material

# This is the main algorithm for the ray-tracing code.

## Section 1 - Define System Parameters and Initial Ray ##

# Define a rectangular prism bounding box:
boundingBox = Material([], "rect", [-0.20, 0.20, -0.20, 0.20, 0.0, 0.50], [], [], [], [], [], [], [], [], [])

# Define system materials:

# Air in front of (and behind) the lens:
a0_1 = tf.constant(1.0)
a1_1 = tf.constant(0.0)
a2_1 = tf.constant(0.0)

# GRIN Lens material:
a0_2 = tf.constant(16.0)
a1_2 = tf.constant(0.0)
a2_2 = tf.constant(5.19337)

b0_2 = tf.constant(2.25)
b1_2 = tf.constant(0.0)
b2_2 = tf.Variable(4.4262)

c0_2 = tf.constant(0.0)
c1_2 = tf.constant(0.0)
c2_2 = tf.constant(0.0)

# Objective function:
def objective_func(boundingBox, a0_1, a1_1, a2_1, a0_2, a1_2, a2_2, b0_2, b1_2, b2_2, c0_2, c1_2, c2_2):

    mat1 = Material(True, "rect", [-0.20, 0.20, -0.05, 0.05, 0.0, 0.05], a0_1, a1_1, a2_1, [], [], [], [], [], [])
    mat2 = Material(False, "rect", [-0.20, 0.20, -0.05, 0.05, 0.05, 0.35], a0_2, a1_2, a2_2, b0_2, b1_2, b2_2, c0_2, c1_2, c2_2)
    mat3 = Material(True, "rect", [-0.20, 0.20, -0.05, 0.05, 0.35, 0.50], a0_1, a1_1, a2_1, [], [], [], [], [], [])

    # Create list of materials:
    matList = [mat1, mat2, mat3]

    # Create a list of completed rays and in progress rays:
    cpRays = []
    ipRays = []

    # Define the incident rays:
    # Y-polarized rays:
    incident_position = [tf.constant(0.01), tf.constant(0.0), tf.constant(0.0000001)]
    pi = [tf.constant(0.000000001), tf.constant(0.0), tf.constant(1.0)]
    Ei = [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
    Ei_xpol = [tf.complex(tf.constant(0.0), tf.constant(1.0)), tf.constant(0.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
    incident_material = hp.getMaterialAtCoordinate(matList, incident_position)
    incident_ray = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray)

    incident_position = [tf.constant(0.02), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray2 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray2)

    incident_position = [tf.constant(0.03), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray3)

    incident_position = [tf.constant(-0.03), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray3)

    incident_position = [tf.constant(-0.02), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray3)

    incident_position = [tf.constant(-0.01), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray3)

    # X-polarized rays:
    incident_position = [tf.constant(0.01), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
    ipRays.append(incident_ray3)

    incident_position = [tf.constant(0.02), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
    ipRays.append(incident_ray3)

    incident_position = [tf.constant(0.03), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
    ipRays.append(incident_ray3)

    incident_position = [tf.constant(-0.01), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
    #ipRays.append(incident_ray3)

    incident_position = [tf.constant(-0.02), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
    #ipRays.append(incident_ray3)

    incident_position = [tf.constant(-0.03), tf.constant(0.0), tf.constant(0.0000001)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_xpol, incident_material, True)
    #ipRays.append(incident_ray3)

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

    ## Section 3 - Calculation of Objective (or Cost) Function:
    f = of.focusObjective(cpRays, 0.45, 0.0, b0_2, b2_2)
    return f

# Start timer:
start_time = time.perf_counter()

# Define optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# Training Loop:
objectives = []
input_1 = []
input_2 = []
input_3 = []
input_4 = []
input_5 = []

for i in range(30):
    with tf.GradientTape() as tape:
        current_objective = objective_func(boundingBox, a0_1, a1_1, a2_1, a0_2, a1_2, a2_2, b0_2, b1_2, b2_2, c0_2, c1_2, c2_2)
    gradients = tape.gradient(current_objective, [b2_2])
    # Keep track on inputs variables over time for analysis purposes:
    input_1.append(b2_2.numpy())

    optimizer.apply_gradients(zip(gradients, [b2_2]))

    objectives.append(current_objective)

# End timer:
end_time = time.perf_counter()

# Calculate and print runtime:
duration = end_time - start_time
print(f"Execution took {duration:.4f} seconds")

# Plot the objective function as a function of iteration number:
plt.figure()
plt.plot(objectives)
plt.xlabel("Iteration")
plt.ylabel("Value of Objective Function")
plt.show()
plt.savefig("Test Optimization")

# Plot the input parameters as a function of iteration number:
plt.figure()
plt.plot(input_1)
plt.xlabel("Iteration")
plt.ylabel("C1")
plt.show()

print(input_1[-1])

