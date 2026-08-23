import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
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
boundingBox = Material([], "sphere", [0,0,0,0,7], [], [], [], [], [], [], [], [], [])

# Define radius of Luneburg Lens:
R = tf.constant(3.0)

# Anisotropic Luneburg Lens material:
a0_1 = R
a1_1 = tf.constant(2.0)
a2_1 = tf.constant(1.0)

b0 = tf.Variable(3.0)
b1 = tf.Variable(2.2)
b2 = tf.Variable(0.9)

c0 = tf.Variable(2.2)
c1 = tf.Variable(-0.8)
c2 = tf.Variable(1.2)

# Air surrounding the lens:
a0_2 = tf.constant(1.0)
a1_2 = tf.constant(1.0)
a2_2 = tf.constant(0.0)

# Objective function:
def objective_func(boundingBox, a0_1, a1_1, a2_1, a0_2, a1_2, a2_2, b0, b1, b2, c0, c1, c2):

    mat1 = Material(False, "sphere", [0,0,0,0,3], a0_1, a1_1, a2_1, b0, b1, b2, c0, c1, c2)
    mat2 = Material(True, "sphere", [0,0,0,3,7], a0_2, a1_2, a2_2, [], [], [], [], [], [])

    # Create list of materials:
    matList = [mat1, mat2]

    # Create a list of completed rays and in progress rays:
    cpRays = []
    ipRays = []

    # Define the incident ray (or multiple rays):
    incident_position = [tf.constant(0.0000001), tf.constant(0.0), tf.constant(-4.999999)]
    pi = [tf.constant(0.0000000001), tf.constant(0.0), tf.constant(1.0)]
    Ei = [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
    incident_material = hp.getMaterialAtCoordinate(matList, incident_position)
    incident_ray = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    #ipRays.append(incident_ray)

    incident_position = [tf.constant(1.0), tf.constant(0.0), tf.constant(-4.999999)]
    incident_ray2 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray2)

    incident_position = [tf.constant(-1.0), tf.constant(0.0), tf.constant(-4.999999)]
    Ei_2 = [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0j, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
    incident_ray3 = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei_2, incident_material, True)
    ipRays.append(incident_ray3)

    # Initialize a boolean variable to check if the ray is in the bounding box:
    inBounds = True

    # Initialize a variable to keep track of the current material:
    currentMat = incident_material

    # Initialize the time step:
    t = tf.constant(0.01) # Arbitrarily chosen for now.

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

    ## Section 3 - Calculation of Objective (or Cost) Function:
    f = of.focusObjective(cpRays, 3, 0)
    return f

learning_rate = 0.01

# Training Loop:
objectives = []
input_1 = []
input_2 = []
input_3 = []
input_4 = []
input_5 = []
input_6 = []

for i in range(20):
    with tf.GradientTape() as tape:
        current_objective = objective_func(boundingBox, a0_1, a1_1, a2_1, a0_2, a1_2, a2_2, b0, b1, b2, c0, c1, c2)
    gradients = tape.gradient(current_objective, [b0, b1, b2, c0, c1, c2])
    # Keep track on inputs variables over time for analysis purposes:
    input_1.append(b0.numpy())
    input_2.append(b1.numpy())
    input_3.append(b2.numpy())
    input_4.append(c0.numpy())
    input_5.append(c1.numpy())
    input_6.append(c2.numpy())
    b0.assign_add(learning_rate * gradients[0])
    b1.assign_add(learning_rate * gradients[1])
    b2.assign_add(learning_rate * gradients[2])
    c0.assign_add(learning_rate * gradients[3])
    c1.assign_add(learning_rate * gradients[3])
    c2.assign_add(learning_rate * gradients[3])

    objectives.append(current_objective)

# Plot the objective function as a function of iteration number:
plt.figure()
plt.plot(objectives)
plt.xlabel("Iteration")
plt.ylabel("Value of Objective Function")
plt.savefig("Test_Optimization.png")

# Plot the input parameters as a function of iteration number:

plt.figure()
plt.plot(input_1)
plt.xlabel("Iteration")
plt.ylabel("b0")
plt.savefig("Updated values of b0")

plt.figure()
plt.plot(input_2)
plt.xlabel("Iteration")
plt.ylabel("b1")
plt.savefig("Updated values of b1")

plt.figure()
plt.plot(input_3)
plt.xlabel("Iteration")
plt.ylabel("b2")
plt.savefig("Updated values of b2")

plt.figure()
plt.plot(input_4)
plt.xlabel("Iteration")
plt.ylabel("c0")
plt.savefig("Updated values of c0")

plt.figure()
plt.plot(input_5)
plt.xlabel("Iteration")
plt.ylabel("c1")
plt.savefig("Updated values of c1")

plt.figure()
plt.plot(input_6)
plt.xlabel("Iteration")
plt.ylabel("c2")
plt.savefig("Updated values of c2")