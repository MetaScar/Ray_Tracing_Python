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

# Define system materials:

# Luneburg Lens material:
a0_1 = R
a1_1 = tf.Variable(2.1)
a2_1 = tf.Variable(0.9)

# Air surrounding the lens:
a0_2 = tf.constant(1.0)
a1_2 = tf.constant(1.0)
a2_2 = tf.constant(0.0)

# Objective function:
def objective_func(boundingBox, a0_1, a1_1, a2_1, a0_2, a1_2, a2_2):

    mat1 = Material(True, "sphere", [0,0,0,0,3] ,a0_1, a1_1, a2_1, [], [], [], [], [], [])
    mat2 = Material(True, "sphere", [0,0,0,3,7], a0_2, a1_2, a2_2, [], [], [], [], [], [])

    # Create list of materials:
    matList = [mat1, mat2]

    # Create a list of completed rays and in progress rays:
    cpRays = []
    ipRays = []

    # Define the incident ray (or multiple rays):
    incident_position = [tf.constant(1.5), tf.constant(0.0), tf.constant(-5.0)]
    pi = [tf.constant(0.00000000001), tf.constant(0.0), tf.constant(1.0)]
    Ei = [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
    incident_material = hp.getMaterialAtCoordinate(matList, incident_position)
    incident_ray = Ray(incident_position[0], incident_position[1], incident_position[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray)

    incident_position2 = [tf.constant(1.0), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray2 = Ray(incident_position2[0], incident_position2[1], incident_position2[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    # ipRays.append(incident_ray2)

    incident_position3 = [tf.constant(0.5), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray3 = Ray(incident_position3[0], incident_position3[1], incident_position3[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray3)

    incident_position3 = [tf.constant(0.0000000000001), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray4 = Ray(incident_position3[0], incident_position3[1], incident_position3[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    # ipRays.append(incident_ray4)

    incident_position5 = [tf.constant(-0.5), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(-1.0), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    # ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(-1.5), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(-2.0), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
   # ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(-2.5), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(1.5), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(2.0), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
   # ipRays.append(incident_ray5)

    incident_position5 = [tf.constant(2.5), tf.constant(0.0), tf.constant(-5.0)]
    incident_ray5 = Ray(incident_position5[0], incident_position5[1], incident_position5[2], pi[0], pi[1], pi[2], 1.0, Ei, incident_material, True)
    ipRays.append(incident_ray5)


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

    ## Section 3 - Calculation of Objective (or Cost) Function:
    f = of.focusObjective(cpRays, 3, 0)
    return f

learning_rate = 0.01

# Training Loop:
objectives = []
input_1 = []
input_2 = []
for i in range(5):
    with tf.GradientTape() as tape:
        current_objective = objective_func(boundingBox, a0_1, a1_1, a2_1, a0_2, a1_2, a2_2)
    gradients = tape.gradient(current_objective, [a1_1, a2_1])
    # Keep track on inputs variables over time for analysis purposes:
    input_1.append(a0_2.numpy())
    input_2.append(a2_2.numpy())
    a1_1.assign_add(learning_rate * gradients[0])
    a2_1.assign_add(learning_rate * gradients[1])

    objectives.append(current_objective)

# Plot the objective function as a function of iteration number:
plt.plot(objectives)
plt.xlabel("Iteration")
plt.ylabel("Value of Objective Function")
plt.savefig("Test_Optimization.png")



'''
## Section 3 - Plotting ##
for i in range(len(cpRays)):
    if not(cpRays[i].ordinary):
        plt.plot(cpRays[i].rz, cpRays[i].rx, color='red')
    else:
        plt.plot(cpRays[i].rz, cpRays[i].rx, color='blue')

# Plot vertical lines to indicate boundaries between different media:
plt.axvline(x=2, color='black', linestyle='--')
# plt.axvline(x=5, color='red', linestyle='--')

plt.title("2D Ray Propagation")
plt.xlabel("Z-Axis")
plt.ylabel("X-Axis")
plt.xlim(boundingBox.zmin, boundingBox.zmax)
plt.ylim(boundingBox.xmin, boundingBox.xmax)

plt.savefig("Test_plot.png")

'''