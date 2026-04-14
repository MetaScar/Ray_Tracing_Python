import tensorflow as tf
import HelperFunctions as hp
from RayClass import Ray
from MaterialClass import Material

# This function is the core logic/algorithm of the ray tracer.
# Inputs: A bounding box for the problem (a material object), a list of material objects, a list of intial rays (Rays objects), a fixed distance step, 
# and a tolerance for the bisection algorithm (determines the distance step near boundaries).
# Outputs: A list of rays objects representing all completed rays in the system.

def RayTrace(boundingBox, matList, initialRays, distance_step, tol_bisection):

    # Step 1: Initialization of variables.

    # Create a list of completed rays and in progress rays:
    cpRays = []
    ipRays = []

    # Populate the "in progress rays" array with the initial rays:
    for element in initialRays:
        ipRays.append(element)

    # Initialize a boolean variable to check if the ray is in the bounding box:
    inBounds = True

    # Initialize the incident material:
    incident_position = [ipRays[-1].rx[0], ipRays[-1].ry[0], ipRays[-1].rz[0]]
    incident_material = hp.getMaterialAtCoordinate(matList, incident_position)

    # Initialize a variable to keep track of the current material:
    currentMat = incident_material

    # Step 2: Looping through and tracing all rays.
    while(ipRays):
        while(True):
            # Perform a single step of propagation:
            ipRays[-1].propagation_step(distance_step)

            # Check if ray is out of bounds:
            inBounds = hp.checkBoundary(boundingBox, [ipRays[-1].rx[-1], ipRays[-1].ry[-1], ipRays[-1].rz[-1]])
            if not(inBounds):
                cpRays.append(ipRays.pop())
                break

            # Check if a new material has been reached:
            currentMat = hp.getMaterialAtCoordinate(matList, [ipRays[-1].rx[-1], ipRays[-1].ry[-1], ipRays[-1].rz[-1]])
            if not(ipRays[-1].Mat == currentMat):
                ipRays[-1].bisection(distance_step, tol_bisection, matList)
                finishedRay = ipRays[-1]
                cpRays.append(ipRays.pop())
                newRays = finishedRay.initialize_new_rays(currentMat)
                for ray in newRays:
                    ipRays.append(ray)

    # Step 3: Return the list of completed rays.
    return cpRays

