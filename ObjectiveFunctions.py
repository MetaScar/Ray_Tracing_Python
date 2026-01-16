import tensorflow as tf
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp

# This objective function returns the total power density passing through a specified 2D rectangular area.
# This is a way of measuring how much of a focus there is at that point.
# The dot product only considered the power density of rays with a specified linear polarization (in this case y-pol)
def focusObjective(cpRays, z, x, C1, C2):
    total_S = 0 # variable to keep track of total power density, scaled by the distance to the focal point and correct polarization.

    for i in range(len(cpRays)):
        if cpRays[i].Mat.zmax == 0.50:
            d2, E_pol = getClosestPosition(cpRays[i], z, x)
            # distance_factor = tf.cast(cpRays[i].PoyntingMag*tf.math.exp(-100.0*d2), dtype=tf.complex64)
            distance_factor = tf.cast(1.0*tf.math.exp(-10000.0*d2), dtype=tf.complex64)
            #if(cpRays[i].ordinary == True):
                #polarization_factor = tf.tensordot(E_pol, [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)], axes=1)
            #else:
                #polarization_factor = tf.tensordot(E_pol, [tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)], axes=1)

            # penalty if permittivity falls to less than one:
            val = (C1 - .0025*C2 - 1)
            penalty = 1/(1+tf.math.exp(-10.0*val - 5.0))
            penalty = tf.cast(penalty, dtype=tf.complex64)
            total_S = total_S + distance_factor*penalty

    return -1.0*abs(total_S)

# Given a ray, this helper function finds the ray position that is closest to the specified focal point.
# It then returns the distance to that focal point squared, as well as the electric field polarization at that point.
def getClosestPosition(ray, zf, xf):
    min_d2 = 100
    for i in range(len(ray.rz)):
        d2 = (ray.rz[i] - zf)**2 + (ray.rx[i] - xf)**2
        if d2 < min_d2:
            min_d2 = d2
            E_pol = ray.Efield[i]

    return min_d2, E_pol

# Given a ray, this helper function determines whether or not the ray passes through a specified 2D rectangular area.
def passThroughFocal(ray, z, x, res):
    flag = False # Boolean to keep track of whether the ray passes through the point.
    for i in range(len(ray.rx)):
        if (ray.rx[i] > x - res) and (ray.rx[i] < x + res) and (ray.rz[i] > z - res) and (ray.rz[i] < z + res):
            flag = True

    return flag    
