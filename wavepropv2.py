import tensorflow as tf
import matplotlib.pyplot as plt
import HelperFunctions as hp

# THIS IS AN ATTEMPT TO FIX POTENTIAL ERRORS WITH THE PREVIOUS VERSION OF THE WAVE PROPAGATION FUNCTION
# This function takes ONE STEP in the first-order Runge-Kutta to calculate the next value of the ray path r(t) and wave normal p(t) for an ordinary wave.
def OrdinaryWavePropagation(rx_init, ry_init, rz_init, pox_init, poy_init, poz_init, mat, distance_step, E): 

    ko = 209.4395102 # For f = 10 GHz

    # Calculate the derivatives of the index of refraction:
    e_perp, deperp_dx, deperp_dy, deperp_dz = hp.getOrdinaryIndex(rx_init, ry_init, rz_init, mat.a0, mat.a1, mat.a2)
    
    # single step of the first-order Runge-Kutta method:
    k1 = deperp_dx
    k2 = deperp_dy
    k3 = deperp_dz
    k4 = 2*pox_init
    k5 = 2*poy_init
    k6 = 2*poz_init

    # Scale k4, k5, and k6 such that the ray travels a fixed distance:
    current_step = tf.math.sqrt(k4**2 + k5**2 + k6**2)
    h = distance_step/current_step

    pox = pox_init + k1*h
    poy = poy_init + k2*h
    poz = poz_init + k3*h
    rx = rx_init + k4*h
    ry = ry_init + k5*h
    rz = rz_init + k6*h

    # Calculate previous phase from previous E-field vector:
    if abs(E[0])>0.5:
        prev_phase = tf.math.angle(E[0])
    elif abs(E[1])>0.5:
        prev_phase = tf.math.angle(E[1])
    else:
        prev_phase = tf.math.angle(E[2])

    # Calculate phase progression and total phase:
    delta_phi = ko*(pox_init*k4*h + poy_init*k5*h + poz_init*k6*h)
    Ephase = -1.0*prev_phase + delta_phi

    if(mat.iso==True):
        Epol = [abs(x) for x in E] 
    else:
        # Calculate director (optical axis) as an intermediate step for calculating the electric polarization vector:
        director, _, _, _, _, _, _, _, _, _ = hp.getDirector(rx_init, ry_init, rz_init, mat.c0, mat.c1, mat.c2)
        # Calculate electric polarization vector:
        Epol = hp.getEfield([], [], pox, poy, poz, director, True)
    
    # Account for phase progression:
    Epol = tf.cast(Epol, dtype=tf.complex64)*tf.math.exp(tf.complex(0.0, -1.0*Ephase))

    return rx, ry, rz, pox, poy, poz, Epol

# This function takes ONE STEP in first-order Runge-Kutta to calculate the next value of the ray path r(t) and wave normal p(t) for an extraordinary wave.
# Note that currently only explicit mathematical formulas for director profiles are supported.
def ExtraordinaryWavePropagation(rx_init, ry_init, rz_init, pex_init, pey_init, pez_init, mat, distance_step, E):

    ko = 209.4395102 # For f = 10 GHz

    # Calculates e_perp, e_paralell, and the associated spatial derivatives:
    e_perp, deperp_dx, deperp_dy, deperp_dz = hp.getOrdinaryIndex(rx_init, ry_init, rz_init, mat.a0, mat.a1, mat.a2)
    e_para, depara_dx, depara_dy, depara_dz = hp.getExtraordinaryIndex(rx_init, ry_init, rz_init, mat.b0, mat.b1, mat.b2)
    
    # Calculate the 'local' director (optical axis) and its derivatives:
    d_local, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z = hp.getDirector(rx_init, ry_init, rz_init, mat.c0, mat.c1, mat.c2)

    # Calculate the derivatives (k's) for each differential equation:
    k1 = -2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*(pex_init*ddx_x + pey_init*ddy_x + pez_init*ddz_x) + e_perp*depara_dx + (e_para - (tf.norm([pex_init, pey_init, pez_init])**2))*deperp_dx - (depara_dx - deperp_dx)*(tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1))**2
    k2 = -2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*(pex_init*ddx_y + pey_init*ddy_y + pez_init*ddz_y) + e_perp*depara_dy + (e_para - (tf.norm([pex_init, pey_init, pez_init])**2))*deperp_dy - (depara_dy - deperp_dy)*(tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1))**2
    k3 = -2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*(pex_init*ddx_z + pey_init*ddy_z + pez_init*ddz_z) + e_perp*depara_dz + (e_para - (tf.norm([pex_init, pey_init, pez_init])**2))*deperp_dz - (depara_dz - deperp_dz)*(tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1))**2
    k4 = 2*e_perp*pex_init + 2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*d_local[0]
    k5 = 2*e_perp*pey_init + 2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*d_local[1]
    k6 = 2*e_perp*pez_init + 2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*d_local[2]

    # Scale k4, k5, and k6 such that the ray travels a fixed distance:
    current_step = tf.math.sqrt(k4**2 + k5**2 + k6**2)
    h = distance_step/current_step

    # Time-step using the first-order Runge-Kutta method:
    pex = pex_init + k1*h
    pey = pey_init + k2*h
    pez = pez_init + k3*h
    rx = rx_init + k4*h
    ry = ry_init + k5*h
    rz = rz_init + k6*h

    # Calculate previous phase from previous E-field vector:
    if abs(E[0])>0.5:
        prev_phase = tf.math.angle(E[0])
    elif abs(E[1])>0.5:
        prev_phase = tf.math.angle(E[1])
    else:
        prev_phase = tf.math.angle(E[2])

    # Calculate phase progression and total phase:
    delta_phi = ko*(pex_init*k4*h + pey_init*k5*h + pez_init*k6*h)
    Ephase = prev_phase + delta_phi

    # Calculate director (optical axis) as an intermediate step for calculating the electric polarization vector:
    director, _, _, _, _, _, _, _, _, _ = hp.getDirector(rx_init, ry_init, rz_init, mat.c0, mat.c1, mat.c2)
    # Calculate electric polarization vector:
    Epol = hp.getEfield(tf.math.sqrt(e_perp), tf.math.sqrt(e_para), pex, pey, pez, director, False)
    # Account for phase progression:
    Epol = tf.cast(Epol, dtype=tf.complex64)*tf.math.exp(tf.complex(0.0, -1.0*Ephase))

    return rx, ry, rz, pex, pey, pez, Epol