import tensorflow as tf
import matplotlib.pyplot as plt
import HelperFunctions as hp

# This function takes ONE STEP in the first-order Runge-Kutta to calculate the next value of the ray path r(t) and wave normal p(t) for an ordinary wave.
def OrdinaryWavePropagation(rx_init, ry_init, rz_init, pox_init, poy_init, poz_init, time_step, E):

    h = time_step; # time step
    ko = 209.4395102 # For f = 10 GHz
    
    # single step of the first-order Runge-Kutta method:
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 2*pox_init
    k5 = 2*poy_init
    k6 = 2*poz_init
    pox = pox_init + k1*h
    poy = poy_init + k2*h
    poz = poz_init + k3*h
    rx = rx_init + k4*h
    ry = ry_init + k5*h
    rz = rz_init + k6*h

    # Calculate phase progression:
    delta_phi = ko*(pox_init*k4*h + poy_init*k5*h + poz_init*k6*h)
    E = E*tf.math.exp(tf.complex(0.0, -1.0*delta_phi))

    return rx, ry, rz, pox, poy, poz, E

# This function takes ONE STEP in first-order Runge-Kutta to calculate the next value of the ray path r(t) and wave normal p(t) for an extraordinary wave.
# Note that currently only explicit mathematical formulas for director profiles are supported.
def ExtraordinaryWavePropagation(rx_init, ry_init, rz_init, pex_init, pey_init, pez_init, no, ne, c0, c1, c2, time_step, E):

    ko = 209.4395102 # For f = 10 GHz

    # Calculate er's from n's:
    e_para = no ** 2
    e_perp = ne ** 2

    h = time_step # time step

    # Calculate the 'local' director (optical axis) and its derivatives:
    d_local, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z = hp.getDirector(rx_init, ry_init, rz_init, c0, c1, c2)

    # Calculate the derivatives (k's) for each differential equation:
    k1 = -2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*(pex_init*ddx_x + pey_init*ddy_x + pez_init*ddz_x)
    k2 = -2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*(pex_init*ddx_y + pey_init*ddy_y + pez_init*ddz_y)
    k3 = -2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*(pex_init*ddx_z + pey_init*ddy_z + pez_init*ddz_z)
    k4 = 2*e_perp*pex_init + 2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*d_local[0]
    k5 = 2*e_perp*pey_init + 2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*d_local[1]
    k6 = 2*e_perp*pez_init + 2*(e_para - e_perp)*tf.tensordot([pex_init, pey_init, pez_init], d_local, axes=1)*d_local[2]

    # Time-step using the first-order Runge-Kutta method:
    pex = pex_init + k1*h
    pey = pey_init + k2*h
    pez = pez_init + k3*h
    rx = rx_init + k4*h
    ry = ry_init + k5*h
    rz = rz_init + k6*h

    # Calculate phase progression:
    delta_phi = ko*(pex_init*k4*h + pey_init*k5*h + pez_init*k6*h)
    E = E*tf.math.exp(tf.complex(0.0, -1.0*delta_phi))

    return rx, ry, rz, pex, pey, pez, E
    
   