import tensorflow as tf


# This file contains all the basis functions for both relative permittivities and director profiles.
# It is split up into two seperate functions, one for permittivities and one for basis functions.

def permittivityBasis(basis, consts, x, y, z):
    match basis:
        case "x_4th_degree_polynomial":
            e = consts[0] + consts[1]*x + consts[2]*x**2 + consts[3]*x**3 + consts[4]*x**4
            de_dx = consts[1] + 2.0*consts[2]*x + 3.0*consts[3]*x**2 + 4.0*consts[4]*x**3
            de_dy = tf.constant(0.0)
            de_dz = tf.constant(0.0)
            return e, de_dx, de_dy, de_dz

def directorBasis(basis, consts, x, y, z):
    match basis:
        case "constant_x":
            dx = tf.constant(1.0)
            dy = tf.constant(0.0)
            dz = tf.constant(0.0)
            director = tf.stack([dx, dy, dz])
            
            ddx_x = tf.constant(0.0)
            ddx_y = tf.constant(0.0)
            ddx_z = tf.constant(0.0)
            ddy_x = tf.constant(0.0)
            ddy_y = tf.constant(0.0)
            ddy_z = tf.constant(0.0)
            ddz_x = tf.constant(0.0)
            ddz_y = tf.constant(0.0)
            ddz_z = tf.constant(0.0)

            return director, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z