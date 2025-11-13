import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp

# Time step:
h = tf.constant(0.015)

# Position Vector:
rx_init = tf.constant(1.0)
ry_init = tf.constant(1.0)
rz_init = tf.constant(1.0)

# Wave vector:
px_init = tf.constant(1.0)
py_init = tf.constant(0.0)
pz_init = tf.constant(0.0)

# Indices of refraction:
no = tf.constant(1.0)
ne = tf.constant(1.5)

# Constants for director profile:
c0 = tf.constant(0.5)
c1 = tf.constant(1.0)
c2 = tf.constant(0.1)

E = [tf.constant(0.0, dtype=tf.complex64), tf.constant(1.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]

rx, ry, rz, pox, poy, poz, E = wp.ExtraordinaryWavePropagation(rx_init, ry_init, rz_init, px_init, py_init, pz_init, no, ne, c0, c1, c2, h, E)

print(rx.numpy())
print(ry.numpy())
print(rz.numpy())
print(E.numpy())




