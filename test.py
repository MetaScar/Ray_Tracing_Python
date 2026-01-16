# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import time
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import ObjectiveFunctions as of
from RayClass import Ray
from MaterialClass import Material

# Testing Gaussian beam functions:
theta = tf.constant(0.0)
phi = tf.constant(0.0)
S = [tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)]
Eo = tf.constant(1.0)
sigma = tf.constant(0.005)
E_pol_vector = [tf.constant(0.0), tf.constant(1.0), tf.constant(0.0)]
E_pol_phase = tf.constant(1.0)
xo = tf.constant(0.5)
yo = tf.constant(-0.2)
zo = tf.constant(0.1)
k = tf.constant(209.44)

# Testing new phase tracking and Epol calculations:
rx_init = tf.constant(1.0)
ry_init = tf.constant(1.0)
rz_init = tf.constant(1.0)
pox_init = tf.constant(1.0)
poy_init = tf.constant(1.0)
poz_init = tf.constant(1.0)

