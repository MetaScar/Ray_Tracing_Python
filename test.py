import tensorflow as tf
import matplotlib.pyplot as plt
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import ObjectiveFunctions as of
from RayClass import Ray
from MaterialClass import Material


# Code to test drawing circles in Matplotlib:

theta = tf.linspace(0.0, 2.0*3.14159, 100)
x = 3*tf.math.cos(theta)
y = 3*tf.math.sin(theta)

plt.figure(figsize=(7,7))
plt.plot(x, y, color='black')

plt.savefig("Circle.png")

