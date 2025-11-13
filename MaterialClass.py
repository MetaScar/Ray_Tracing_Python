import tensorflow as tf
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp

# The Material Class:
class Material:
    def __init__(self, iso, xmin, xmax, ymin, ymax, zmin, zmax, a0, a1, a2, b0, b1, b2, c0, c1, c2):
        self.iso = iso
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        