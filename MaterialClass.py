import tensorflow as tf
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp

# The Material Class:
class Material:
    def __init__(self, iso, xmin, xmax, ymin, ymax, zmin, zmax, no, ne, c0, c1, c2):
        self.iso = iso
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.no = no
        self.ne = ne
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        