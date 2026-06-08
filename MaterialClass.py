import tensorflow as tf
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp

# The Material Class:
class Material:
    def __init__(self, iso, type, geometry_vector, a0, a1, a2, b0, b1, b2, c0, c1, c2):
        # These values are set for all types of geometries
        self.iso = iso
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

        # Rectangular Prism
        if type == "rect":
            self.type = "rect"
            self.xmin = geometry_vector[0]
            self.xmax = geometry_vector[1]
            self.ymin = geometry_vector[2]
            self.ymax = geometry_vector[3]
            self.zmin = geometry_vector[4]
            self.zmax = geometry_vector[5]
        
        # Sphere
        if type == "sphere":
            self.type = "sphere"
            self.centerPoint = [geometry_vector[0], geometry_vector[1], geometry_vector[2]]
            self.rmin = geometry_vector[3]
            self.rmax = geometry_vector[4]