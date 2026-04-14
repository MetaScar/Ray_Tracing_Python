import tensorflow as tf
import WavePropagation as wp
import InterfaceAnalysis as ia
import HelperFunctions as hp
import BasisFunctions as bs

# IMPORTANT Note: When defining cylinders in a material list, the smallest cylinder MUST be placed first in the list. Only one other (larger) cylinder is supported.

# The Material Class:
class Material:
    def __init__(self, iso, type, geometry_vector, ordinary_constants, extraordinary_constants, director_constants, ordinary_basis, extraordinary_basis, director_basis):
        # These values are set for all types of geometries
        self.iso = iso
        self.ordinary_constants = ordinary_constants
        self.extraordinary_constants = extraordinary_constants
        self.director_constants = director_constants
        self.ordinary_basis = ordinary_basis
        self.extraordinary_basis = extraordinary_basis
        self.director_basis = director_basis

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

        # Cylinder (centered at the origin)
        if type == "cylinder":
            self.type = "cylinder"
            self.rmin = geometry_vector[0]
            self.rmax = geometry_vector[1]
            self.hmin = geometry_vector[2]
            self.hmax = geometry_vector[3]

    # This function returns the "ordinary" relative permittivity and its associated spatial derivatives.
    # Three basis functions are associated with each material object, corresponding to the spatial distributions of
    # the "ordinary" relative permittivity, "extraordinary" relative permittivity, and the director profile.
    def getOrdinaryPermittivity(self, x, y, z):
        e_perp, deperp_dx, deperp_dy, deperp_dz = bs.permittivityBasis(self.ordinary_basis, self.ordinary_constants, x, y, z)
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    # This function returns the "extraordinary" relative permittivity and its associated spatial derivatives.
    def getExtraordinaryPermittivity(self, x, y, z):
        e_para, depara_dx, depara_dy, depara_dz = bs.permittivityBasis(self.extraordinary_basis, self.extraordinary_constants, x, y, z)
        return e_para, depara_dx, depara_dy, depara_dz

    # This function returns the director (a unit vector), as well as the three spatial derivates for each director component (9 derivates total).
    def getDirector(self, x, y, z):
        director, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z = bs.directorBasis(self.director_basis, self.director_constants, x, y, z)
        return director, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z