import tensorflow as tf
import WavePropagation as wp
import wavepropv2 as wpv2
import InterfaceAnalysis as ia
import HelperFunctions as hp

# The Ray class:
class Ray:
    # Ray initialization function:
    def __init__(self, rx, ry, rz, px, py, pz, PoyntingMag, Efield, startingMat, ordinary):
        self.rx = []
        self.ry = []
        self.rz = []
        self.px = []
        self.py = []
        self.pz = []
        self.Efield = []
        self.rx.append(rx)
        self.ry.append(ry)
        self.rz.append(rz)
        self.px.append(px)
        self.py.append(py)
        self.pz.append(pz)
        self.Efield.append(Efield)
        self.PoyntingMag = PoyntingMag
        self.Mat = startingMat
        self.ordinary = ordinary

    # Function for single time step of ray propagation:
    def propagation_step(self, time_step):
        if self.ordinary:
            rx, ry, rz, pox, poy, poz, E = wp.OrdinaryWavePropagation(self.rx[-1], self.ry[-1], self.rz[-1], self.px[-1], self.py[-1], self.pz[-1], self.Mat, time_step, self.Efield[-1])
            self.rx.append(rx)
            self.ry.append(ry)
            self.rz.append(rz)
            self.px.append(pox)
            self.py.append(poy)
            self.pz.append(poz)
            self.Efield.append(E)
        else:
            rx, ry, rz, pex, pey, pez, E = wp.ExtraordinaryWavePropagation(self.rx[-1], self.ry[-1], self.rz[-1], self.px[-1], self.py[-1], self.pz[-1], self.Mat, time_step, self.Efield[-1])
            self.rx.append(rx)
            self.ry.append(ry)
            self.rz.append(rz)
            self.px.append(pex)
            self.py.append(pey)
            self.pz.append(pez)
            self.Efield.append(E)
    
    # Bisection algorithm - an algorithm which automatically decreses the distance step size of rays near a boundary.
    def bisection(self, d, tol, matList):
        while(True):
            currentMat = hp.getMaterialAtCoordinate(matList, [self.rx[-1], self.ry[-1], self.rz[-1]])
            if (self.Mat == currentMat):
                self.propagation_step(d)
            else:
                if(d <= tol):
                    break
                else: 
                    # Undo propagation step:
                    self.rx.pop()
                    self.ry.pop()
                    self.rz.pop()
                    self.px.pop()
                    self.py.pop()
                    self.pz.pop()
                    self.Efield.pop()
                    # Halve the distance step and perform another time step:
                    d = d/2.0
                    self.propagation_step(d)

    # This function returns a list of Rays which wil be subsequently appended to the total list of rays in the main function.
    def initialize_new_rays(self, currentMat):
        rays = []
        n = hp.getSurfaceNormal(currentMat, self.Mat, [self.rx[-1], self.ry[-1], self.rz[-1]])
        # Isotropic-Isotropic Interface:
        if self.Mat.iso and currentMat.iso:
            # Find the indices of refraction at the boundary:
            e_perp, _, _, _ = self.Mat.getOrdinaryPermittivity(self.rx[-2], self.ry[-2], self.rz[-2])
            no1 = tf.math.sqrt(e_perp)
            e_perp, _, _, _ = currentMat.getOrdinaryPermittivity(self.rx[-1], self.ry[-1], self.rz[-1])
            no2 = tf.math.sqrt(e_perp)

            p_r, p_t, E_r, E_t, S_r, S_t = ia.Isotropic_Isotropic(n, no1, no2, [self.px[-2], self.py[-2], self.pz[-2]], self.Efield[-2])
            # Initialize reflected ray:
            # Check for negligible magnitude and/or evanescent wave:
            check = hp.validRay(S_r, p_r)
            if check:
                p_r = tf.math.real(p_r)
                ray1 = Ray(self.rx[-2], self.ry[-2], self.rz[-2], p_r[0], p_r[1], p_r[2], S_r, E_r, self.Mat, True)
                #rays.append(ray1)
            # Initialize transmitted ray:
            check = hp.validRay(S_t, p_t)
            if check:
                p_t = tf.math.real(p_t)
                ray2 = Ray(self.rx[-1], self.ry[-1], self.rz[-1], p_t[0], p_t[1], p_t[2], S_t, E_t, currentMat, True)
                rays.append(ray2)
            return rays
        # Isotropic-Anisotropic Interface:
        if self.Mat.iso and not(currentMat.iso):
            # Find the indices of refraction and optical axis at the boundary:
            e_perp, _, _, _ = self.Mat.getOrdinaryPermittivity(self.rx[-2], self.ry[-2], self.rz[-2])
            no1 = tf.math.sqrt(e_perp)
            e_perp, _, _, _ = currentMat.getOrdinaryPermittivity(self.rx[-1], self.ry[-1], self.rz[-1])
            no2 = tf.math.sqrt(e_perp)
            e_para, _, _, _ = currentMat.getExtraordinaryPermittivity(self.rx[-1], self.ry[-1], self.rz[-1])
            ne2 = tf.math.sqrt(e_para)
            o2, _, _, _, _, _, _, _, _, _ = currentMat.getDirector(self.rx[-1], self.ry[-1], self.rz[-1])

            p_r, p_to, p_te, E_r, E_to, E_te, S_r, S_to, S_te = ia.Isotropic_Anisotropic(n, o2, no1, no2, ne2, [self.px[-2], self.py[-2], self.pz[-2]], self.Efield[-2])
            # Initialize the reflected ray:
            check = hp.validRay(S_r, p_r)
            if check:
                p_r = tf.math.real(p_r)
                ray1 = Ray(self.rx[-2], self.ry[-2], self.rz[-2], p_r[0], p_r[1], p_r[2], S_r, E_r, self.Mat, True)
                #rays.append(ray1)
            # Initialize the transmitted ordinary ray:
            check = hp.validRay(S_to, p_to)
            if check:
                p_to = tf.math.real(p_to)
                ray2 = Ray(self.rx[-1], self.ry[-1], self.rz[-1], p_to[0], p_to[1], p_to[2], S_to, E_to, currentMat, True)
                rays.append(ray2)
            # Initialize the transmitted extraordinary ray:
            check = hp.validRay(S_te, p_te)
            if check:
                p_te = tf.math.real(p_te)
                ray3 = Ray(self.rx[-1], self.ry[-1], self.rz[-1], p_te[0], p_te[1], p_te[2], S_te, E_te, currentMat, False)
                rays.append(ray3)
            return rays
        # Anisotropic-Isotropic Interface:
        if not(self.Mat.iso) and currentMat.iso:
            # Find the indices of refraction and optical axis at the boundary:
            e_perp, _, _, _ = self.Mat.getOrdinaryPermittivity(self.rx[-2], self.ry[-2], self.rz[-2])
            no1 = tf.math.sqrt(e_perp)
            e_para, _, _, _ = self.Mat.getExtraordinaryPermittivity(self.rx[-2], self.ry[-2], self.rz[-2])
            ne1 = tf.math.sqrt(e_para)
            e_perp, _, _, _ = currentMat.getOrdinaryPermittivity(self.rx[-1], self.ry[-1], self.rz[-1])
            no2 = tf.math.sqrt(e_perp)
            o1, _, _, _, _, _, _, _, _, _ = self.Mat.getDirector(self.rx[-2], self.ry[-2], self.rz[-2])
            
            p_ro, p_re, p_t, E_ro, E_re, E_t, S_ro, S_re, S_t = ia.Anisotropic_Isotropic(n, o1, no1, ne1, no2, [self.px[-2], self.py[-2], self.pz[-2]], self.Efield[-2])
            # Initialize the reflected ordinary ray:
            check = hp.validRay(S_ro, p_ro)
            if check:
                p_ro = tf.math.real(p_ro)
                ray1 = Ray(self.rx[-2], self.ry[-2], self.rz[-2], p_ro[0], p_ro[1], p_ro[2], S_ro, E_ro, self.Mat, True)
                #rays.append(ray1)
            # Initialize the reflected extraordinary ray:
            check = hp.validRay(S_re, p_re)
            if check:
                p_re = tf.math.real(p_re)
                ray2 = Ray(self.rx[-2], self.ry[-2], self.rz[-2], p_re[0], p_re[1], p_re[2], S_re, E_re, self.Mat, False)
                #rays.append(ray2)
            # Initialize the transmitted ray:
            check = hp.validRay(S_t, p_t)
            if check:
                p_t = tf.math.real(p_t)
                ray3 = Ray(self.rx[-1], self.ry[-1], self.rz[-1], p_t[0], p_t[1], p_t[2], S_t, E_t, currentMat, True)
                rays.append(ray3)
            return rays
        # Anisotropic-Anisotropic Interface:
        if not(self.Mat.iso) and not(currentMat.iso):
            # Find the indices of refraction and optical axis at the boundary:
            e_perp, _, _, _ = self.Mat.getOrdinaryPermittivity(self.rx[-2], self.ry[-2], self.rz[-2])
            no1 = tf.math.sqrt(e_perp)
            e_para, _, _, _ = self.Mat.getExtraordinaryPermittivity(self.rx[-2], self.ry[-2], self.rz[-2])
            ne1 = tf.math.sqrt(e_para)
            e_perp, _, _, _ = currentMat.getOrdinaryPermittivity(self.rx[-1], self.ry[-1], self.rz[-1])
            no2 = tf.math.sqrt(e_perp)
            e_para, _, _, _ = currentMat.getExtraordinaryPermittivity(self.rx[-1], self.ry[-1], self.rz[-1])
            o1, _, _, _, _, _, _, _, _, _ = self.Mat.getDirector(self.rx[-2], self.ry[-2], self.rz[-2])
            o2, _, _, _, _, _, _, _, _, _ = currentMat.getDirector(self.rx[-1], self.ry[-1], self.rz[-1])
            
            p_ro, p_re, p_to, p_te, E_ro, E_re, E_to, E_te, S_ro, S_re, S_to, S_te = ia.Anisotropic_Anisotropic(n, o1, o2, no1, ne1, no2, ne2, [self.px[-2], self.py[-2], self.pz[-2]], self.Efield[-2])
            # Initialize the reflected ordinary ray:
            check = hp.validRay(S_ro, p_ro)
            if check:
                p_ro = tf.math.real(p_ro)
                ray1 = Ray(self.rx[-2], self.ry[-2], self.rz[-2], p_ro[0], p_ro[1], p_ro[2], S_ro, E_ro, self.Mat, True)
                rays.append(ray1)
            # Initialize the reflected extraordinary ray:
            check = hp.validRay(S_re, p_re)
            if check:
                p_re = tf.math.real(p_re)
                ray2 = Ray(self.rx[-2], self.ry[-2], self.rz[-2], p_re[0], p_re[1], p_re[2], S_re, E_re, self.Mat, False)
                rays.append(ray2)
            # Initialize the transmitted ordinary ray:
            check = hp.validRay(S_to, p_to)
            if check:
                p_to = tf.math.real(p_to)
                ray3 = Ray(self.rx[-1], self.ry[-1], self.rz[-1], p_to[0], p_to[1], p_to[2], S_to, E_to, currentMat, True)
                rays.append(ray3)
            # Initialize the transmitted extraordinary ray:
            check = hp.validRay(S_te, p_te)
            if check:
                p_te = tf.math.real(p_te)
                ray4 = Ray(self.rx[-1], self.ry[-1], self.rz[-1], p_te[0], p_te[1], p_te[2], S_te, E_te, currentMat, False)
                rays.append(ray4)
            return rays