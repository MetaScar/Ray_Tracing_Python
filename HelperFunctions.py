import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# This function calculates and returns the 3x3 rotation matrix A needed to rotate the optical axis in the principle coordinate system.
# In other words, it finds the matrix A that satifies A*o = <0,0,1>. The input o must be a 3x1 unit vector.
def findRotationMatrix(o):
    phi1 = tf.atan2(tf.math.real(o[1]), tf.math.real(o[0])) # Angle of o prcted onto the xy plane w.r.t the x-axis
    Az = tf.stack([[tf.cos(-phi1), -tf.sin(-phi1), 0], [tf.sin(-phi1), tf.cos(-phi1), 0], [0,0,1]]) # Rotation matrix to rotate o to the xz plane (Rotation of -phi1 about z-axis)
    oxz = Az@tf.stack([[tf.math.real(o[0])], [tf.math.real(o[1])], [tf.math.real(o[2])]]) # Multiplying by the rotation matrix Az
    phi2 = tf.atan2(tf.math.real(oxz[0, 0]), tf.math.real(oxz[2, 0])) # Angle of oxz, moving from the z-axis towards the x-axis
    Ay = tf.stack([[tf.cos(-phi2), 0, tf.sin(-phi2)], [0,1,0], [-tf.sin(-phi2), 0, tf.cos(phi2)]])
    Ay = tf.cast(Ay, dtype = tf.complex64)
    Az = tf.cast(Az, tf.complex64)
    R_matrix = Ay@Az # The product of the two matrices is the overall 3x3 Rotation matrix
    return R_matrix

# Given no and ne of medium 1, the tangential component of the wave normal in the p.c.s, and the surface normal vector in the p.c.s.,
# this function calculates the ordinary and extraordinary reflected wave normals (pro and pre).
def findReflectedNormals(no, ne, ptn, n):
    po = ptn - (tf.math.sqrt(no**2 - tf.norm(ptn)**2))*n
    A = (n[2]**2)/no**2 + (n[0]**2 + n[1]**2)/ne**2
    B = 2*ptn[2]*n[2]/no**2 + (2*ptn[0]*n[0] + 2*ptn[1]*n[1])/ne**2
    C = ptn[2]**2/no**2 + (ptn[0]**2 + ptn[1]**2)/ne**2 - 1
    discriminant = B**2 - 4*A*C
    # Only keep real parts of A, B, C, and discriminant:
    A = tf.cast(tf.math.real(A), tf.complex64)
    B = tf.cast(tf.math.real(B), tf.complex64)
    C = tf.cast(tf.math.real(C), tf.complex64)
    discriminant = tf.cast(tf.math.real(discriminant), tf.complex64)
    xi = (-B - tf.math.sqrt(discriminant))/(2*A)
    pe = ptn + xi*n
    return po, pe

# Given no and ne of medium 2, the tangential component of the wave normal in the p.c.s of medium 2, and the surface normal vector in the p.c.s. of medium 2,
# this function calculates the ordinary and extraordinary transmitted wave normals (pto and pte).
def findTransmittedNormals(no, ne, ptn, n):
    po = ptn + (tf.math.sqrt(no**2 - tf.norm(ptn)**2))*n
    A = (n[2]**2)/no**2 + (n[0]**2 + n[1]**2)/ne**2
    B = 2*ptn[2]*n[2]/no**2 + (2*ptn[0]*n[0] + 2*ptn[1]*n[1])/ne**2
    C = ptn[2]**2/no**2 + (ptn[0]**2 + ptn[1]**2)/ne**2 - 1
    discriminant = B**2 - 4*A*C
    # Only keep real parts of A, B, C, and discriminant:
    A = tf.cast(tf.math.real(A), tf.complex64)
    B = tf.cast(tf.math.real(B), tf.complex64)
    C = tf.cast(tf.math.real(C), tf.complex64)
    discriminant = tf.cast(tf.math.real(discriminant), tf.complex64)
    xi = (-1.0*B + tf.math.sqrt(discriminant))/(2*A)
    pe = ptn + xi*n
    return po, pe

# Given the associated extraordinary wave normal vector (either p_re or p_te) and optical axis, this function
# calculates the associated electric polarization vector in the original coordinate system.
# This is achieved by applying equation (57) from the paper in the principle coordinate system of the medium,
# and then transforming back to the original coordinate system using the appropriate rotation matrix.
def findEPolVector(A, pe, o, no, ne):
    # Transform pe and o into the principle coordinate system:
    o_p = A@tf.stack([[o[0]], [o[1]], [o[2]]])
    pe_p = A@tf.stack([[pe[0]], [pe[1]], [pe[2]]])
    # Calculate the gradient of He with respect to px, py, pz:
    grad_p_He = tf.stack([[2*pe_p[0,0]/ne**2],[2*pe_p[1,0]/ne**2],[2*pe_p[2,0]/no**2]])
    # Calculate the Electric Polarization vector using (45) from the paper:
    E_pol_p = cross(cross(pe_p, o_p), grad_p_He)/tf.norm(cross(cross(pe_p, o_p), grad_p_He))
    # Transforming back to the original coordinate system:
    E_pol = tf.linalg.inv(A)@E_pol_p
    return E_pol

# This function takes in the wave vector at an anisotropic boundary and calculates the associated incident Electric field vector.
# THIS FUNCTION IS NO LONGER USED. The electric field is now tracked along with ray propagation.
def getEfield(no, ne, px, py, pz, o, ordinary):
    p = [px, py, pz]
    A1 = findRotationMatrix(tf.transpose(tf.squeeze(o)))
    p_p = A1@tf.stack([[p[0]], [p[1]], [p[2]]])
    o_p = A1@tf.stack([[o[0]], [o[1]], [o[2]]])

    if ordinary:
        E_p = tf.transpose(tf.linalg.cross(tf.transpose(p_p), tf.transpose(o_p)))/tf.norm(tf.linalg.cross(tf.transpose(p_p), tf.transpose(o_p)))
        E = tf.transpose(tf.linalg.inv(A1)@E_p)
        return E
    else:
        grad_p_He = tf.stack([[2*p_p[0,0]/ne**2],[2*p_p[1,0]/ne**2],[2*p_p[2,0]/no**2]])
        E_p = tf.transpose(tf.linalg.cross(tf.linalg.cross(tf.transpose(p_p), tf.transpose(o_p)), tf.transpose(grad_p_He))/tf.norm(tf.linalg.cross(tf.linalg.cross(tf.transpose(p_p), tf.transpose(o_p)), tf.transpose(grad_p_He))))
        E = tf.transpose(tf.linalg.inv(A1)@E_p)
        return E

# This function contains the analytical functions for the director_profile and its derivatives.
# The derivatives must be analytically calculated by hand.
def  getDirector(x,y,z, c0, c1, c2):
    theta_d = c0 + c1*z + c2*z**2
    director = tf.stack([tf.math.sin(theta_d), tf.constant(0.0), tf.math.cos(theta_d)])
    ddx_x = tf.constant(0.0)
    ddx_y = tf.constant(0.0)
    ddx_z = (c1 + 2*c2*z)*tf.math.cos(theta_d)
    ddy_x = tf.constant(0.0)
    ddy_y = tf.constant(0.0)
    ddy_z = tf.constant(0.0)
    ddz_x = tf.constant(0.0)
    ddz_y = tf.constant(0.0)
    ddz_z = -1*(c1 + 2*c2*z)*tf.math.sin(theta_d)

    return director, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z

# These function contains the analytical functions for no and ne and their spatial derivatives.
# The derivatives must be analytically calculated by hand.
def getOrdinaryIndex(x,y,z, a0, a1, a2):
    e_perp = a1 - a2*(x**2 + y**2 + z**2)/a0**2
    deperp_dx = -2*x*a2/a0**2
    deperp_dy = -2*y*a2/a0**2
    deperp_dz = -2*z*a2/a0**2
    return e_perp, deperp_dx, deperp_dy, deperp_dz

def getExtraordinaryIndex(x,y,z, b0, b1, b2):
    ne = b0 + b1*x + b2*x**2
    e_para = ne**2
    dne_dx = b1 * 2*b2*x
    depara_dx = 2*ne*dne_dx
    depara_dy = tf.constant(0.0)
    depara_dz = tf.constant(0.0)
    return e_para, depara_dx, depara_dy, depara_dz

# This function takes in a list of materials and a specified 3D coordinate, then returns the material that corresponds to that coordinate.
# This code works for either rectangular slabs or concentric spheres (all materials must be the same type!)
def getMaterialAtCoordinate(materials, coordinate):
    x = coordinate[0]
    y = coordinate[1]
    z = coordinate[2]
    if materials[0].type == "rect":
        for element in materials:
            if (x>=element.xmin) and (x<=element.xmax) and (y>=element.ymin) and (y<=element.ymax) and (z>=element.zmin) and (z<=element.zmax):
                actual_material = element
                break
        return actual_material
    if materials[0].type == "sphere":
        r = tf.math.sqrt(x**2 + y**2 + z**2)
        for element in materials:
            if (r>=element.rmin) and (r<=element.rmax):
                actual_material = element
                break
        return actual_material


# This function takes in a position (x,y,z) and returns True if the point is within the bouding box, False if outside the bounding box.
# This code works for either rectangular slabs or concentric spheres (all materials must be the same type!)    
def checkBoundary(boundingBox, coordinate):
    x = coordinate[0]
    y = coordinate[1]
    z = coordinate[2]
    if boundingBox.type == "rect":
        if (x>=boundingBox.xmin) and (x<=boundingBox.xmax) and (y>=boundingBox.ymin) and (y<=boundingBox.ymax) and (z>=boundingBox.zmin) and (z<=boundingBox.zmax):
            return True
        else:
            return False
    if boundingBox.type == "sphere":
        r = tf.math.sqrt(x**2 + y**2 + z**2)
        if (r<=boundingBox.rmax):
            return True
        else:
            return False
    

# This function takes in a current Material and a previous material and calculates the surface normal unit vector.
# This code works for either rectangular slabs or concentric spheres (all materials must be the same type!)    
def getSurfaceNormal(currentMat, prevMat, coordinate):
    if currentMat.type == "rect":
        if currentMat.zmax >= prevMat.zmax:
            return [0.0, 0.0, 1.0]
        else:
            return [0.0, 0.0, -1.0]
    if currentMat.type == "sphere":
        if prevMat.rmax > currentMat.rmax:
            return [coordinate[0].numpy(), coordinate[1].numpy(), coordinate[2].numpy()]/(-1.0*tf.norm(coordinate).numpy())
        else:
            return [coordinate[0].numpy(), coordinate[1].numpy(), coordinate[2].numpy()]/tf.norm(coordinate).numpy()

# This function calculates the cross product of two, in general complex, 1x3 vectors.
def cross(a, b):
    result = [tf.constant(0.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64), tf.constant(0.0, dtype=tf.complex64)]
    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - a[1]*b[0]
    return result

# This function simultaneously checks whether a ray has a negligibly small magnitude and/or is an evanescent wave.
# If either condition is true, the function will return False, indicating that the ray should not be traced.
def validRay(S, p):
    check = True
    if S <= tf.constant(1.0e-10, dtype=tf.float32):
        check = False
    if tf.abs(tf.math.imag(p[0])) >= tf.constant(1.0e-5, dtype=tf.float32) or tf.abs(tf.math.imag(p[1])) >= tf.constant(1.0e-5, dtype=tf.float32) or tf.abs(tf.math.imag(p[2])) >= tf.constant(1.0e-5, dtype=tf.float32):
        check = False
    return check

# This is a function to test autodiff using Tensorflow.
def testfunc(x, y, z):
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    z = tf.constant(z, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)
        tape.watch(z)

        u = 2*x + y
        v = x**2 - z
        f = u + v

    df_dx = tape.gradient(f, x)
    df_dy = tape.gradient(f, y)
    df_dz = tape.gradient(f, z)
    return f, df_dx, df_dy, df_dz
