import tensorflow as tf
import matplotlib.pyplot as plt
import HelperFunctions as hp

# This function calculates reflected and transmitted wave normals, as well as reflected and trasmitted E-field and Poynting vectors for an anisotropic-anisotropic interface.
# Note that all input vectors must be expressed in cartesian coords. as 3x1 'vertical' vectors.
def Anisotropic_Anisotropic(surface_normal, optical_axis_1, optical_axis_2, no1, ne1, no2, ne2, p, E_i):

    # Cast everything to complex 64 (except Ei, which should already be complex):
    surface_normal = tf.cast(surface_normal, dtype=tf.complex64)
    optical_axis_1 = tf.cast(optical_axis_1, dtype=tf.complex64)
    optical_axis_2 = tf.cast(optical_axis_2, dtype=tf.complex64)
    no1 = tf.cast(no1, dtype=tf.complex64)
    ne1 = tf.cast(ne1, dtype=tf.complex64)
    no2 = tf.cast(no2, dtype=tf.complex64)
    ne2 = tf.cast(ne2, dtype=tf.complex64)
    p = tf.cast(p, dtype=tf.complex64)

    ### Section 1: Calculation of Wave Normals. ###
    A1 = hp.findRotationMatrix(optical_axis_1)

    # Transform optical_axis_1, surface_normal, and S_i into the principle C.S. of medium 1 by using matrix A1:
    surface_normal_p1 = A1@tf.stack([[surface_normal[0]], [surface_normal[1]], [surface_normal[2]]])
    p_i_p1 = A1@tf.stack([[p[0]], [p[1]], [p[2]]])

    p_tn_p1 = p_i_p1 - tf.tensordot(tf.squeeze(tf.transpose(p_i_p1)), tf.squeeze(tf.transpose(surface_normal_p1)), axes=1)*surface_normal_p1

    # Next we calculate the reflected ordinary and extraordinary wave normals in the p.c.s of medium 1, which we will denote p_ro_p1 and p_re_p1:

    p_ro_p1, p_re_p1 = hp.findReflectedNormals(no1, ne1, p_tn_p1, surface_normal_p1)

    # The vectors p_ie_p1, p_tn_p1, p_ro_p1, and p_re_p1 are then transformed back to the original coordinate system.
    # This is achieved by multiplication with the inverse of the rotation matrix A1:

    p_i = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_i_p1))
    p_tn = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_tn_p1))
    p_ro = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_ro_p1))
    p_re = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_re_p1))

    # We now need to calculate the wave normals associated with the transmitted waves in medium 2.
    # However, we first need to transform the surface normal vector and p_tn into the principle coordinate system of medium 2.

    A2 = hp.findRotationMatrix(optical_axis_2)
    surface_normal_p2 = A2@tf.stack([[surface_normal[0]], [surface_normal[1]], [surface_normal[2]]])
    p_tn_p2 = A2@tf.stack([[p_tn[0]], [p_tn[1]], [p_tn[2]]])

    # Now, we can use the formulas [(43)-(45) from the paper] to calculate the transmitted wave normals, remembering that we need to
    # apply the '+' sign in these equations.
    p_to_p2, p_te_p2 = hp.findTransmittedNormals(no2, ne2, p_tn_p2, surface_normal_p2)

    # The transmitted wave normals are then transformed back to the original coordinate system:
    p_to = tf.squeeze(tf.transpose(tf.linalg.inv(A2)@p_to_p2))
    p_te = tf.squeeze(tf.transpose(tf.linalg.inv(A2)@p_te_p2))

    ### Section 2: Calculation of Electric and Magnetic Polarization Vectors. ###
    # Note: All Electric Polarization are unit vectors, while the magnetic polarization vectors are not.

    E_to = tf.squeeze(hp.cross(p_to, optical_axis_2)/tf.norm(hp.cross(p_to, optical_axis_2)))
    E_te = tf.squeeze(hp.findEPolVector(A2, p_te, optical_axis_2, no2, ne2))
    E_ro = tf.squeeze(hp.cross(p_ro, optical_axis_1)/tf.norm(hp.cross(p_ro, optical_axis_1)))
    E_re = tf.squeeze(hp.findEPolVector(A1, p_re, optical_axis_1, no1, ne1))

    H_i = hp.cross(p_i, E_i)
    H_to = hp.cross(p_to, E_to)
    H_te = hp.cross(p_te, E_te)
    H_ro = hp.cross(p_ro, E_ro)
    H_re = hp.cross(p_re, E_re)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = hp.cross(p_i, surface_normal)
    t_p = hp.cross(surface_normal, t_s)
    neg_one = tf.complex(-1.0, 0.0)

    ### Section 3: Calculation of the complex Fresenel coefficients.
    # This is achieved by solving equation (48) from the paper numerically.
    A = tf.stack([[tf.tensordot(t_s, E_to, axes=1), tf.tensordot(t_s, E_te, axes=1), tf.tensordot(neg_one*t_s, E_ro, axes=1), tf.tensordot(neg_one*t_s, E_re, axes=1)],
                 [tf.tensordot(t_p, E_to, axes=1), tf.tensordot(t_p, E_te, axes=1), tf.tensordot(neg_one*t_p, E_ro, axes=1), tf.tensordot(neg_one*t_p, E_re, axes=1)],
                 [tf.tensordot(t_s, H_to, axes=1), tf.tensordot(t_s, H_te, axes=1), tf.tensordot(neg_one*t_s, H_ro, axes=1), tf.tensordot(neg_one*t_s, H_re, axes=1)],
                 [tf.tensordot(t_p, H_to, axes=1), tf.tensordot(t_p, H_te, axes=1), tf.tensordot(neg_one*t_p, H_ro, axes=1), tf.tensordot(neg_one*t_p, H_re, axes=1)]])
    
    b = tf.stack([[tf.tensordot(t_s, E_i, axes=1)], [tf.tensordot(t_p, E_i, axes=1)], [tf.tensordot(t_s, H_i, axes=1)], [tf.tensordot(t_p, H_i, axes=1)]])

    fresnel_coefs = tf.linalg.inv(A)@b

    a_to = fresnel_coefs[0]
    a_te = fresnel_coefs[1]
    a_ro = fresnel_coefs[2]
    a_re = fresnel_coefs[3]

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector (which has a magnitude of 1)
    S_i_unormalized = 0.5*tf.math.real(hp.cross(E_i, tf.math.conj(H_i)))
    S_to = 0.5*tf.math.real(hp.cross(a_to*E_to, tf.math.conj(a_to*H_to)))/tf.norm(S_i_unormalized)
    S_te = 0.5*tf.math.real(hp.cross(a_te*E_te, tf.math.conj(a_te*H_te)))/tf.norm(S_i_unormalized)
    S_ro = 0.5*tf.math.real(hp.cross(a_ro*E_ro, tf.math.conj(a_ro*H_ro)))/tf.norm(S_i_unormalized)
    S_re = 0.5*tf.math.real(hp.cross(a_re*E_re, tf.math.conj(a_re*H_re)))/tf.norm(S_i_unormalized)

    # Calculation of Electric Polarization vectors (accounting for phase shifts introduced by the Fresnel coefficients):
    E_to = a_to*E_to
    E_to = E_to/tf.norm(E_to)
    E_te = a_te*E_te
    E_te = E_te/tf.norm(E_te)
    E_ro =a_ro*E_ro
    E_ro = E_ro/tf.norm(E_ro)
    E_re = a_re*E_re
    E_re = E_re/tf.norm(E_re)

    # Poynting Vector Magnitudes:
    S_ro = tf.norm(S_ro)
    S_re = tf.norm(S_re)
    S_to = tf.norm(S_to)
    S_te = tf.norm(S_te)
    
    return p_ro, p_re, p_to, p_te, E_ro, E_re, E_to, E_te, S_ro, S_re, S_to, S_te

# This function calculates the complex Fresnel coefficients, as well as Transmittance and Reflectance factors for an anisotropic-isotropic interface.
# Note that all input vectors must be expressed in cartesian coords. as 3x1 'vertical' vectors.
def Anisotropic_Isotropic(surface_normal, optical_axis_1, no1, ne1, no2, p, E_i):

    # Cast everything to complex 64 (except Ei, which should already be complex):
    surface_normal = tf.cast(surface_normal, dtype=tf.complex64)
    optical_axis_1 = tf.cast(optical_axis_1, dtype=tf.complex64)
    no1 = tf.cast(no1, dtype=tf.complex64)
    ne1 = tf.cast(ne1, dtype=tf.complex64)
    no2 = tf.cast(no2, dtype=tf.complex64)
    p = tf.cast(p, dtype=tf.complex64)

    # E_i = tf.squeeze(E_i) # Need to think about this line of code

    ### Section 1: Calculation of Wave Normals. ###
    A1 = hp.findRotationMatrix(optical_axis_1)

    # Transform optical_axis_1, surface_normal, and S_i into the principle C.S. of medium 1 by using matrix A1:
    surface_normal_p1 = A1@tf.stack([[surface_normal[0]], [surface_normal[1]], [surface_normal[2]]])
    p_i_p1 = A1@tf.stack([[p[0]], [p[1]], [p[2]]])

    p_tn_p1 = p_i_p1 - tf.tensordot(tf.squeeze(tf.transpose(p_i_p1)), tf.squeeze(tf.transpose(surface_normal_p1)), axes=1)*surface_normal_p1

    # Next we calculate the reflected ordinary and extraordinary wave normals in the p.c.s of medium 1, which we will denote p_ro_p1 and p_re_p1:

    p_ro_p1, p_re_p1 = hp.findReflectedNormals(no1, ne1, p_tn_p1, surface_normal_p1)

    # The vectors p_ie_p1, p_tn_p1, p_ro_p1, and p_re_p1 are then transformed back to the original coordinate system.
    # This is achieved by multiplication with the inverse of the rotation matrix A1:

    p_i = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_i_p1))
    p_tn = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_tn_p1))
    p_ro = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_ro_p1))
    p_re = tf.squeeze(tf.transpose(tf.linalg.inv(A1)@p_re_p1))

    # We now need to calculate the wave normal associated with the transmitted wave in medium 2 (the isotropic medium)
    p_t = p_tn + (tf.math.sqrt(no2**2 - tf.norm(p_tn)**2))*surface_normal

    ### Section 2: Calculation of Electric and Magnetic Polarization Vectors.

    E_ts = tf.squeeze(hp.cross(p_t, surface_normal)/tf.norm(hp.cross(p_t, surface_normal)))
    E_tp = tf.squeeze(hp.cross(E_ts, p_t)/tf.norm(hp.cross(E_ts, p_t)))
    E_ro = tf.squeeze(hp.cross(p_ro, optical_axis_1)/tf.norm(hp.cross(p_ro, optical_axis_1)))
    E_re = tf.squeeze(hp.findEPolVector(A1, p_re, optical_axis_1, no1, ne1))

    H_i = hp.cross(p_i, E_i)
    H_ts = hp.cross(p_t, E_ts)
    H_tp = hp.cross(p_t, E_tp)
    H_ro = hp.cross(p_ro, E_ro)
    H_re = hp.cross(p_re, E_re)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = hp.cross(p_i, surface_normal)
    t_p = hp.cross(surface_normal, t_s)
    neg_one = tf.complex(-1.0, 0.0)

    ### Section 3: Calculation of the complex Fresenel coefficients.
    # This is achieved by solving equation (48) from the paper numerically.
    A = tf.stack([[tf.tensordot(t_s, E_ts, axes=1), tf.tensordot(t_s, E_tp, axes=1), tf.tensordot(neg_one*t_s, E_ro, axes=1), tf.tensordot(neg_one*t_s, E_re, axes=1)],
                 [tf.tensordot(t_p, E_ts, axes=1), tf.tensordot(t_p, E_tp, axes=1), tf.tensordot(neg_one*t_p, E_ro, axes=1), tf.tensordot(neg_one*t_p, E_re, axes=1)],
                 [tf.tensordot(t_s, H_ts, axes=1), tf.tensordot(t_s, H_tp, axes=1), tf.tensordot(neg_one*t_s, H_ro, axes=1), tf.tensordot(neg_one*t_s, H_re, axes=1)],
                 [tf.tensordot(t_p, H_ts, axes=1), tf.tensordot(t_p, H_tp, axes=1), tf.tensordot(neg_one*t_p, H_ro, axes=1), tf.tensordot(neg_one*t_p, H_re, axes=1)]])
    
    b = tf.stack([[tf.tensordot(t_s, E_i, axes=1)], [tf.tensordot(t_p, E_i, axes=1)], [tf.tensordot(t_s, H_i, axes=1)], [tf.tensordot(t_p, H_i, axes=1)]])

    fresnel_coefs = tf.linalg.inv(A)@b

    a_ts = fresnel_coefs[0]
    a_tp = fresnel_coefs[1]
    a_ro = fresnel_coefs[2]
    a_re = fresnel_coefs[3]

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.math.real(hp.cross(E_i, tf.math.conj(H_i)))
    S_ro = 0.5*tf.math.real(hp.cross(a_ro*E_ro, tf.math.conj(a_ro*H_ro)))/tf.norm(S_i_unormalized)
    S_re = 0.5*tf.math.real(hp.cross(a_re*E_re, tf.math.conj(a_re*H_re)))/tf.norm(S_i_unormalized)
    
    # Calculate E_t and S_t:
    E_t = a_ts*E_ts + a_tp*E_tp
    H_t = a_ts*H_ts + a_tp*H_tp
    S_t = 0.5*tf.math.real(hp.cross(E_t, tf.math.conj(H_t)))/tf.norm(S_i_unormalized)
    E_t = E_t/tf.norm(E_t)

    # Calculation of Electric Polarization vectors (accounting for phase shifts introduced by the Fresnel coefficients):
    E_ro =a_ro*E_ro
    E_ro = E_ro/tf.norm(E_ro)
    E_re = a_re*E_re
    E_re = E_re/tf.norm(E_re) 

    # Poynting Vector Magnitudes:
    S_ro = tf.norm(S_ro)
    S_re = tf.norm(S_re)
    S_t = tf.norm(S_t)
    
    return p_ro, p_re, p_t, E_ro, E_re, E_t, S_ro, S_re, S_t

# This function calculates the transmitted and reflected wave normals, the complex Fresnel coefficients, 
# and Transmittance and Reflectance factors for an isotropic-anisotropic interface.
def Isotropic_Anisotropic(surface_normal, optical_axis_2, no1, no2, ne2, p, E_i):

    # Cast everything to complex 64 (except Ei, which should already be complex):
    surface_normal = tf.cast(surface_normal, dtype=tf.complex64)
    optical_axis_2 = tf.cast(optical_axis_2, dtype=tf.complex64)
    no1 = tf.cast(no1, dtype=tf.complex64)
    no2 = tf.cast(no2, dtype=tf.complex64)
    ne2 = tf.cast(ne2, dtype=tf.complex64)
    p = tf.cast(p, dtype=tf.complex64)

    # E_i = tf.squeeze(E_i) # Need to think about this line of code

    ### Section 1: Calculation of Wave Normals. ###

    # First, we need to calculate the incident wave normal and its component tangential to the boundary.

    p_i = p
    p_tn = p_i - tf.tensordot(p_i, tf.cast(surface_normal, tf.complex64), axes=1)*surface_normal

    # Next, we calculate the reflected wave normal:
    p_r = p_tn - (tf.math.sqrt(no1**2 - tf.norm(p_tn)**2))*surface_normal

    # We now need to calculate the wave normals associated with the transmitted waves in medium 2.
    # However, we first need to transform the surface normal vector and p_tn into the principle coordinate system of medium 2.
    
    A2 = hp.findRotationMatrix(optical_axis_2)
    surface_normal_p2 = A2@tf.stack([[surface_normal[0]], [surface_normal[1]], [surface_normal[2]]])
    p_tn_p2 = A2@tf.stack([[p_tn[0]], [p_tn[1]], [p_tn[2]]])

    # Now, we can use the formulas [(43)-(45) from the paper] to calculate the transmitted wave normals, remembering that we need to
    # apply the '+' sign in these equations.
    p_to_p2, p_te_p2 = hp.findTransmittedNormals(no2, ne2, p_tn_p2, surface_normal_p2)

    # The transmitted wave normals are then transformed back to the original coordinate system:
    p_to = tf.squeeze(tf.transpose(tf.linalg.inv(A2)@p_to_p2))
    p_te = tf.squeeze(tf.transpose(tf.linalg.inv(A2)@p_te_p2))

    ### Section 2. Calculation of Electric and Magnetic Polarization Vectors. ###
    E_to = tf.squeeze(hp.cross(p_to, optical_axis_2)/tf.norm(hp.cross(p_to, optical_axis_2)))
    E_te = tf.squeeze(hp.findEPolVector(A2, p_te, optical_axis_2, no2, ne2))
    E_rs = tf.squeeze(hp.cross(p_r, surface_normal)/tf.norm(hp.cross(p_r, surface_normal)))
    E_rp = tf.squeeze(hp.cross(E_rs, p_r)/tf.norm(hp.cross(E_rs, p_r)))

    H_i = hp.cross(p_i, E_i)
    H_to = hp.cross(p_to, E_to)
    H_te = hp.cross(p_te, E_te)
    H_rs = hp.cross(p_r, E_rs)
    H_rp = hp.cross(p_r, E_rp)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = hp.cross(p_i, surface_normal)
    t_p = hp.cross(surface_normal, t_s)
    neg_one = tf.complex(-1.0, 0.0)

    ### Section 3: Calculation of the complex Fresenel coefficients.
    # This is achieved by solving equation (48) from the paper numerically.
    A = tf.stack([[tf.tensordot(t_s, E_to, axes=1), tf.tensordot(t_s, E_te, axes=1), tf.tensordot(neg_one*t_s, E_rs, axes=1), tf.tensordot(neg_one*t_s, E_rp, axes=1)],
                 [tf.tensordot(t_p, E_to, axes=1), tf.tensordot(t_p, E_te, axes=1), tf.tensordot(neg_one*t_p, E_rs, axes=1), tf.tensordot(neg_one*t_p, E_rp, axes=1)],
                 [tf.tensordot(t_s, H_to, axes=1), tf.tensordot(t_s, H_te, axes=1), tf.tensordot(neg_one*t_s, H_rs, axes=1), tf.tensordot(neg_one*t_s, H_rp, axes=1)],
                 [tf.tensordot(t_p, H_to, axes=1), tf.tensordot(t_p, H_te, axes=1), tf.tensordot(neg_one*t_p, H_rs, axes=1), tf.tensordot(neg_one*t_p, H_rp, axes=1)]])
    
    b = tf.stack([[tf.tensordot(t_s, E_i, axes=1)], [tf.tensordot(t_p, E_i, axes=1)], [tf.tensordot(t_s, H_i, axes=1)], [tf.tensordot(t_p, H_i, axes=1)]])

    fresnel_coefs = tf.linalg.inv(A)@b

    a_to = fresnel_coefs[0]
    a_te = fresnel_coefs[1]
    a_rs = fresnel_coefs[2]
    a_rp = fresnel_coefs[3]

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.math.real(hp.cross(E_i, tf.math.conj(H_i)))
    S_to = 0.5*tf.math.real(hp.cross(a_to*E_to, tf.math.conj(a_to*H_to)))/tf.norm(S_i_unormalized)
    S_te = 0.5*tf.math.real(hp.cross(a_te*E_te, tf.math.conj(a_te*H_te)))/tf.norm(S_i_unormalized)
    
    # Calculate E_r and S_r:
    E_r = a_rs*E_rs + a_rp*E_rp
    H_r = a_rs*H_rs + a_rp*H_rp
    S_r = 0.5*tf.math.real(hp.cross(E_r, tf.math.conj(H_r)))/tf.norm(S_i_unormalized)
    E_r = E_r/tf.norm(E_r)

    # Calculation of Electric Polarization vectors (accounting for phase shifts introduced by the Fresnel coefficients):
    E_to = a_to*E_to
    E_to = E_to/tf.norm(E_to)
    E_te = a_te*E_te
    E_te = E_te/tf.norm(E_te)

    # Poynting Vector Magnitudes:
    S_r = tf.norm(S_r)
    S_to = tf.norm(S_to)
    S_te = tf.norm(S_te)

    return p_r, p_to, p_te, E_r, E_to, E_te, S_r, S_to, S_te

# This function calculates the complex Fresnel coefficients, as well as Transmittance and Reflectance factors for an isotropic-isotropic interface.
def Isotropic_Isotropic(surface_normal, no1, no2, p, E_i):

    # Cast everything to complex 64 (except Ei, which should already be complex):
    surface_normal = tf.cast(surface_normal, dtype=tf.complex64)
    no1 = tf.cast(no1, dtype=tf.complex64)
    no2 = tf.cast(no2, dtype=tf.complex64)
    p = tf.cast(p, dtype=tf.complex64)

    ### Section 1: Calculation of Wave Normals. ###

    # First, we need to calculate the incident wave normal and its component tangential to the boundary.

    p_i = p
    p_tn = p_i - tf.tensordot(p_i, surface_normal, axes=1)*surface_normal

    # Next, we calculate the reflected wave normal:
    p_r = p_tn - (tf.math.sqrt(no1**2 - tf.norm(p_tn)**2))*surface_normal

    # We now need to calculate the wave normal associated with the transmitted wave in medium 2.
    p_t = p_tn + (tf.math.sqrt(no2**2 - tf.norm(p_tn)**2))*surface_normal

    ### Section 2: Calculation of Electric and Magnetic Polarization Vectors. ###

    E_ts = (hp.cross(p_t, surface_normal)/tf.norm(hp.cross(p_t, surface_normal)))
    E_tp = (hp.cross(E_ts, p_t)/tf.norm(hp.cross(E_ts, p_t)))
    E_rs = (hp.cross(p_r, surface_normal)/tf.norm(hp.cross(p_r, surface_normal)))
    E_rp = (hp.cross(E_rs, p_r)/tf.norm(hp.cross(E_rs, p_r)))

    H_i = hp.cross(p_i, E_i)
    H_ts = hp.cross(p_t, E_ts)
    H_tp = hp.cross(p_t, E_tp)
    H_rs = hp.cross(p_r, E_rs)
    H_rp = hp.cross(p_r, E_rp)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = hp.cross(p_i, surface_normal)
    t_p = hp.cross(surface_normal, t_s)
    neg_one = tf.complex(-1.0, 0.0)

    ### Section 3: Calculation of the complex Fresenel coefficients. ###

    A = tf.stack([[tf.tensordot(t_s, E_ts, axes=1), tf.tensordot(t_s, E_tp, axes=1), tf.tensordot(neg_one*t_s, E_rs, axes=1), tf.tensordot(neg_one*t_s, E_rp, axes=1)],
                 [tf.tensordot(t_p, E_ts, axes=1), tf.tensordot(t_p, E_tp, axes=1), tf.tensordot(neg_one*t_p, E_rs, axes=1), tf.tensordot(neg_one*t_p, E_rp, axes=1)],
                 [tf.tensordot(t_s, H_ts, axes=1), tf.tensordot(t_s, H_tp, axes=1), tf.tensordot(neg_one*t_s, H_rs, axes=1), tf.tensordot(neg_one*t_s, H_rp, axes=1)],
                 [tf.tensordot(t_p, H_ts, axes=1), tf.tensordot(t_p, H_tp, axes=1), tf.tensordot(neg_one*t_p, H_rs, axes=1), tf.tensordot(neg_one*t_p, H_rp, axes=1)]])
    
    b = tf.stack([[tf.tensordot(t_s, E_i, axes=1)], [tf.tensordot(t_p, E_i, axes=1)], [tf.tensordot(t_s, H_i, axes=1)], [tf.tensordot(t_p, H_i, axes=1)]])

    fresnel_coefs = tf.linalg.inv(A)@b

    a_ts = fresnel_coefs[0]
    a_tp = fresnel_coefs[1]
    a_rs = fresnel_coefs[2]
    a_rp = fresnel_coefs[3]

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.math.real(hp.cross(E_i, tf.math.conj(H_i)))
    
    # Calculate E_t and S_t:
    E_t = a_ts*E_ts + a_tp*E_tp
    H_t = a_ts*H_ts + a_tp*H_tp
    S_t = 0.5*tf.math.real(hp.cross(E_t, tf.math.conj(H_t)))/tf.norm(S_i_unormalized)
    E_t = E_t/tf.norm(E_t) 

    # Calculate E_r and S_r:
    E_r = a_rs*E_rs + a_rp*E_rp
    H_r = a_rs*H_rs + a_rp*H_rp
    S_r = 0.5*tf.math.real(hp.cross(E_r, tf.math.conj(H_r)))/tf.norm(S_i_unormalized)
    E_r = E_r/tf.norm(E_r)

    # Poynting Vector Magnitudes:
    S_r = tf.norm(S_r)
    S_t = tf.norm(S_t)

    return p_r, p_t, E_r, E_t, S_r, S_t




