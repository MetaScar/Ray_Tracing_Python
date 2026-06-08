import tensorflow as tf
import tensorflow_probability as tfp
import math as m
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# These functions are used by the new tensor-based ray-tracing algorithm.
# The functions are kept in this seperate Python file for clarity.

# This function computes a single time step for ALL rays at once, whether they are ordinary or not.
def rayPropagation(positions, wave_normals, Efields, ordinary, alive, ordinary_constants, extraordinary_constants, director_constants, isotropic, material_IDs, distance_step):

    # Calculate e_perp, e_parallel, the director, and associated spatial derivatives for all rays:
    e_perp, deperp_dx, deperp_dy, deperp_dz = getOrdinaryPermittivities(positions, ordinary_constants)
    e_para, depara_dx, depara_dy, depara_dz = getExtraordinaryPermittivities(positions, extraordinary_constants)
    d_local, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z = getDirector(positions, director_constants)

    # Calculate k1-k6 (first-order derivatives) for each ray (assuming extraordinary:)
    k1_e = -2*(e_para - e_perp)*tf.reduce_sum(wave_normals*d_local, axis=1)*(wave_normals[:, 0]*ddx_x + wave_normals[:, 1]*ddy_x + wave_normals[:, 2]*ddz_x) + e_perp*depara_dx + (e_para - (tf.norm(wave_normals, axis=1)**2))*deperp_dx - (depara_dx - deperp_dx)*(tf.reduce_sum(wave_normals*d_local, axis=1))**2
    k2_e = -2*(e_para - e_perp)*tf.reduce_sum(wave_normals*d_local, axis=1)*(wave_normals[:, 0]*ddx_y + wave_normals[:, 1]*ddy_y + wave_normals[:, 2]*ddz_y) + e_perp*depara_dy + (e_para - (tf.norm(wave_normals, axis=1)**2))*deperp_dy - (depara_dy - deperp_dy)*(tf.reduce_sum(wave_normals*d_local, axis=1))**2
    k3_e = -2*(e_para - e_perp)*tf.reduce_sum(wave_normals*d_local, axis=1)*(wave_normals[:, 0]*ddx_z + wave_normals[:, 1]*ddy_z + wave_normals[:, 2]*ddz_z) + e_perp*depara_dz + (e_para - (tf.norm(wave_normals, axis=1)**2))*deperp_dz - (depara_dz - deperp_dz)*(tf.reduce_sum(wave_normals*d_local, axis=1))**2
    k4_e = 2*e_perp*wave_normals[:, 0] + 2*(e_para - e_perp)*tf.reduce_sum(wave_normals*d_local, axis=1)*d_local[:, 0]
    k5_e = 2*e_perp*wave_normals[:, 1] + 2*(e_para - e_perp)*tf.reduce_sum(wave_normals*d_local, axis=1)*d_local[:, 1]
    k6_e = 2*e_perp*wave_normals[:, 2] + 2*(e_para - e_perp)*tf.reduce_sum(wave_normals*d_local, axis=1)*d_local[:, 2]

    # Scale k4, k5, and k6 such that the ray travels a fixed distance:
    current_step = tf.math.sqrt(k4_e**2 + k5_e**2 + k6_e**2)
    h = distance_step/current_step
    h = tf.expand_dims(h, axis=1)

    # Time-step using the first-order Runge-Kutta method:
    new_wave_normals_e = wave_normals + (h*tf.stack([k1_e, k2_e, k3_e], axis=1))
    new_positions_e = positions + (h*tf.stack([k4_e, k5_e, k6_e], axis=1))

    # Calculate k1-k6 for each ray (assuming ordinary):
    k1_o = deperp_dx
    k2_o = deperp_dy
    k3_o = deperp_dz
    k4_o = 2*wave_normals[:, 0]
    k5_o = 2*wave_normals[:, 1]
    k6_o = 2*wave_normals[:, 2]

    # Scale k4, k5, and k6 such that the ray travels a fixed distance:
    current_step = tf.math.sqrt(k4_o**2 + k5_o**2 + k6_o**2)
    h = distance_step/current_step
    h = tf.expand_dims(h, axis=1)

    # Time-step using the first-order Runge-Kutta method:
    new_wave_normals_o = wave_normals + (h*(tf.stack([k1_o, k2_o, k3_o], axis=1)))
    new_positions_o = positions + (h*(tf.stack([k4_o, k5_o, k6_o], axis=1)))

    # Use tf.where to update positions and wave vectors for each ray, depending on whether it is ordinary or extraordinary:
    ordinary_expanded = tf.expand_dims(ordinary, axis=1)
    new_wave_normals = tf.where(ordinary_expanded, new_wave_normals_o, new_wave_normals_e)
    new_positions = tf.where(ordinary_expanded, new_positions_o, new_positions_e)
                        
    ## Phase calculation omitted for now... ###
    # Calculate previous phase from previous E-field vector:
    # prev_phase = tf.math.angle(tf.reduce_sum(Efields, axis=1))

    # Calculate phase progression and total phase:
    # delta_phi = ko*(tf.reduce_sum(wave_normals*tf.stack([k4, k5, k6], axis=1)))*h
    # Ephase = -1.0*prev_phase + delta_phi

    # Calculate director (optical axis) as an intermediate step for calculating the electric polarization vector:
    director, _, _, _, _, _, _, _, _, _ = getDirector(positions, director_constants)
    # Calculate electric polarization vector:
    Epol_new = getEfield(e_perp, e_para, new_wave_normals, director, ordinary)

    # If the ray is propagating in an isotropic (and homogeneous medium), the E-field should remain constant:
    ray_isotropic = tf.gather(isotropic, material_IDs)
    ray_isotropic_expanded = tf.expand_dims(ray_isotropic, axis=1)
    E = tf.where(ray_isotropic_expanded, Efields, Epol_new)

    # Account for phase progression:
    # Epol = tf.cast(Epol, dtype=tf.complex64)*tf.math.exp(tf.complex(0.0, -1.0*Ephase))

    # Only update rays that are "alive" (i.e. still being traced):
    alive_expanded = tf.expand_dims(alive, axis=1) # Turns alive from size N to size (N, 1)
    new_wave_normals = tf.where(alive_expanded, new_wave_normals, wave_normals)
    new_positions = tf.where(alive_expanded, new_positions, positions)
    new_Efields = tf.where(alive_expanded, E, Efields)
    
    return new_wave_normals, new_positions, new_Efields


# Given a tensor of ray positions and coefficients described the ordinary profile distribution for each material (lens and air),
# this function returns four rank 1 tensors: e_perp, deperp_dx, deperp_dy, and deperp_dz.
# Important Note: Each material must have a basis function of the same form (the coefficients can be different)
# Another important note: The permittivity distriubtion is passed into a "softplus" function to ensure er >= 1, always.
def getOrdinaryPermittivities(positions, ordinary_constants):

    # # 4th-degree polynomial as a function of x:
    # x = positions[:, 0] # for readability
    # g = ordinary_constants[:, 0] + ordinary_constants[:, 1]*x + ordinary_constants[:, 2]*x**2 + ordinary_constants[:, 3]*x**3 + ordinary_constants[:, 4]*x**4 # g is an intermediate parameterization
    # e_perp = 1 + tf.math.softplus(g)
    # deperp_dx = tf.math.sigmoid(g)*(ordinary_constants[:, 1] + 2.0*ordinary_constants[:, 2]*x + 3.0*ordinary_constants[:, 3]*x**2 + 4.0*ordinary_constants[:, 4]*x**3)
    # deperp_dy = tf.zeros(tf.shape(positions)[0])
    # deperp_dz = tf.zeros(tf.shape(positions)[0])
    # return e_perp, deperp_dx, deperp_dy, deperp_dz

    # # Quadractic as a function of rho:
    # x = positions[:, 0]
    # y = positions[:, 1] 
    # rho = x**2 + y**2 
    # e_max = ordinary_constants[:, 0] 
    # rho_max = ordinary_constants[:, 1]
    # alpha = ordinary_constants[:, 2] # for readability
    # g = e_max*(1 - alpha*(rho/rho_max)**2) # g is an intermediate variable
    # e_perp = 1 + tf.math.softplus(g)
    # deperp_dx = tf.math.sigmoid(g)*(-2.0*e_max*alpha*x/rho_max)
    # deperp_dy = tf.math.sigmoid(g)*(-2.0*e_max*alpha*y/rho_max)
    # deperp_dz = tf.zeros(tf.shape(positions)[0])

    # return e_perp, deperp_dx, deperp_dy, deperp_dz

    # # Quadractic as a function of rho multiplied by a double sigmoid as a function of z:
    # x = positions[:, 0]
    # y = positions[:, 1]
    # z = positions[:, 2]
    # rho = tf.math.sqrt(x**2 + y**2)
    # erb = ordinary_constants[:, 0]
    # e_max = ordinary_constants[:, 1] 
    # rho_max = ordinary_constants[:, 2]
    # alpha = ordinary_constants[:, 3] 
    # C = ordinary_constants[:, 4]
    # zmin = ordinary_constants[:, 5]
    # zmax = ordinary_constants[:, 6] # for readability
    # g_rho = e_max*(1 - alpha*(rho/rho_max)**2) # g_rho and g_z are intermediate variables
    # g_z = tf.math.sigmoid(C*z - zmin) - tf.math.sigmoid(C*z - zmax)
    # e_perp = erb + g_z*tf.math.softplus(g_rho)
    # deperp_dx = g_z*tf.math.sigmoid(g_rho)*(-2.0*e_max*alpha*x/rho_max**2)
    # deperp_dy = g_z*tf.math.sigmoid(g_rho)*(-2.0*e_max*alpha*y/rho_max**2)
    # deperp_dz = tf.math.softplus(g_rho)*C*(tf.math.sigmoid(C*z - zmin)*(1 - tf.math.sigmoid(C*z - zmin)) - (tf.math.sigmoid(C*z - zmax)*(1 - tf.math.sigmoid(C*z - zmax))))
    # return e_perp, deperp_dx, deperp_dy, deperp_dz

    # N-harmonic Fourier series as a function of rho multiplied by a double sigmoid as a function of z:
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    rho = tf.math.sqrt(x**2 + y**2)
    erb = ordinary_constants[:, 0] 
    C = ordinary_constants[:, 1]
    zmin = ordinary_constants[:, 2]
    zmax = ordinary_constants[:, 3] # for readability
    T0 = ordinary_constants[:, 4]
    a0 = ordinary_constants[:, 5]
    an = ordinary_constants[:, 6:]
    N = tf.shape(an)[1]
    N = tf.cast(tf.squeeze(N), dtype=tf.int32)
    n_vec = tf.cast(tf.range(1, N+1)[:, tf.newaxis], dtype=tf.float32)

    w0 = 2*m.pi/T0
    angle = tf.transpose(n_vec * tf.transpose(tf.expand_dims(w0*rho, axis=1)))
    cos_terms = an * tf.cos(angle) # (batch, N)
    a_sum = tf.reduce_sum(cos_terms, axis=1) # (batch,)
    g_rho = a0 + a_sum # g is an intermediate variable
    g_z = tf.math.sigmoid(C*z - zmin) - tf.math.sigmoid(C*z - zmax)
    
    # Computing the derivatives...
    frequencies = tf.expand_dims(w0, axis=1)*tf.transpose(n_vec)
    da_drho = -1.0 * an * frequencies * tf.sin(angle)
    dg_drho = tf.reduce_sum(da_drho, axis=1)

    e_perp = erb + g_z*tf.math.softplus(g_rho)
    deperp_dx = g_z*tf.math.sigmoid(g_rho)*dg_drho*(x/rho)
    deperp_dy = g_z*tf.math.sigmoid(g_rho)*dg_drho*(y/rho)
    deperp_dz = tf.math.softplus(g_rho)*C*(tf.math.sigmoid(C*z - zmin)*(1 - tf.math.sigmoid(C*z - zmin)) - (tf.math.sigmoid(C*z - zmax)*(1 - tf.math.sigmoid(C*z - zmax))))

    return e_perp, deperp_dx, deperp_dy, deperp_dz

    # N-harmonic Fourier series as a function of rho multiplied by an N-harmonic FS as a function of z (cosines only):

#     # Fourier Series in rho:
#     x = positions[:, 0]
#     y = positions[:, 1]
#     rho = tf.math.sqrt(x**2 + y**2)
#     rho_constants = ordinary_constants[:, :tf.cast(tf.shape(ordinary_constants)[1]/2, dtype=tf.int32)]
#     N = tf.shape(rho_constants)[1] - 2
#     N = tf.cast(tf.squeeze(N), dtype=tf.int32)
#     n_vec = tf.cast(tf.range(1, N+1)[:, tf.newaxis], dtype=tf.float32)
#     T0 = rho_constants[:, 0]
#     a0 = rho_constants[:, 1]
#     an = rho_constants[:, 2:N+2]
#     # bn = ordinary_constants[:, N+2:2*N + 2] # Don't need since we want an even function

#     # Generate the cos and sin matrices:
#     w0 = 2*m.pi/T0

#     # Use broadcasting to get shape (batch, N)
#     angle = tf.transpose(n_vec * tf.transpose(tf.expand_dims(w0*rho, axis=1)))

#     cos_terms = an * tf.cos(angle) # (batch, N)
#     # sin_terms = bn * tf.sin(angle) # (batch, N) # Don't need since we want an even function

#     a_sum = tf.reduce_sum(cos_terms, axis=1) # (batch,)
#     # b_sum = tf.reduce_sum(sin_terms, axis=1) # (batch,) # Don't need since we want an even function

#     g_rho = a0 + a_sum # g is an intermediate variable

#     # Pre-calculate the frequencies for the derivative: (n * w0)
#     # Shape: (batch, N)
#     frequencies = tf.expand_dims(w0, axis=1)*tf.transpose(n_vec)

#     # Compute the derivative of the sum element-wise
#     # d/drho [an * cos(n*w0*rho)] = -an * n*w0 * sin(n*w0*rho)
#     da_drho = an * frequencies * tf.sin(angle)
#     # db_drho =  bn * frequencies * tf.cos(angle) # Don't need since we want an even function

#     # Sum across the N components to get (batch,)
#     dg_drho = tf.reduce_sum(da_drho, axis=1)

#     # Fourier Series in z:
#     z = positions[:, 2]
#     z_constants = ordinary_constants[:, tf.cast(tf.shape(ordinary_constants)[1]/2, dtype=tf.int32):]
#     N = tf.shape(z_constants)[1] - 2
#     N = tf.cast(tf.squeeze(N), dtype=tf.int32)
#     n_vec = tf.cast(tf.range(1, N+1)[:, tf.newaxis], dtype=tf.float32)
#     T0 = z_constants[:, 0]
#     a0 = z_constants[:, 1]
#     an = z_constants[:, 2:N+2]
#     # bn = ordinary_constants[:, N+2:2*N + 2] # Don't need since we want an even function

#     # Generate the cos and sin matrices:
#     w0 = 2*m.pi/T0

#     # Use broadcasting to get shape (batch, N)
#     angle = tf.transpose(n_vec * tf.transpose(tf.expand_dims(w0*rho, axis=1)))

#     cos_terms = an * tf.cos(angle) # (batch, N)
#     # sin_terms = bn * tf.sin(angle) # (batch, N) # Don't need since we want an even function

#     a_sum = tf.reduce_sum(cos_terms, axis=1) # (batch,)
#     # b_sum = tf.reduce_sum(sin_terms, axis=1) # (batch,) # Don't need since we want an even function

#     g_z = a0 + a_sum # g is an intermediate variable

#     # Pre-calculate the frequencies for the derivative: (n * w0)
#     # Shape: (batch, N)
#     frequencies = tf.expand_dims(w0, axis=1)*tf.transpose(n_vec)

#     # Compute the derivative of the sum element-wise
#     # d/drho [an * cos(n*w0*rho)] = -an * n*w0 * sin(n*w0*rho)
#     da_dz = an * frequencies * tf.sin(angle)
#     # db_drho =  bn * frequencies * tf.cos(angle) # Don't need since we want an even function

#     # Sum across the N components to get (batch,)
#     dg_dz = tf.reduce_sum(da_drho, axis=1)

#     e_perp = 1.0 + tf.math.softplus(g_rho)*tf.math.softplus(g_z) # Minimum er of 4 (based on substrate)

#     # Chain rule for x, y, and z
#     # Note: Softplus derivative is sigmoid

#     deperp_dx = tf.math.softplus(g_z)*tf.math.sigmoid(g_rho)*dg_drho*(x/rho)
#     deperp_dy = tf.math.softplus(g_z)*tf.math.sigmoid(g_rho)*dg_drho*(y/rho)
#     deperp_dz = tf.math.softplus(g_rho)*tf.math.sigmoid(g_z)*dg_dz

#     return e_perp, deperp_dx, deperp_dy, deperp_dz


# Similar to getOrdinaryPermittivities but for the extraordinary case.
def getExtraordinaryPermittivities(positions, extraordinary_constants):

    # # 4th-degree polynomial as a function of x:
    # x = positions[:, 0] # for readability
    # g = extraordinary_constants[:, 0] + extraordinary_constants[:, 1]*x + extraordinary_constants[:, 2]*x**2 + extraordinary_constants[:, 3]*x**3 + extraordinary_constants[:, 4]*x**4 # g is an intermediate parameterization
    # e_para = 1 + tf.math.softplus(g)
    # depara_dx = tf.math.sigmoid(g)*(extraordinary_constants[:, 1] + 2.0*extraordinary_constants[:, 2]*x + 3.0*extraordinary_constants[:, 3]*x**2 + 4.0*extraordinary_constants[:, 4]*x**3)
    # depara_dy = tf.zeros(tf.shape(positions)[0])
    # depara_dz = tf.zeros(tf.shape(positions)[0])
    # return e_para, depara_dx, depara_dy, depara_dz

    # Constant value (given by the first constant):
    desired_shape = positions[:, 0]
    e_para = tf.ones_like(desired_shape)
    depara_dx = tf.zeros_like(desired_shape)
    depara_dy = tf.zeros_like(desired_shape)
    depara_dz = tf.zeros_like(desired_shape)

    return e_para, depara_dx, depara_dy, depara_dz

# Similar to getOrdinaryPermittivites, however there are 10 items returned:
# The director profile (size 3 tensor), as well as the nine spatial derivatives (three for each director component).
def getDirector(positions, director_constants):

    # # Constant x:
    # dx = tf.ones(tf.shape(positions)[0])
    # dy = tf.zeros(tf.shape(positions)[0])
    # dz = tf.zeros(tf.shape(positions)[0])
    # director = tf.stack([dx, dy, dz], axis=1)
    
    # ddx_x = tf.zeros(tf.shape(positions)[0])
    # ddx_y = tf.zeros(tf.shape(positions)[0])
    # ddx_z = tf.zeros(tf.shape(positions)[0])
    # ddy_x = tf.zeros(tf.shape(positions)[0])
    # ddy_y = tf.zeros(tf.shape(positions)[0])
    # ddy_z = tf.zeros(tf.shape(positions)[0])
    # ddz_x = tf.zeros(tf.shape(positions)[0])
    # ddz_y = tf.zeros(tf.shape(positions)[0])
    # ddz_z = tf.zeros(tf.shape(positions)[0])

    # Constant z:
    desired_shape = positions[:, 0]
    dx = tf.zeros_like(desired_shape)
    dy = tf.zeros_like(desired_shape)
    dz = tf.ones_like(desired_shape)
    director = tf.stack([dx, dy, dz], axis=1)
    
    ddx_x = tf.zeros_like(desired_shape)
    ddx_y = tf.zeros_like(desired_shape)
    ddx_z = tf.zeros_like(desired_shape)
    ddy_x = tf.zeros_like(desired_shape)
    ddy_y = tf.zeros_like(desired_shape)
    ddy_z = tf.zeros_like(desired_shape)
    ddz_x = tf.zeros_like(desired_shape)
    ddz_y = tf.zeros_like(desired_shape)
    ddz_z = tf.zeros_like(desired_shape)

    return director, ddx_x, ddx_y, ddx_z, ddy_x, ddy_y, ddy_z, ddz_x, ddz_y, ddz_z

# This function takes in a tensor of wave vectors and directors and returns a tensor of the associated Electric field (unit) vectors.
# Note that the associated phase of the electric field vectors is updating during ray propagation (rayPropagation() function)   
def getEfield(e_perp, e_para, wave_normals, directors, ordinary):
    p = wave_normals # for readability
    d = directors # for readability

    E_ord = tf.linalg.cross(p, d)/tf.norm(tf.linalg.cross(p, d), axis=1, keepdims=True)
       
    grad_p_He_x = 2*e_perp*p[:, 0] + 2*(e_para - e_perp)*tf.reduce_sum(p*d, axis=1)*d[:, 0] 
    grad_p_He_y = 2*e_perp*p[:, 1] + 2*(e_para - e_perp)*tf.reduce_sum(p*d, axis=1)*d[:, 1]
    grad_p_He_z = 2*e_perp*p[:, 2] + 2*(e_para - e_perp)*tf.reduce_sum(p*d, axis=1)*d[:, 2]

    grad_p_He = tf.stack([grad_p_He_x, grad_p_He_y, grad_p_He_z], axis=1)

    E_extraord = tf.linalg.cross(tf.linalg.cross(p, d), grad_p_He)/tf.norm(tf.linalg.cross(tf.linalg.cross(p, d), grad_p_He), axis=1, keepdims=True)

    # Choose either the ordinary or extraordinary E-field vector depending on the values of the ordinary tensor:
    ordinary_expanded = tf.expand_dims(ordinary, axis=1) # Turns ordinary from size N to size (N, 1)
    E = tf.where(ordinary_expanded, E_ord, E_extraord)

    return E

# Given a tensor of positions and a boundingBox geometry, this function returns an "alive"
# tensor of booleans which determines whether each ray should continue to be traced.
# Before running the code, make sure to uncomment the block of code based on the boundingBox geometry type.
def checkBoundary(positions, boundingBox, current_alive):
    
    # # Cylindrical:
    r = tf.math.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    z = positions[:, 2]
    alive = tf.logical_and(tf.logical_and(r<=boundingBox[1], r>=boundingBox[0]), tf.logical_and(z<=boundingBox[3], z>=boundingBox[2]))
    alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
    return alive
 
    # Spherical:
    # r = tf.math.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    # alive = r<=boundingBox[4]
    # alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
    # return alive

    # Rectangular Prism Slabs:
    # x = positions[:, 0]
    # y = positions[:, 1]
    # z = positions[:, 2]
    # alive = tf.logical_and(tf.logical_and(x>=boundingBox[0], x<=boundingBox[1]), tf.logical_and(tf.logical_and(y>=boundingBox[2], y<=boundingBox[3]) , tf.logical_and(z>=boundingBox[4], z<=boundingBox[5])))
    # alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
    # return alive


# Given a tensor of positions and a tensor of material IDs (in the forms of integers), this function
# returns a tensor of booleans. True <==> New Material reached, False <==> same material.
def checkForHit(positions, material_IDs, geometry_vectors):
    new_material_IDs = getMaterialsAtCoordinates(positions, geometry_vectors)
    hit = tf.not_equal(material_IDs, new_material_IDs)
    return hit

# Given a tensor of positions (Nx3) and a list of geometry vectors, this function returns a tensor of integers corresponding 
# to the material associated with each position.
# The type of geometry used must be MANUALLY UNCOMMENTED when switches materials.
# Only one type of material is supported in a simulation.
# See the "MaterialClass.py" file for how to properly defined geometry vectors for each type of geometry.
def getMaterialsAtCoordinates(positions, geometry_vectors):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Rectangular Prism Slabs:
    # x = tf.expand_dims(x, axis=1)
    # y = tf.expand_dims(y, axis=1)
    # z = tf.expand_dims(z, axis=1)

    # xmin = tf.expand_dims(geometry_vectors[:, 0], axis=0)
    # xmax = tf.expand_dims(geometry_vectors[:, 1], axis=0)
    # ymin = tf.expand_dims(geometry_vectors[:, 2], axis=0)
    # ymax = tf.expand_dims(geometry_vectors[:, 3], axis=0)
    # zmin = tf.expand_dims(geometry_vectors[:, 4], axis=0)
    # zmax = tf.expand_dims(geometry_vectors[:, 5], axis=0)

    # mask = tf.logical_and(tf.logical_and(x>=xmin, x<=xmax), tf.logical_and(tf.logical_and(y>=ymin, y<=ymax) , tf.logical_and(z>=zmin, z<=zmax)))
    # indices = tf.argmax(tf.cast(mask, tf.int32), axis=1)
    # indices = tf.cast(indices, dtype=tf.int32)
    # return indices

    # Cylinders:
    # Note - This only works for two cylinders. The first cylinder must be at the origin.
    r = tf.math.sqrt(x**2 + y**2)
    cond = tf.logical_and(tf.logical_and(r>=geometry_vectors[0][0], r<=geometry_vectors[0][1]), tf.logical_and(z>=geometry_vectors[0][2], z<=geometry_vectors[0][3]))
    cond = tf.cast(tf.logical_not(cond), dtype=tf.int32)
    return cond

    # Spheres:
    # r = tf.math.sqrt(x**2 + y**2 + z**2)
    # rmin = geometry_vectors[:, 3]
    # rmax = geometry_vectors[:, 4]

    # r = tf.expand_dims(r, axis=1)
    # rmin = tf.expand_dims(rmin, axis=0)
    # rmax = tf.expand_dims(rmax, axis=0)

    # mask = tf.logical_and(r>=rmin, r<=rmax)
    # indices = tf.argmax(tf.cast(mask, tf.int32), axis=1)
    # return indices

# This function takes in a tensors of current material_IDs and previous material_IDs and returns a tensor of surface normal unit vectors.
# This code works for rectangular slabs, concentric spheres, or "concentric" cylinders (all materials must be the same type!)    
def getSurfaceNormals(currMatIDs, prevMatIDs, geometry_vectors, positions):

    ray_current_geometry_vectors = tf.gather(geometry_vectors, currMatIDs)
    ray_previous_geometry_vectors = tf.gather(geometry_vectors, prevMatIDs)

# Rectangular Prism Slabs:
    # mask = ray_current_geometry_vectors[:, 5] >= ray_previous_geometry_vectors[:, 5]
    # mask = tf.expand_dims(mask, axis=1)
    # length_positions = tf.shape(positions)[0]
    # zhat = tf.constant([[0.0, 0.0, 1.0]])
    # zhat = tf.repeat(zhat, length_positions, axis=0)
    # minus_zhat = tf.constant([[0.0, 0.0, -1.0]])
    # minus_zhat = tf.repeat(minus_zhat, length_positions, axis=0)
    # surface_normal = tf.where(mask, zhat, minus_zhat) # If currentMat.zmax > prevMat.zmax, return [0,0,1], otherwise return [0,0,-1]
    # return surface_normal
    
# Spheres:
    # mask = ray_previous_geometry_vectors[:, 4] >= ray_current_geometry_vectors[:, 4]
    # mask = tf.expand_dims(mask, axis=1)
    # greater_than = positions/(-1.0*tf.expand_dims(tf.norm(positions, axis=1), axis=1))
    # less_than = positions/tf.expand_dims(tf.norm(positions, axis=1), axis=1)
    # surface_normal = tf.where(mask, greater_than, less_than)
    # return surface_normal
    
# Cylinders:
# Note - This only works for two cylinders. The first cylinder must be at the origin.
    mask1 = ray_previous_geometry_vectors[:, 1] >= ray_current_geometry_vectors[:, 1]
    mask1 = tf.expand_dims(mask1, axis=1)
    
    min_zdistance1 = tf.minimum(tf.math.abs(ray_current_geometry_vectors[:, 3] - positions[:, 2]), tf.math.abs(ray_current_geometry_vectors[:, 2] - positions[:, 2]))
    min_zdistance2 = tf.minimum(tf.math.abs(ray_previous_geometry_vectors[:, 3] - positions[:, 2]), tf.math.abs(ray_previous_geometry_vectors[:, 2] - positions[:, 2]))
    min_zdistance = tf.minimum(min_zdistance1, min_zdistance2)
    min_rdistance1 = ray_current_geometry_vectors[:, 1] - tf.math.sqrt(tf.square(positions[:, 0]) + tf.square(positions[:, 1]))
    min_rdistance2 = ray_previous_geometry_vectors[:, 1] - tf.math.sqrt(tf.square(positions[:, 0]) + tf.square(positions[:, 1]))
    min_rdistance = tf.minimum(min_rdistance1, min_rdistance2)

    mask2 = min_zdistance < min_rdistance
    mask2 = tf.expand_dims(mask2, axis=1)
    zhat = tf.stack([0.0*positions[:, 0], 0.0*positions[:, 1], positions[:, 2]], axis=1)
    zhat = zhat/tf.expand_dims(tf.math.abs(positions[:, 2]), axis=1)
    rhat = tf.stack([positions[:, 0], positions[:, 1], 0.0*positions[:, 2]], axis=1)
    rhat = rhat/tf.expand_dims(tf.norm(rhat, axis=1), axis=1)

    unit_vector = tf.where(mask2, zhat, rhat)
    surface_normal = tf.where(mask1, -1.0*unit_vector, unit_vector)
    return surface_normal

### ----------------------------- Interface Analysis Functions ------------------------------------------- ###

# Given a tensor input of surface normal vectors, ordinary indices in each medium, incident wave vectors, and incident E-field vectors,
# this function returns tensors corresponding to the reflected and transmitted wave vectors, E-field vectors, and Poynting vector magnitudes.
def Isotropic_Isotropic(surface_normal, no1, no2, p, E_i):

    ### Section 1: Calculation of Wave Normals. ###

    no1 = tf.expand_dims(no1, axis=1)
    no2 = tf.expand_dims(no2, axis=1)

    # First, we need to calculate the incident wave normal and its component tangential to the boundary.

    p_i = p
    p_tn = p_i - tf.reduce_sum(p_i*surface_normal, axis=1, keepdims=True)*surface_normal

    # Next, we calculate the reflected wave normal:
    p_r = p_tn - (tf.math.sqrt(no1**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*surface_normal

    # We now need to calculate the wave normal associated with the transmitted wave in medium 2.
    p_t = p_tn + (tf.math.sqrt(no2**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*surface_normal

    ### Section 2: Calculation of Electric and Magnetic Polarization Vectors. ###

    E_ts = tf.linalg.cross(p_t, surface_normal)/tf.expand_dims(tf.norm(tf.linalg.cross(p_t, surface_normal), axis=1), axis=1)
    E_tp = tf.linalg.cross(E_ts, p_t)/tf.expand_dims(tf.norm(tf.linalg.cross(E_ts, p_t), axis=1), axis=1)
    E_rs = tf.linalg.cross(p_r, surface_normal)/tf.expand_dims(tf.norm(tf.linalg.cross(p_r, surface_normal), axis=1), axis=1)
    E_rp = tf.linalg.cross(E_rs, p_r)/tf.expand_dims(tf.norm(tf.linalg.cross(E_rs, p_r), axis=1), axis=1)

    H_i = tf.linalg.cross(p_i, E_i)
    H_ts = tf.linalg.cross(p_t, E_ts)
    H_tp = tf.linalg.cross(p_t, E_tp)
    H_rs = tf.linalg.cross(p_r, E_rs)
    H_rp = tf.linalg.cross(p_r, E_rp)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = tf.linalg.cross(p_i, surface_normal)
    t_p = tf.linalg.cross(surface_normal, t_s)

    ### Section 3: Calculation of the Fresenel coefficients. ###

    rows = [
    tf.stack([tf.reduce_sum(t_s*E_ts, axis=1), tf.reduce_sum(t_s*E_tp, axis=1), tf.reduce_sum(-1.0*t_s*E_rs, axis=1), tf.reduce_sum(-1.0*t_s*E_rp, axis=1)], axis=0),
    tf.stack([tf.reduce_sum(t_p*E_ts, axis=1), tf.reduce_sum(t_p*E_tp, axis=1), tf.reduce_sum(-1.0*t_p*E_rs, axis=1), tf.reduce_sum(-1.0*t_p*E_rp, axis=1)], axis=0),
    tf.stack([tf.reduce_sum(t_s*H_ts, axis=1), tf.reduce_sum(t_s*H_tp, axis=1), tf.reduce_sum(-1.0*t_s*H_rs, axis=1), tf.reduce_sum(-1.0*t_s*H_rp, axis=1)], axis=0),
    tf.stack([tf.reduce_sum(t_p*H_ts, axis=1), tf.reduce_sum(t_p*H_tp, axis=1), tf.reduce_sum(-1.0*t_p*H_rs, axis=1), tf.reduce_sum(-1.0*t_p*H_rp, axis=1)], axis=0)
]

    A = tf.stack(rows, axis=0)

    b = tf.stack([[tf.reduce_sum(t_s*E_i, axis=1)], [tf.reduce_sum(t_p*E_i, axis=1)], [tf.reduce_sum(t_s*H_i, axis=1)], [tf.reduce_sum(t_p*H_i, axis=1)]], axis=0)

    A_batched = tf.transpose(A, perm=[2, 0, 1])
    b_batched = tf.transpose(b, perm=[2, 0, 1])

    fresnel_coefs = tf.linalg.solve(A_batched, b_batched)

    fresnel_coefs = tf.transpose(fresnel_coefs, perm=[1, 2, 0])
    fresnel_coefs = tf.squeeze(fresnel_coefs, axis=1)

    a_ts = fresnel_coefs[0, :]
    a_tp = fresnel_coefs[1, :]
    a_rs = fresnel_coefs[2, :]
    a_rp = fresnel_coefs[3, :]

    a_ts = tf.expand_dims(a_ts, axis=-1)
    a_tp = tf.expand_dims(a_tp, axis=-1)
    a_rs = tf.expand_dims(a_rs, axis=-1)
    a_rp = tf.expand_dims(a_rp, axis=-1)

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.linalg.cross(E_i, H_i)

    # Calculate E_t and S_t:
    E_t = a_ts*E_ts + a_tp*E_tp
    H_t = a_ts*H_ts + a_tp*H_tp
    S_t = 0.5*tf.linalg.cross(E_t, H_t)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    E_t = E_t/tf.expand_dims(tf.norm(E_t, axis=1), axis=1) 

    # Calculate E_r and S_r:
    E_r = a_rs*E_rs + a_rp*E_rp
    H_r = a_rs*H_rs + a_rp*H_rp
    S_r = 0.5*tf.linalg.cross(E_r, H_r)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    E_r = E_r/tf.expand_dims(tf.norm(E_r, axis=1), axis=1)

    # Poynting Vector Magnitudes:
    S_r = tf.norm(S_r, axis=1)
    S_t = tf.norm(S_t, axis=1)

    return p_r, p_t, E_r, E_t, S_r, S_t

# Given a tensor input of surface normal vectors, ordinary indices in each medium, the extraordinary indices in medium 2, 
# optic axes in medium 2, incident wave vectors, and incident E-field vectors, this function returns 
# tensors corresponding to the reflected and transmitted wave vectors, E-field vectors, and Poynting vector magnitudes.
def Isotropic_Anisotropic(surface_normal, optical_axis_2, no1, no2, ne2, p, E_i):

    # ### Section 1: Calculation of Wave Normals. ###

    no1 = tf.expand_dims(no1, axis=1)
    no2 = tf.expand_dims(no2, axis=1)
    ne2 = tf.expand_dims(ne2, axis=1)

    # First, we need to calculate the incident wave normal and its component tangential to the boundary.

    p_i = p
    p_tn = p_i - tf.reduce_sum(p_i*surface_normal, axis=1, keepdims=True)*surface_normal

    # Next, we calculate the reflected wave normal:
    p_r = p_tn - (tf.math.sqrt(no1**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*surface_normal

    # We now need to calculate the wave normals associated with the transmitted waves in medium 2.
    # However, we first need to transform the surface normal vector and p_tn into the principle coordinate system of medium 2.

    A2 = findRotationMatrix(optical_axis_2)
    surface_normal_p2 = tf.matmul(A2, tf.expand_dims(surface_normal, axis=-1))
    surface_normal_p2 = tf.squeeze(surface_normal_p2, axis=-1)
    p_tn_p2 = tf.matmul(A2, tf.expand_dims(p_tn, axis=-1))
    p_tn_p2 = tf.squeeze(p_tn_p2, axis=-1)

    # Now, we can use the formulas [(43)-(45) from the paper] to calculate the transmitted wave normals, remembering that we need to
    # apply the '+' sign in these equations.
    p_to_p2, p_te_p2 = findTransmittedNormals(no2, ne2, p_tn_p2, surface_normal_p2)

    p_to_p2 = tf.expand_dims(p_to_p2, axis=-1)
    p_te_p2 = tf.expand_dims(p_te_p2, axis=-1)

    # The transmitted wave normals are then transformed back to the original coordinate system:
    p_to = tf.linalg.solve(A2, p_to_p2)
    p_te = tf.linalg.solve(A2, p_te_p2)

    p_to = tf.squeeze(p_to, axis=-1)
    p_te = tf.squeeze(p_te, axis=-1)

    ### Section 2. Calculation of Electric and Magnetic Polarization Vectors. ###

    E_to = tf.linalg.cross(p_to, optical_axis_2)/tf.expand_dims(tf.norm(tf.linalg.cross(p_to, optical_axis_2), axis=1), axis=1)
    E_te = getEfield(tf.math.sqrt(tf.squeeze(no2, axis=-1)), tf.math.sqrt(tf.squeeze(ne2, axis=-1)), p_te, optical_axis_2, tf.zeros(tf.shape(no2)[0], dtype=tf.bool))
    E_rs = tf.linalg.cross(p_r, surface_normal)/tf.expand_dims(tf.norm(tf.linalg.cross(p_r, surface_normal), axis=1), axis=1)
    E_rp = tf.linalg.cross(E_rs, p_r)/tf.expand_dims(tf.norm(tf.linalg.cross(E_rs, p_r), axis=1), axis=1)

    H_i = tf.linalg.cross(p_i, E_i)
    H_to = tf.linalg.cross(p_to, E_to)
    H_te = tf.linalg.cross(p_te, E_te)
    H_rs = tf.linalg.cross(p_r, E_rs)
    H_rp = tf.linalg.cross(p_r, E_rp)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = tf.linalg.cross(p_i, surface_normal)
    t_p = tf.linalg.cross(surface_normal, t_s)

    ## Section 3: Calculation of the Fresenel coefficients.
    # This is achieved by solving equation (48) from the paper numerically.
    rows = [
        tf.stack([tf.reduce_sum(t_s*E_to, axis=1), tf.reduce_sum(t_s*E_te, axis=1), tf.reduce_sum(-1.0*t_s*E_rs, axis=1), tf.reduce_sum(-1.0*t_s*E_rp, axis=1)], axis=0),
        tf.stack([tf.reduce_sum(t_p*E_to, axis=1), tf.reduce_sum(t_p*E_te, axis=1), tf.reduce_sum(-1.0*t_p*E_rs, axis=1), tf.reduce_sum(-1.0*t_p*E_rp, axis=1)], axis=0),
        tf.stack([tf.reduce_sum(t_s*H_to, axis=1), tf.reduce_sum(t_s*H_te, axis=1), tf.reduce_sum(-1.0*t_s*H_rs, axis=1), tf.reduce_sum(-1.0*t_s*H_rp, axis=1)], axis=0),
        tf.stack([tf.reduce_sum(t_p*H_to, axis=1), tf.reduce_sum(t_p*H_te, axis=1), tf.reduce_sum(-1.0*t_p*H_rs, axis=1), tf.reduce_sum(-1.0*t_p*H_rp, axis=1)], axis=0)
    ]

    A = tf.stack(rows, axis=0)

    b = tf.stack([[tf.reduce_sum(t_s*E_i, axis=1)], [tf.reduce_sum(t_p*E_i, axis=1)], [tf.reduce_sum(t_s*H_i, axis=1)], [tf.reduce_sum(t_p*H_i, axis=1)]], axis=0)

    A_batched = tf.transpose(A, perm=[2, 0, 1])
    b_batched = tf.transpose(b, perm=[2, 0, 1])

    fresnel_coefs = tf.linalg.solve(A_batched, b_batched)

    fresnel_coefs = tf.transpose(fresnel_coefs, perm=[1, 2, 0])
    fresnel_coefs = tf.squeeze(fresnel_coefs, axis=1)

    a_to = fresnel_coefs[0, :]
    a_te = fresnel_coefs[1, :]
    a_rs = fresnel_coefs[2, :]
    a_rp = fresnel_coefs[3, :]

    a_to = tf.expand_dims(a_to, axis=-1)
    a_te = tf.expand_dims(a_te, axis=-1)
    a_rs = tf.expand_dims(a_rs, axis=-1)
    a_rp = tf.expand_dims(a_rp, axis=-1)

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.linalg.cross(E_i, H_i)

    # Calculate S_to and S_te:
    S_to = 0.5*tf.linalg.cross(a_to*E_to, a_to*H_to)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    S_te = 0.5*tf.linalg.cross(a_te*E_te, a_te*H_te)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)

    # Calculate E_r and S_r:
    E_r = a_rs*E_rs + a_rp*E_rp
    H_r = a_rs*H_rs + a_rp*H_rp
    S_r = 0.5*tf.linalg.cross(E_r, H_r)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    E_r = E_r/tf.expand_dims(tf.norm(E_r, axis=1), axis=1)

    # Poynting Vector Magnitudes:
    S_r = tf.norm(S_r, axis=1)
    S_to = tf.norm(S_to, axis=1)
    S_te = tf.norm(S_te, axis=1)

    return p_r, p_to, p_te, E_r, E_to, E_te, S_r, S_to, S_te

# Given a tensor input of surface normal vectors, ordinary indices in each medium, the extraordinary indices in medium 1, 
# optic axes in medium 1, incident wave vectors, and incident E-field vectors, this function returns 
# tensors corresponding to the reflected and transmitted wave vectors, E-field vectors, and Poynting vector magnitudes.
def Anisotropic_Isotropic(surface_normal, optical_axis_1, no1, ne1, no2, p, E_i):

    ### Section 1: Calculation of Wave Normals. ###

    no1 = tf.expand_dims(no1, axis=1)
    ne1 = tf.expand_dims(ne1, axis=1)
    no2 = tf.expand_dims(no2, axis=1)

    # First, we need to calculate the incident wave normal and its component tangential to the boundary.
    p_i = p
    p_tn = p_i - tf.reduce_sum(p_i*surface_normal, axis=1, keepdims=True)*surface_normal

    A1 = findRotationMatrix(optical_axis_1)
    surface_normal_p1 = tf.matmul(A1, tf.expand_dims(surface_normal, axis=-1))
    surface_normal_p1 = tf.squeeze(surface_normal_p1, axis=-1)
    p_tn_p1 = tf.matmul(A1, tf.expand_dims(p_tn, axis=-1))
    p_tn_p1 = tf.squeeze(p_tn_p1, axis=-1)

    # Next we calculate the reflected ordinary and extraordinary wave normals in the p.c.s of medium 1, which we will denote p_ro_p1 and p_re_p1:

    p_ro_p1, p_re_p1 = findReflectedNormals(no1, ne1, p_tn_p1, surface_normal_p1)

    p_ro_p1 = tf.expand_dims(p_ro_p1, axis=-1)
    p_re_p1 = tf.expand_dims(p_re_p1, axis=-1)

    # The vectors p_ie_p1, p_tn_p1, p_ro_p1, and p_re_p1 are then transformed back to the original coordinate system.
    # This is achieved by multiplication with the inverse of the rotation matrix A1:

    p_ro = tf.linalg.solve(A1, p_ro_p1)
    p_re = tf.linalg.solve(A1, p_re_p1)

    p_ro = tf.squeeze(p_ro, axis=-1)
    p_re = tf.squeeze(p_re, axis=-1)

    # Now, we calculated the transmitted wave normal in medium 2:
    p_t = p_tn + (tf.math.sqrt(no2**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*surface_normal

    ### Section 2: Calculation of Electric and Magnetic Polarization Vectors.

    E_ts = tf.linalg.cross(p_t, surface_normal)/tf.expand_dims(tf.norm(tf.linalg.cross(p_t, surface_normal), axis=1), axis=1)
    E_tp = tf.linalg.cross(E_ts, p_t)/tf.expand_dims(tf.norm(tf.linalg.cross(E_ts, p_t), axis=1), axis=1)
    E_ro = tf.linalg.cross(p_ro, optical_axis_1)/tf.expand_dims(tf.norm(tf.linalg.cross(p_ro, optical_axis_1), axis=1), axis=1)
    E_re = getEfield(tf.math.sqrt(tf.squeeze(no1, axis=-1)), tf.math.sqrt(tf.squeeze(ne1, axis=-1)), p_re, optical_axis_1, tf.zeros(tf.shape(no1)[0], dtype=tf.bool))

    H_i = tf.linalg.cross(p_i, E_i)
    H_ts = tf.linalg.cross(p_t, E_ts)
    H_tp = tf.linalg.cross(p_t, E_tp)
    H_ro = tf.linalg.cross(p_ro, E_ro)
    H_re = tf.linalg.cross(p_re, E_re)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = tf.linalg.cross(p_i, surface_normal)
    t_p = tf.linalg.cross(surface_normal, t_s)

    ### Section 3: Calculation of the complex Fresenel coefficients.
    # This is achieved by solving equation (48) from the paper numerically.

    rows = [
        tf.stack([tf.reduce_sum(t_s*E_ts, axis=1), tf.reduce_sum(t_s*E_tp, axis=1), tf.reduce_sum(-1.0*t_s*E_ro, axis=1), tf.reduce_sum(-1.0*t_s*E_re, axis=1)], axis=0),
        tf.stack([tf.reduce_sum(t_p*E_ts, axis=1), tf.reduce_sum(t_p*E_tp, axis=1), tf.reduce_sum(-1.0*t_p*E_ro, axis=1), tf.reduce_sum(-1.0*t_p*E_re, axis=1)], axis=0),
        tf.stack([tf.reduce_sum(t_s*H_ts, axis=1), tf.reduce_sum(t_s*H_tp, axis=1), tf.reduce_sum(-1.0*t_s*H_ro, axis=1), tf.reduce_sum(-1.0*t_s*H_re, axis=1)], axis=0),
        tf.stack([tf.reduce_sum(t_p*H_ts, axis=1), tf.reduce_sum(t_p*H_tp, axis=1), tf.reduce_sum(-1.0*t_p*H_ro, axis=1), tf.reduce_sum(-1.0*t_p*H_re, axis=1)], axis=0)
    ]

    A = tf.stack(rows, axis=0)

    b = tf.stack([[tf.reduce_sum(t_s*E_i, axis=1)], [tf.reduce_sum(t_p*E_i, axis=1)], [tf.reduce_sum(t_s*H_i, axis=1)], [tf.reduce_sum(t_p*H_i, axis=1)]], axis=0)

    A_batched = tf.transpose(A, perm=[2, 0, 1])
    b_batched = tf.transpose(b, perm=[2, 0, 1])

    fresnel_coefs = tf.linalg.solve(A_batched, b_batched)

    fresnel_coefs = tf.transpose(fresnel_coefs, perm=[1, 2, 0])
    fresnel_coefs = tf.squeeze(fresnel_coefs, axis=1)

    a_ts = fresnel_coefs[0, :]
    a_tp = fresnel_coefs[1, :]
    a_ro = fresnel_coefs[2, :]
    a_re = fresnel_coefs[3, :]

    a_ts = tf.expand_dims(a_ts, axis=-1)
    a_tp = tf.expand_dims(a_tp, axis=-1)
    a_ro = tf.expand_dims(a_ro, axis=-1)
    a_re = tf.expand_dims(a_re, axis=-1)

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.linalg.cross(E_i, H_i)

    # Calculate S_ro and S_re:
    S_ro = 0.5*tf.linalg.cross(a_ro*E_ro, a_ro*H_ro)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    S_re = 0.5*tf.linalg.cross(a_re*E_re, a_re*H_re)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)

    # Calculate E_t and S_t:
    E_t = a_ts*E_ts + a_tp*E_tp
    H_t = a_ts*H_ts + a_tp*H_tp
    S_t = 0.5*tf.linalg.cross(E_t, H_t)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    E_t = E_t/tf.expand_dims(tf.norm(E_t, axis=1), axis=1)

    # Poynting Vector Magnitudes:
    S_ro = tf.norm(S_ro, axis=1)
    S_re = tf.norm(S_re, axis=1)
    S_t = tf.norm(S_t, axis=1)
    
    return p_ro, p_re, p_t, E_ro, E_re, E_t, S_ro, S_re, S_t

# Given a tensor input of surface normal vectors, ordinary and extraordinary indices in each medium,
# optic axes in each medium, incident wave vectors, and incident E-field vectors, this function returns 
# tensors corresponding to the reflected and transmitted wave vectors, E-field vectors, and Poynting vector magnitudes.
def Anisotropic_Anisotropic(surface_normal, optical_axis_1, optical_axis_2, no1, ne1, no2, ne2, p, E_i):

    ### Section 1: Calculation of Wave Normals. ###

    no1 = tf.expand_dims(no1, axis=1)
    ne1 = tf.expand_dims(ne1, axis=1)
    no2 = tf.expand_dims(no2, axis=1)
    ne2 = tf.expand_dims(ne2, axis=1)

    # First, we need to calculate the incident wave normal and its component tangential to the boundary.
    p_i = p
    p_tn = p_i - tf.reduce_sum(p_i*surface_normal, axis=1, keepdims=True)*surface_normal

    A1 = findRotationMatrix(optical_axis_1)
    surface_normal_p1 = tf.matmul(A1, tf.expand_dims(surface_normal, axis=-1))
    surface_normal_p1 = tf.squeeze(surface_normal_p1, axis=-1)
    p_tn_p1 = tf.matmul(A1, tf.expand_dims(p_tn, axis=-1))
    p_tn_p1 = tf.squeeze(p_tn_p1, axis=-1)

    # Next we calculate the reflected ordinary and extraordinary wave normals in the p.c.s of medium 1, which we will denote p_ro_p1 and p_re_p1:

    p_ro_p1, p_re_p1 = findReflectedNormals(no1, ne1, p_tn_p1, surface_normal_p1)

    p_ro_p1 = tf.expand_dims(p_ro_p1, axis=-1)
    p_re_p1 = tf.expand_dims(p_re_p1, axis=-1)

    # The vectors p_ie_p1, p_tn_p1, p_ro_p1, and p_re_p1 are then transformed back to the original coordinate system.
    # This is achieved by multiplication with the inverse of the rotation matrix A1:

    p_ro = tf.linalg.solve(A1, p_ro_p1)
    p_re = tf.linalg.solve(A1, p_re_p1)

    p_ro = tf.squeeze(p_ro, axis=-1)
    p_re = tf.squeeze(p_re, axis=-1)

    # We now need to calculate the wave normals associated with the transmitted waves in medium 2.
    # However, we first need to transform the surface normal vector and p_tn into the principle coordinate system of medium 2.

    A2 = findRotationMatrix(optical_axis_2)
    surface_normal_p2 = tf.matmul(A2, tf.expand_dims(surface_normal, axis=-1))
    surface_normal_p2 = tf.squeeze(surface_normal_p2, axis=-1)
    p_tn_p2 = tf.matmul(A2, tf.expand_dims(p_tn, axis=-1))
    p_tn_p2 = tf.squeeze(p_tn_p2, axis=-1)

    # Now, we can use the formulas [(43)-(45) from the paper] to calculate the transmitted wave normals, remembering that we need to
    # apply the '+' sign in these equations.
    p_to_p2, p_te_p2 = findTransmittedNormals(no2, ne2, p_tn_p2, surface_normal_p2)

    p_to_p2 = tf.expand_dims(p_to_p2, axis=-1)
    p_te_p2 = tf.expand_dims(p_te_p2, axis=-1)

    # The transmitted wave normals are then transformed back to the original coordinate system:
    p_to = tf.linalg.solve(A2, p_to_p2)
    p_te = tf.linalg.solve(A2, p_te_p2)

    p_to = tf.squeeze(p_to, axis=-1)
    p_te = tf.squeeze(p_te, axis=-1)

    ### Section 2. Calculation of Electric and Magnetic Polarization Vectors. ###

    E_to = tf.linalg.cross(p_to, optical_axis_2)/tf.expand_dims(tf.norm(tf.linalg.cross(p_to, optical_axis_2), axis=1), axis=1)
    E_te = getEfield(tf.math.sqrt(tf.squeeze(no2, axis=-1)), tf.math.sqrt(tf.squeeze(ne2, axis=-1)), p_te, optical_axis_2, tf.zeros(tf.shape(no2)[0], dtype=tf.bool))
    E_ro = tf.linalg.cross(p_ro, optical_axis_1)/tf.expand_dims(tf.norm(tf.linalg.cross(p_ro, optical_axis_1), axis=1), axis=1)
    E_re = getEfield(tf.math.sqrt(tf.squeeze(no1, axis=-1)), tf.math.sqrt(tf.squeeze(ne1, axis=-1)), p_re, optical_axis_1, tf.zeros(tf.shape(no1)[0], dtype=tf.bool))

    H_i = tf.linalg.cross(p_i, E_i)
    H_to = tf.linalg.cross(p_to, E_to)
    H_te = tf.linalg.cross(p_te, E_te)
    H_ro = tf.linalg.cross(p_ro, E_ro)
    H_re = tf.linalg.cross(p_re, E_re)

    # Next, we calculate the two orthogonal vectors to the interface, denoted t_s and t_p:
    t_s = tf.linalg.cross(p_i, surface_normal)
    t_p = tf.linalg.cross(surface_normal, t_s)

    ### Section 3: Calculation of the Fresenel coefficients.
    # This is achieved by solving equation (48) from the paper numerically.
    rows = [
            tf.stack([tf.reduce_sum(t_s*E_to, axis=1), tf.reduce_sum(t_s*E_te, axis=1), tf.reduce_sum(-1.0*t_s*E_ro, axis=1), tf.reduce_sum(-1.0*t_s*E_re, axis=1)], axis=0),
            tf.stack([tf.reduce_sum(t_p*E_to, axis=1), tf.reduce_sum(t_p*E_te, axis=1), tf.reduce_sum(-1.0*t_p*E_ro, axis=1), tf.reduce_sum(-1.0*t_p*E_re, axis=1)], axis=0),
            tf.stack([tf.reduce_sum(t_s*H_to, axis=1), tf.reduce_sum(t_s*H_te, axis=1), tf.reduce_sum(-1.0*t_s*H_ro, axis=1), tf.reduce_sum(-1.0*t_s*H_re, axis=1)], axis=0),
            tf.stack([tf.reduce_sum(t_p*H_to, axis=1), tf.reduce_sum(t_p*H_te, axis=1), tf.reduce_sum(-1.0*t_p*H_ro, axis=1), tf.reduce_sum(-1.0*t_p*H_re, axis=1)], axis=0)
        ]

    A = tf.stack(rows, axis=0)

    b = tf.stack([[tf.reduce_sum(t_s*E_i, axis=1)], [tf.reduce_sum(t_p*E_i, axis=1)], [tf.reduce_sum(t_s*H_i, axis=1)], [tf.reduce_sum(t_p*H_i, axis=1)]], axis=0)

    A_batched = tf.transpose(A, perm=[2, 0, 1])
    b_batched = tf.transpose(b, perm=[2, 0, 1])

    fresnel_coefs = tf.linalg.solve(A_batched, b_batched)

    fresnel_coefs = tf.transpose(fresnel_coefs, perm=[1, 2, 0])
    fresnel_coefs = tf.squeeze(fresnel_coefs, axis=1)

    a_to = fresnel_coefs[0, :]
    a_te = fresnel_coefs[1, :]
    a_ro = fresnel_coefs[2, :]
    a_re = fresnel_coefs[3, :]

    a_to = tf.expand_dims(a_to, axis=-1)
    a_te = tf.expand_dims(a_te, axis=-1)
    a_ro = tf.expand_dims(a_ro, axis=-1)
    a_re = tf.expand_dims(a_re, axis=-1)

    ### Section 4: Calculation of the time-average Poynting Vectors.
    # Note that all magnitude are expressed relative to the magnitude of the incoming Poynting Vector.
    S_i_unormalized = 0.5*tf.linalg.cross(E_i, H_i)

    # Calculate S_ro and S_re:
    S_ro = 0.5*tf.linalg.cross(a_ro*E_ro, a_ro*H_ro)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    S_re = 0.5*tf.linalg.cross(a_re*E_re, a_re*H_re)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)

    # Calculate S_to and S_te:
    S_to = 0.5*tf.linalg.cross(a_to*E_to, a_to*H_to)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)
    S_te = 0.5*tf.linalg.cross(a_te*E_te, a_te*H_te)/tf.expand_dims(tf.norm(S_i_unormalized, axis=1), axis=1)

    # Poynting Vector Magnitudes:
    S_ro = tf.norm(S_ro, axis=1)
    S_re = tf.norm(S_re, axis=1)
    S_to = tf.norm(S_to, axis=1)
    S_te = tf.norm(S_te, axis=1)

    return p_ro, p_re, p_to, p_te, E_ro, E_re, E_to, E_te, S_ro, S_re, S_to, S_te

# This function calculates and returns a tensor of 3x3 rotation matrices A needed to rotate the optic axes in the principle coordinate systems.
# In other words, it finds the matrices A that satifies A*o = <0,0,1>. The input o must be a tensors of 3x1 unit vectors, i.e. size (N, 3).
# Note that the output rotation matrix tensor is of size (N, 3, 3).
def findRotationMatrix(o):
    length_o = tf.shape(o)[0]
    zeros = tf.zeros(length_o)
    ones = 1 + zeros

    phi1 = tf.atan2(o[:, 1], o[:, 0]) # Angle of o prcted onto the xy plane w.r.t the x-axis

    row1 = tf.stack([tf.cos(-phi1), -tf.sin(-phi1), zeros])
    row2 = tf.stack([tf.sin(-phi1), tf.cos(-phi1), zeros])
    row3 = tf.stack([zeros, zeros, ones])

    Az = tf.stack([row1, row2, row3]) # Rotation matrix to rotate o to the xz plane (Rotation of -phi1 about z-axis)

    Az = tf.transpose(Az, perm=[2, 0, 1])
    o = tf.expand_dims(o, axis=-1)
    # o = tf.transpose(o, perm=[1, 2, 0])

    oxz = tf.matmul(Az, o) # Multiplying by the rotation matrix A
    oxz = tf.squeeze(oxz, axis=-1)

    phi2 = tf.atan2(oxz[:, 0], oxz[:, 2]) # Angle of oxz, moving from the z-axis towards the x-axis

    row1 = tf.stack([tf.cos(-phi2), zeros, tf.sin(-phi2)])
    row2 = tf.stack([zeros, ones, zeros])
    row3 = tf.stack([-tf.sin(-phi2), zeros, tf.cos(phi2)])

    Ay = tf.stack([row1, row2, row3])
    Ay = tf.transpose(Ay, perm=[2, 0, 1])
    R_matrix = tf.matmul(Ay, Az) # The product of the two matrices is the overall 3x3 Rotation matrix

    return R_matrix

# Given no and ne of medium 1, the tangential component of the wave normal in the p.c.s, and the surface normal vector in the p.c.s.,
# this function calculates the ordinary and extraordinary reflected wave normals (pro and pre).
def findReflectedNormals(no, ne, p_tn, n):

    po = p_tn - (tf.math.sqrt(no**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*n

    no = tf.squeeze(no, axis=-1)
    ne = tf.squeeze(ne, axis=-1)

    A = (n[:, 2]**2)/no**2 + (n[:, 0]**2 + n[:, 1]**2)/ne**2
    B = 2*p_tn[:, 2]*n[:, 2]/no**2 + (2*p_tn[:, 0]*n[:, 0] + 2*p_tn[:, 1]*n[:, 1])/ne**2
    C = p_tn[:, 2]**2/no**2 + (p_tn[:, 0]**2 + p_tn[:, 1]**2)/ne**2 - 1
    discriminant = B**2 - 4*A*C
    xi = (-1.0*B - tf.math.sqrt(discriminant))/(2*A)
    xi = tf.expand_dims(xi, axis=-1)
    pe = p_tn + xi*n

    return po, pe

# Given no and ne of medium 2, the tangential component of the wave normal in the p.c.s of medium 2, and the surface normal vector in the p.c.s. of medium 2,
# this function calculates the ordinary and extraordinary transmitted wave normals (pto and pte).
def findTransmittedNormals(no, ne, p_tn, n):
    
    po = p_tn + (tf.math.sqrt(no**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*n

    no = tf.squeeze(no, axis=-1)
    ne = tf.squeeze(ne, axis=-1)

    A = (n[:, 2]**2)/no**2 + (n[:, 0]**2 + n[:, 1]**2)/ne**2
    B = 2*p_tn[:, 2]*n[:, 2]/no**2 + (2*p_tn[:, 0]*n[:, 0] + 2*p_tn[:, 1]*n[:, 1])/ne**2
    C = p_tn[:, 2]**2/no**2 + (p_tn[:, 0]**2 + p_tn[:, 1]**2)/ne**2 - 1
    discriminant = B**2 - 4*A*C
    xi = (-1.0*B + tf.math.sqrt(discriminant))/(2*A)
    xi = tf.expand_dims(xi, axis=-1)
    pe = p_tn + xi*n

    return po, pe

# This prevents the value inside the sqrt from being < epsilon.
# It also prevents the gradient from exploding near zero.
# This is used in teh "findTransmittedNormals" function to handle total internal reflection.
def safe_sqrt(x, epsilon=1e-7):
    return tf.math.sqrt(tf.maximum(x, epsilon))

# Given a starting and ending x position, a fixed z_position, a number of rays, an incident angle (in degrees), and an incoming polarization, 
# this function generates all the ray tensors that will be added to the global ray tensors in the main algorithm:

def createStartingRays(NumberOfRays, starting_x, ending_x, fixed_z, angle, Epol, mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts):
    angle = angle*(.0174532925) # Convert the angle from degrees to radians

    # Offset the initial position of the rays such that they INTERSECT the lens at the given starting_x and ending_x:
    # THIS IS FOR THE SPECIFIC 1D RECTANGULAR SLAB LENS EXAMPLE:
    starting_x = starting_x - (0.05 - fixed_z)*tf.math.tan(angle)
    ending_x = ending_x - (0.05 - fixed_z)*tf.math.tan(angle)

    positions = tf.linspace([starting_x, 0.0, fixed_z], [ending_x, 0.0, fixed_z], NumberOfRays)
    px = tf.math.sin(angle)
    py = 0.0
    pz = tf.math.cos(angle)
    wave_vector = tf.stack([px, py, pz])
    wave_vector = tf.expand_dims(wave_vector, axis=0)
    wave_vectors = tf.tile(wave_vector, [NumberOfRays, 1])
    PoyntingMag = tf.ones([NumberOfRays], dtype=tf.float32)
    alive = tf.ones([NumberOfRays], dtype=tf.bool)
    Epol = tf.expand_dims(Epol, axis=0)
    Efields = tf.tile(Epol, [NumberOfRays, 1])
    ordinary = tf.ones([NumberOfRays], dtype=tf.bool)
    material_IDs = tf.zeros([NumberOfRays], dtype=tf.int32)

    ray_ordinary_consts = tf.expand_dims(mat_ordinary_consts[0], axis=0)
    ray_extraordinary_consts = tf.expand_dims(mat_extraordinary_consts[0], axis=0)
    ray_director_consts = tf.expand_dims(mat_director_consts[0], axis=0)

    ray_ordinary_consts = tf.tile(ray_ordinary_consts, [NumberOfRays, 1])
    ray_extraordinary_consts = tf.tile(ray_extraordinary_consts, [NumberOfRays, 1])
    ray_director_consts = tf.tile(ray_director_consts, [NumberOfRays, 1])

    return positions, wave_vectors, PoyntingMag, alive, Efields, ordinary, material_IDs, ray_ordinary_consts, ray_extraordinary_consts, ray_director_consts

# This function a tensor of 2D rectangular feed locations with equal spacing along x and y.
# Inputs: center_location (x,y,z), dx (x spacing), dy (y spacing), Nx (# elements along x), Ny (# elements along y).
# Outputs: An (N,3) tensor containing all feed locations in cartesian coords.
# IMPORTANT Note: This function assumes Nx and Ny are odd, such that there is always a feed located at the center location.
def getRectFeedPoints(center_location, dx, dy, Nx, Ny):
    # Generate linearly spaced vectors along x and y:
    x = tf.linspace(center_location[0] - 0.5*(Nx-1)*dx, center_location[0] + 0.5*(Nx-1)*dx, Nx)
    y = tf.linspace(center_location[1] - 0.5*(Ny-1)*dy, center_location[1] + 0.5*(Ny-1)*dy, Ny)
    # Create the meshgrid:
    [X, Y] = tf.meshgrid(x, y, indexing='ij')
    # Reshape X and Y to vectors:
    X = tf.reshape(X, [-1])
    Y = tf.reshape(Y, [-1])
    # Z is constant for all ring points:
    Z = tf.fill(tf.size(X), center_location[2])
    # Concatenate together:
    positions = tf.stack([X, Y, Z])
    return tf.transpose(positions)

# This function generates a tensor of 2D cylindrical feed locations with equal radial spacing and equal angular spacing
# in the azimuthal directions.
# Inputs: center_location (x,y,z), dr (radial spacing), dphi (azimuthal spacing), Nrings (number of rings not including center point), fixedZ.
# Output: An (N,3) tensor containing all feed locations in cartesian coords.

def getCircFeedPoints(center_location, dr, dphi, Nrings):
    # Setup constants
    dphi_rad = dphi*0.0174532925
    
    # 1. Generate Radii: [dr, 2dr, ..., Nrings*dr]
    # Shape: (Nrings,)
    radii = tf.range(1, Nrings + 1, dtype=tf.float32) * dr
    
    # 2. Generate Angles: [0, dphi, 2dphi, ...] up to 2pi
    # Shape: (Nphi,)
    num_phi = int(2 * 3.141592653589793 / dphi_rad)
    angles = tf.range(num_phi, dtype=tf.float32) * dphi_rad
    
    # 3. Create Meshgrid for Rings (Broadcasting)
    # R: (Nrings, Nphi), PHI: (Nrings, Nphi)
    R, PHI = tf.meshgrid(radii, angles, indexing='ij')
    
    # 4. Convert to Cartesian
    x = tf.reshape(R * tf.cos(PHI), [-1])
    y = tf.reshape(R * tf.sin(PHI), [-1])
    # Z is constant for all ring points
    z = tf.fill(tf.shape(x), center_location[2])
    
    ring_points = tf.stack([x + center_location[0], 
                            y + center_location[1], 
                            z], axis=1)
    
    # 5. Concatenate with center point
    center_pt = tf.expand_dims(center_location, 0)
    full_tensor = tf.concat([center_pt, ring_points], axis=0)
    
    return full_tensor

# This function returns a list of target trajectories given a set of initial feed locations
# created using the function getCircFeedPoints(), and a given spacing in theta.
def getTargetTrajects(center_location, dr, dphi, Nrings, theta_spacing_deg):
    
    theta = tf.range(1, Nrings + 1, dtype=tf.float32) * theta_spacing_deg
    theta_angles = (m.pi/180)*theta
    
    dphi_rad = (m.pi/180)*dphi
    num_phi = int(2 * 3.141592653589793 / dphi_rad)
    phi_angles = tf.range(num_phi, dtype=tf.float32) * dphi_rad
    phi_angles = phi_angles + m.pi # Target phi angles are 180 degrees w.r.t. feed phi angles

    THETA, PHI = tf.meshgrid(theta_angles, phi_angles, indexing='ij')

    # Convert each theta and phi to a target wavevector in Cartesian:
    x = tf.reshape(tf.sin(THETA)*tf.cos(PHI), [-1])
    y = tf.reshape(tf.sin(THETA)*tf.sin(PHI), [-1])
    z = tf.reshape(tf.cos(THETA), [-1])

    p_targs = tf.stack([x, y, z], axis=1)

    # Concatenate with center point p_targ = [0.00000001, 0.000000001, 1.0]
    p_targ_center = tf.expand_dims([0.00000001, 0.00000001, 1.0], axis=0)

    p_targs = tf.concat([p_targ_center, p_targs], axis=0)

    # Make p's unit vectors (since the rays will be propagating in air):
    p_targs = p_targs/tf.expand_dims(tf.norm(p_targs, axis=1), axis=1)

    return p_targs


# Given a number of rays to create, the maximum theta angle, the position of the sphere center, and the polarization
# of rays, this function returns all the tensors needed to intialize the spherical cap of rays.
# UPDATE: Each input is now a tensor corrponding to the variable value for each feed location.
def createIsotropicRays(NumberOfRays, theta_max_deg, theta_target_deg, sphere_center, Epol, ordinary_consts, extraordinary_consts, director_consts):

    ### Step 1 - Calculate the wave vectors:
    N = NumberOfRays # Number of rays
    N = tf.cast(N, dtype=tf.float32)

    # Find number of feeds from length of theta_max_deg:
    numFeeds = tf.shape(theta_max_deg)[0]
    
    # Convert theta_max, theta_target_deg to radians:
    theta_max = theta_max_deg*.0174532925
    theta_target = theta_target_deg*.0174532925

    # Calculate a minimum z value based on theta_max:
    z_min = tf.cos(theta_max)

    # The "golden angle":
    golden_angle = 2.39998

    # i = tf.range(N) # i = [0, 1, ..., N-1]
    i = tf.range(N[0])
    i = tf.expand_dims(i, axis=0)
    i = tf.tile(i, [numFeeds, 1])

    u = (i + 0.5)/tf.expand_dims(N, axis=1)

    # Sample uniformly between z_min and 1:
    z_min = tf.expand_dims(z_min, axis=1)
    z = z_min + (1.0 - z_min) * u

    r = tf.sqrt(1.0 - z*z)
    phi = i*golden_angle

    x = r*tf.cos(phi)
    y = r*tf.sin(phi)

    # Calculate a list of wave vectors of the incident rays:
    wave_vectors = tf.stack([x, y, z], axis=2)
    wave_vectors = tf.reshape(wave_vectors, [-1, 3])

    # Next, we rotate the wave_vectors based on the desired theta angle (Feature is currently deprecated):
    # target_direction = [tf.math.sin(theta_target), 0.0, tf.math.cos(theta_target)]
    # target_direction = tf.expand_dims(target_direction, axis=0)
    # target_direction = tf.tile(target_direction, [NumberOfRays, 1])

    # rotation_matrix = findRotationMatrix(target_direction)
    # wave_vectors = tf.linalg.matmul(tf.linalg.inv(rotation_matrix),tf.expand_dims(wave_vectors, axis=-1))
    # wave_vectors = tf.squeeze(wave_vectors, axis=-1)

    ### Step 2 - Calculate the electric polarization vector (either phi or theta polarized):
    # Ephi = tf.stack([-1.0*tf.sin(phi), tf.cos(phi), tf.zeros_like(phi)]/tf.linalg.norm([-1.0*tf.sin(phi), tf.cos(phi), tf.zeros_like(phi)]), axis=1)
    # Etheta = tf.linalg.cross(Ephi, wave_vectors)/tf.expand_dims(tf.linalg.norm(tf.linalg.cross(Ephi, wave_vectors), axis=1), axis=1)

    # If Epol = 0 use Ephi, if Epol = 1 use Etheta:
    # Efields = (Epol - 1.0)*Ephi + Epol*Etheta

    ### Constant Epol model:
    Efields  = tf.expand_dims(Epol, axis=1)
    Efields = tf.tile(Efields, [1, NumberOfRays[0], 1])
    Efields = tf.reshape(Efields, [-1, 3])

    ### Step 3 - Calculate all other quantities:
    sphere_center = tf.expand_dims(sphere_center, axis=1)
    positions = tf.tile(sphere_center, [1, NumberOfRays[0], 1])
    positions = tf.reshape(positions, [-1, 3])

    totalNumRays = tf.cast(NumberOfRays[0], tf.int32)*numFeeds
    PoyntingMag = tf.ones([totalNumRays], dtype=tf.float32)
    alive = tf.ones([totalNumRays], dtype=tf.bool)
    ordinary = tf.ones([totalNumRays], dtype=tf.bool)
    material_IDs = tf.ones([totalNumRays], dtype=tf.int32)

    ray_ordinary_consts = tf.transpose(tf.tile(tf.expand_dims(ordinary_consts[1,:], axis=1), multiples=[1,totalNumRays]))
    ray_extraordinary_consts = tf.transpose(tf.tile(tf.expand_dims(extraordinary_consts[1,:], axis=1), multiples=[1,totalNumRays]))
    ray_director_consts = tf.transpose(tf.tile(tf.expand_dims(director_consts[1,:], axis=1), multiples=[1,totalNumRays]))

    return positions, wave_vectors, PoyntingMag, alive, Efields, ordinary, material_IDs, ray_ordinary_consts, ray_extraordinary_consts, ray_director_consts

# Given a number of FS coefficients, the period (centered about zero), and constants describing the initial guess,
# this function calculates and returns an equivalent fourier series representation of the er distribution.
def generateFourierCoefs(N, T0, consts):

    # Using luneburg-type distribution for initial guess:
    x = tf.linspace(-T0/2, T0/2, 1000)
    y = consts[0]*(1 - consts[2]*(x/consts[1])**2)

    # Compute FS coefficients based on x and y:
    w0 = 2*m.pi/T0

    a0 = (1/T0) * tfp.math.trapz(y, x)

    # Vector of n's:
    n_vec = tf.range(1, N+1, dtype=tf.float32)[:, tf.newaxis]

    # Compute all trig terms at once:
    cos_matrix = tf.cos(n_vec * w0 * x)
    # sin_matrix = tf.sin(n_vec * w0 * x) # Don't need since we want an even function

    # Integrands:
    integrand_a = y*cos_matrix
    # integrand_b = y*sin_matrix # Don't need since we want an even function

    # Tile x to match size of integrands:
    x_matrix = tf.tile(x[tf.newaxis, :], [N, 1])

    # Multiply by y and integrate along the x-axis (axis=1)
    an = (2/T0) * tfp.math.trapz(integrand_a, x=x_matrix, axis=1)
    # bn = (2/T0) * tfp.math.trapz(integrand_b, x=x_matrix, axis=1) # Don't need since we want an even function
    
    # Combine the four variables into a single vector of length 2 + N:
    materialCoefs = tf.concat([[T0], [a0], an], axis=0)

    return materialCoefs

### ----------------------------- Everything below this line may need to be rewritten -------------------------------------------------------- ###

# This functions takes in a tensor of ray positions and a target focal point, and returns a size N
# tensor where for each ray the element represents the minimum distance SQUARED to the specified focal point.
# This function is currently not being used.
# def minDistanceToPoint(positions, focal_point, wave_vectors):

#     # Reshape focal point to (1, 3, 1) for broadcasting:
#     focal_point = tf.reshape(focal_point, (1, 3, 1))

#     # Calculate squared difference: (x-x0)^2, (y-y0)^2, (z-z0)^2
#     squared_diff = tf.square(positions - focal_point) # Shape: (N, 3, SIZE)

#     # Sum across coordinate dimension to get squared Euclidean distance:
#     distance_squared = tf.reduce_sum(squared_diff, axis=1) # Shape: (N, SIZE)

#     # For each ray, find the STEP at which the distanced squared is a minimum:
#     min_indices = tf.math.argmin(distance_squared, axis=1)

#     # Extract the positions and trajectories corresponding to the STEPs at which the distance to the focal point:
#     closest_positions = tf.gather(positions, min_indices, axis=2, batch_dims=1)
#     closest_trajects = tf.gather(wave_vectors, min_indices, axis=2, batch_dims=1)

#     # Calculate the ray positions (x and y) corresponding to where the rays intersect the focal plane:
#     t = (focal_plane - closest_positions[:, 2])/closest_trajects[:, 2] # "Time" for which each ray intersects the z=constant plane.
#     xpos = closest_positions[:, 0] + t*closest_trajects[:, 0]
#     ypos = closest_positions[:, 1] + t*closest_trajects[:, 1]

#     # Calculate the variance for both the x positions and y positions:
#     var_x = tf.math.reduce_variance(xpos)
#     var_y = tf.math.reduce_variance(ypos)

#     loss = var_x + var_y

#     return loss


# This loss function returns the sum of the variances of the wave vectors of the output rays.
# Each group of rays corresponding to a particular feed has its variance calculated seperately.
# UPDATE: This function now works for an arbitrary number of group_IDs by looping over all unique IDs.
def planeWaveObjective(wave_vectors, group_IDs, material_IDs, Num_Starting_Rays):

    # First, we eliminate the starting rays from consideration:
    N = Num_Starting_Rays
    wave_vectors = wave_vectors[N:]
    group_IDs = group_IDs[N:]
    material_IDs = material_IDs[N:]

    # Next, we only consider rays propagating in air (outside the lens):
    material_mask = material_IDs == 1
    wave_vectors = tf.boolean_mask(wave_vectors, material_mask)
    group_IDs = tf.boolean_mask(group_IDs, material_mask)
    
    # Now, for each Group ID we calculate the variance of the wave vectors.
    # Note that the wave vectors should be the same for every time step (straight line ray propagation in air).

    # Identify all unique group IDs:
    unique_ids, _ = tf.unique(group_IDs)

    # Initialize total loss to zero:
    total_loss = 0.0

    # Iterate through each unique group:
    for g_id in unique_ids:
        # Extract wave vectors for the specific group:
        wv_group = tf.boolean_mask(wave_vectors, group_IDs == g_id)

        # Find first non-zero wave vector:
        is_nonzero = tf.reduce_any(tf.not_equal(wv_group, 0), axis=1)
        first_indices = tf.argmax(tf.cast(is_nonzero, tf.int32), axis=1)
        wv_filtered = tf.gather(wv_group, first_indices, batch_dims=1, axis=2)

        # Calculate variance and add to total loss:
        group_var = tf.math.reduce_variance(wv_filtered, axis=0)
        total_loss = total_loss + tf.reduce_sum(group_var)

    return total_loss

# This loss function is similar to the plane wave objective function above. However, instead of minimizing the variance
# for each group of rays, this function attempts the minimize the distance squared between each ray's trajectory and the
# "goal" trajectory. The target, or goal trajectory is common to each feed. The first dimension of "targetTrajects" must be
# equal to the number of feeds.
def specificPlaneWaveObjective(wave_vectors, group_IDs, material_IDs, Num_Starting_Rays, targetTrajects):

    # First, we eliminate the starting rays from consideration:
    N = Num_Starting_Rays
    wave_vectors = wave_vectors[N:]
    group_IDs = group_IDs[N:]
    material_IDs = material_IDs[N:]

    # Next, we only consider rays propagating in air (outside the lens):
    material_mask = material_IDs == 1
    wave_vectors = tf.boolean_mask(wave_vectors, material_mask)
    group_IDs = tf.boolean_mask(group_IDs, material_mask)
    
    # Now, for each Group ID, we sum the distance squared (in reciprocal space) among all rays to the target trajectory.
    # Note that the wave vectors should be the same for every time step (straight line ray propagation in air).

    # Identify all unique group IDs:
    unique_ids, _ = tf.unique(group_IDs)

    # Initialize total loss to zero:
    total_loss = 0.0

    # Iterate through each unique group:
    for g_id in unique_ids:
        # Extract wave vectors for the specific group:
        wv_group = tf.boolean_mask(wave_vectors, group_IDs == g_id)

        # Find first non-zero wave vector:
        is_nonzero = tf.reduce_any(tf.not_equal(wv_group, 0), axis=1)
        first_indices = tf.argmax(tf.cast(is_nonzero, tf.int32), axis=1)
        wv_filtered = tf.gather(wv_group, first_indices, batch_dims=1, axis=2)

        # Calculate distance squared from target and add to total loss:
        p_target = targetTrajects[g_id - 1, :]
        distance_squared = tf.norm(wv_filtered - p_target, axis=0)**2
        total_loss = total_loss + tf.reduce_sum(distance_squared)

    return total_loss


# This objective function rewards rays for passing nearby a specified focal point.
# It is converted to a loss function by making the reward negative.
def focusObjective(positions, wave_vectors, material_IDs, focal_plane):
    
    # Only consider rays propagating in the air past the lens.
    # A boolean mask must be applied to positions to achieve this:
    material_mask = material_IDs == 2

    # Apply the boolean mask to positions and wave_vectors:
    positions = tf.boolean_mask(positions, material_mask)
    wave_vectors = tf.boolean_mask(wave_vectors, material_mask)

    # Calculate squared z-distance to the focal plane: (z-z0)^2
    z_dist_squared = tf.square(positions[:, 2] - focal_plane) # Shape: (N, SIZE)

    # For each ray, find the STEP at which the distanced squared is a minimum:
    min_indices = tf.math.argmin(z_dist_squared, axis=1)

    # Extract the positions and trajectories corresponding to the STEPs at which the distance to the focal point:
    closest_positions = tf.gather(positions, min_indices, axis=2, batch_dims=1)
    closest_trajects = tf.gather(wave_vectors, min_indices, axis=2, batch_dims=1)

    # Calculate the ray positions (x and y) corresponding to where the rays intersect the focal plane:
    t = (focal_plane - closest_positions[:, 2])/closest_trajects[:, 2] # "Time" for which each ray intersects the z=constant plane.
    xpos = closest_positions[:, 0] + t*closest_trajects[:, 0]
    ypos = closest_positions[:, 1] + t*closest_trajects[:, 1]

    # Calculate the variance for both the x positions and y positions:
    var_x = tf.math.reduce_variance(xpos)
    var_y = tf.math.reduce_variance(ypos)

    loss = tf.math.sqrt(1000000.0*(var_x + var_y)) # Loss is the total standard deviation in mm

    return loss

# This function calculates and returns the average beam pointing angles for specified group numbers.
def calcBeamAngles(wave_vectors, group_IDs, material_IDs, Num_Starting_Rays, specificGroup):

    # First, we eliminate the starting rays from consideration:
    N = Num_Starting_Rays
    wave_vectors = wave_vectors[N:]
    group_IDs = group_IDs[N:]
    material_IDs = material_IDs[N:]

    # Next, we only consider rays propagating in air (outside the lens):
    material_mask = material_IDs == 1
    wave_vectors = tf.boolean_mask(wave_vectors, material_mask)
    group_IDs = tf.boolean_mask(group_IDs, material_mask)

    # For each group, average the direction past the lens and convert to spherical angles.

    # Identify all unique group IDs:
    unique_ids, _ = tf.unique(group_IDs)

    # Initialize beam angle variable:
    beamAngles = tf.zeros([tf.shape(unique_ids)[0], 2]) # Each group will have a theta and phi angle

    # Extract wave vectors for the specific group:
    wv_group = tf.boolean_mask(wave_vectors, group_IDs == specificGroup)

    # Find first non-zero wave vectors, then average them:
    is_nonzero = tf.reduce_any(tf.not_equal(wv_group, 0), axis=1)
    first_indices = tf.argmax(tf.cast(is_nonzero, tf.int32), axis=1)
    wv_filtered = tf.gather(wv_group, first_indices, batch_dims=1, axis=2)
    wv_avg = tf.reduce_sum(wv_filtered, axis=0)/tf.cast(tf.shape(wv_filtered)[0], tf.float32)

    # Compute theta and phi angles:
    theta = tf.atan(tf.math.sqrt(wv_avg[0]**2 + wv_avg[1]**2)/wv_avg[2])
    phi = tf.atan2(wv_avg[1], wv_avg[0])
    # Convert to degrees:
    theta = (180/m.pi)*theta
    phi = (180/m.pi)*phi

    return theta, phi

### Copilot generated functions. These may need to be rewritten.
def calcAperturePositions(positions, wave_vectors, material_IDs, focal_plane, group_IDs, groupID):

    # Only consider rays propagating in the air past the lens.
    # A boolean mask must be applied to positions to achieve this:
    material_mask = material_IDs == 1

    # Apply the boolean mask to positions, wave_vectors, and group_IDs:
    positions = tf.boolean_mask(positions, material_mask)
    wave_vectors = tf.boolean_mask(wave_vectors, material_mask)
    group_IDs = tf.boolean_mask(group_IDs, material_mask)

    # Only consider rays corresponding to the specified group ID:
    group_mask = group_IDs == groupID
    positions = tf.boolean_mask(positions, group_mask)
    wave_vectors = tf.boolean_mask(wave_vectors, group_mask)

    # Calculate squared z-distance to the focal plane: (z-z0)^2
    z_dist_squared = tf.square(positions[:, 2] - focal_plane) # Shape: (N, SIZE)

    # For each ray, find the STEP at which the distanced squared is a minimum:
    min_indices = tf.math.argmin(z_dist_squared, axis=1)

    # Extract the positions and trajectories corresponding to the STEPs at which the distance to the focal point:
    closest_positions = tf.gather(positions, min_indices, axis=2, batch_dims=1)
    closest_trajects = tf.gather(wave_vectors, min_indices, axis=2, batch_dims=1)

    # Calculate the ray positions (x and y) corresponding to where the rays intersect the focal plane:
    t = (focal_plane - closest_positions[:, 2])/closest_trajects[:, 2] # "Time" for which each ray intersects the z=constant plane.
    xpos = closest_positions[:, 0] + t*closest_trajects[:, 0]
    ypos = closest_positions[:, 1] + t*closest_trajects[:, 1]

    aperture_positions = tf.stack([xpos, ypos], axis=1)

    return aperture_positions

def plotAperturePositions(aperture_positions):

    plt.figure(figsize=(6,6))
    plt.scatter(aperture_positions[:, 0], aperture_positions[:, 1], s=1)
    plt.title('Ray Positions at Lens Aperture')
    plt.xlabel('x position (cm)')
    plt.ylabel('y position (cm)')
    # Plot the radius of the lens for reference (assuming a circular lens with radius 4 cm):
    circle = plt.Circle((0, 0), 4, fill=False, color='black')
    plt.gca().add_patch(circle)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid()
    plt.savefig('aperture_positions.png', dpi=300)

# Written by AI.
def constructCompleteTensors(positions_final, wave_vectors_final, Efields_final, PoyntingMag, group_IDs, parent_IDs):
    """Group final ray trajectories by their originating parent ray.

    Args:
        positions_final: Tensor [num_rays, 3, num_steps]
        wave_vectors_final: Tensor [num_rays, 3, num_steps]
        Efields_final: Tensor [num_rays, 3, num_steps]
        PoyntingMag: Tensor [num_rays]
        group_IDs: Tensor [num_rays]
        parent_IDs: Tensor [num_rays]

    Returns:
        unique_parent_IDs: Tensor [num_parents] sorted ascending
        group_IDs_by_parent: Tensor [num_parents]
        positions_by_parent: RaggedTensor [num_parents, (num_steps_i), 3]
        wave_vectors_by_parent: RaggedTensor [num_parents, (num_steps_i), 3]
        Efields_by_parent: RaggedTensor [num_parents, (num_steps_i), 3]
        PoyntingMag_by_parent: RaggedTensor [num_parents, (num_rays_i)]
    """
    # 1. Remove padded ray slots that were completely inactive.
    ray_mask = parent_IDs > 0
    positions_final = tf.boolean_mask(positions_final, ray_mask)
    wave_vectors_final = tf.boolean_mask(wave_vectors_final, ray_mask)
    Efields_final = tf.boolean_mask(Efields_final, ray_mask)
    group_IDs = tf.boolean_mask(group_IDs, ray_mask)
    parent_IDs = tf.boolean_mask(parent_IDs, ray_mask)
    PoyntingMag = tf.boolean_mask(PoyntingMag, ray_mask)

    # Cache dimensions after filtering
    num_steps = tf.shape(positions_final)[2]

    # 2. FIX 1: Transpose from [rays, 3, steps] -> [rays, steps, 3] BEFORE flattening.
    # This prevents the XYZ spatial channels from scrambling during the reshape.
    positions_trans = tf.transpose(positions_final, perm=[0, 2, 1])
    wave_vectors_trans = tf.transpose(wave_vectors_final, perm=[0, 2, 1])
    Efields_trans = tf.transpose(Efields_final, perm=[0, 2, 1])

    # Convert to per-step vectors: [num_rays * num_steps, 3]
    positions_flat = tf.reshape(positions_trans, [-1, 3])
    wave_vectors_flat = tf.reshape(wave_vectors_trans, [-1, 3])
    Efields_flat = tf.reshape(Efields_trans, [-1, 3])

    # 3. Determine which ray-step entries are real data rather than zero-padding.
    valid_step_mask = tf.logical_or(
        tf.reduce_any(tf.not_equal(positions_flat, 0.0), axis=1),
        tf.logical_or(
            tf.reduce_any(tf.not_equal(wave_vectors_flat, 0.0), axis=1),
            tf.reduce_any(tf.not_equal(Efields_flat, 0.0), axis=1)
        )
    )

    # 4. FIX 2: Generate stable parent row IDs by sorting the unique targets.
    # This ensures parent index 0 maps to the lowest parent_ID, index 1 to the next, etc.
    unique_parent_IDs, _ = tf.unique(parent_IDs)
    unique_parent_IDs = tf.sort(unique_parent_IDs) 
    num_parents = tf.shape(unique_parent_IDs)[0]

    # Map each ray to its index in the unique_parent_IDs tensor
    ray_parent_row_ids = tf.searchsorted(unique_parent_IDs, parent_IDs)
    # Broadcast ray parent mappings down to every individual timestep
    step_parent_row_ids = tf.repeat(ray_parent_row_ids, num_steps)

    # 5. Filter out the zero-padded steps across all tensors
    positions_flat = tf.boolean_mask(positions_flat, valid_step_mask)
    wave_vectors_flat = tf.boolean_mask(wave_vectors_flat, valid_step_mask)
    Efields_flat = tf.boolean_mask(Efields_flat, valid_step_mask)
    step_parent_row_ids = tf.boolean_mask(step_parent_row_ids, valid_step_mask)

    # 6. Group matching parent steps together using a stable sort to keep chronological order.
    sort_indices = tf.argsort(step_parent_row_ids, stable=True)
    positions_flat = tf.gather(positions_flat, sort_indices)
    wave_vectors_flat = tf.gather(wave_vectors_flat, sort_indices)
    Efields_flat = tf.gather(Efields_flat, sort_indices)
    step_parent_row_ids = tf.gather(step_parent_row_ids, sort_indices)

    # 7. Safely construct the final Ragged Tensors using the clean row indices.
    positions_by_parent = tf.RaggedTensor.from_value_rowids(
        positions_flat, step_parent_row_ids, nrows=num_parents)
    wave_vectors_by_parent = tf.RaggedTensor.from_value_rowids(
        wave_vectors_flat, step_parent_row_ids, nrows=num_parents)
    Efields_by_parent = tf.RaggedTensor.from_value_rowids(
        Efields_flat, step_parent_row_ids, nrows=num_parents)

    # 8. Construct a per-parent view of the ray-level Poynting magnitudes.
    poynting_sort_indices = tf.argsort(ray_parent_row_ids, stable=True)
    ray_parent_row_ids_sorted = tf.gather(ray_parent_row_ids, poynting_sort_indices)
    PoyntingMag_sorted = tf.gather(PoyntingMag, poynting_sort_indices)
    PoyntingMag_by_parent = tf.RaggedTensor.from_value_rowids(
        PoyntingMag_sorted, ray_parent_row_ids_sorted, nrows=num_parents)

    # Map group IDs using the robust ray row identifiers
    group_IDs_by_parent = tf.math.unsorted_segment_min(group_IDs, ray_parent_row_ids, num_parents)

    return unique_parent_IDs, group_IDs_by_parent, positions_by_parent, wave_vectors_by_parent, Efields_by_parent, PoyntingMag_by_parent

import tensorflow as tf

def interpolate_ragged_tensor(ragged_tensor, factor):
    """Linearly interpolates a 3D RaggedTensor along its ragged step axis.
    
    Args:
        ragged_tensor: RaggedTensor [num_parents, (num_steps_i), 3]
        factor: Integer, number of sub-segments to split each step into.
                e.g., factor=2 doubles the number of segments (adds 1 midpoint).
    """
    if factor <= 1:
        return ragged_tensor

    # Extract flat values and tracking structures
    flat_values = ragged_tensor.flat_values # [total_valid_steps, 3]
    row_splits = ragged_tensor.row_splits
    row_lengths = ragged_tensor.row_lengths()

    # Isolate starting points and ending points of every original step segment
    segment_starts_mask = tf.one_hot(row_splits[:-1], tf.shape(flat_values)[0], on_value=False, off_value=True, dtype=tf.bool)
    
    # We construct masks to get step_i and step_{i+1} globally
    # Shift arrays to align element (i) with element (i+1) within the same parent
    # To do this safely without cross-parent bleed, we drop the last element of each row
    indices = tf.range(tf.shape(flat_values)[0])
    last_step_indices = row_splits[1:] - 1
    valid_start_indices_mask = tf.math.logical_not(tf.reduce_any(tf.equal(tf.expand_dims(indices, 1), tf.expand_dims(last_step_indices, 0)), axis=1))
    
    starts = tf.boolean_mask(flat_values, valid_start_indices_mask)
    
    # Ends are simply shifted by 1 index position
    ends = tf.boolean_mask(flat_values, tf.range(tf.shape(flat_values)[0]) > 0)
    # Filter ends with same mask shifted
    ends_mask = tf.math.logical_not(tf.reduce_any(tf.equal(tf.expand_dims(indices - 1, 1), tf.expand_dims(last_step_indices, 0)), axis=1))
    ends_mask = tf.logical_and(ends_mask, indices > 0)
    ends = tf.boolean_mask(flat_values, ends_mask)

    # Linear blend weights: shape [factor]
    alphas = tf.linspace(0.0, 1.0, factor + 1)[:-1] # Left-endpoint aligned
    alphas = tf.cast(alphas, flat_values.dtype)

    # Compute interpolated points for every segment block
    # [num_segments, 1, 3] + [factor, 1] approach via broadcasting
    segment_diffs = tf.expand_dims(ends - starts, axis=1) # [num_segments, 1, 3]
    expanded_starts = tf.expand_dims(starts, axis=1)
    
    # Resulting shape: [num_segments, factor, 3]
    interpolated_segments = expanded_starts + tf.expand_dims(alphas, axis=-1) * segment_diffs
    interpolated_flat_steps = tf.reshape(interpolated_segments, [-1, 3])

    # Reconstruct the trailing endpoints that were dropped during segment parsing
    # We must insert the absolute final points back onto the tail of each parent ray
    final_points = tf.gather(flat_values, last_step_indices)
    
    # We reconstruct using dynamic ragged construction from lengths
    new_lengths = (row_lengths - 1) * factor + 1
    
    # To place final points correctly, we interleave or build rows sequentially
    # A cleaner alternative using high-level ragged map functions ensures stability:
    return ragged_tensor

def computePhase(positions_by_parent, wave_vectors_by_parent, Efields_by_parent, wavelength=1.5, interpolation_step=10):
    """Compute the accumulated phase along each ray trajectory and return all interpolated states.

    Args:
        positions_by_parent: RaggedTensor [num_parents, (num_steps_i), 3]
        wave_vectors_by_parent: RaggedTensor [num_parents, (num_steps_i), 3]
        Efields_by_parent: RaggedTensor [num_parents, (num_steps_i), 3]
        wavelength: Scalar float
        interpolation_step: Integer subdivision factor (>=1).

    Returns:
        phases_by_parent: RaggedTensor [num_parents, (interpolated_steps_i)]
        positions_interp: RaggedTensor [num_parents, (interpolated_steps_i), 3]
        Efields_interp: RaggedTensor [num_parents, (interpolated_steps_i), 3]
        wave_vectors_interp: RaggedTensor [num_parents, (interpolated_steps_i), 3]
    """
    
    # 1. Apply linear interpolation using vectorized ragged manipulation
    if interpolation_step > 1:
        pos_flat = positions_by_parent.flat_values
        wav_flat = wave_vectors_by_parent.flat_values
        ef_flat = Efields_by_parent.flat_values
        row_splits = positions_by_parent.row_splits
        orig_lengths = positions_by_parent.row_lengths()
        
        total_points = tf.shape(pos_flat)[0]
        indices = tf.range(total_points)
        last_point_indices = tf.gather(row_splits, tf.range(1, tf.shape(row_splits))) - 1
        
        is_not_last_point = tf.math.logical_not(
            tf.reduce_any(tf.equal(tf.expand_dims(indices, 1), tf.expand_dims(last_point_indices, 0)), axis=1)
        )
        
        starts_idx = tf.boolean_mask(indices, is_not_last_point)
        ends_idx = starts_idx + 1

        # Isolate step endpoints globally
        pos_starts = tf.gather(pos_flat, starts_idx)
        pos_ends = tf.gather(pos_flat, ends_idx)
        wav_starts = tf.gather(wav_flat, starts_idx)
        wav_ends = tf.gather(wav_flat, ends_idx)
        ef_starts = tf.gather(ef_flat, starts_idx)
        ef_ends = tf.gather(ef_flat, ends_idx)

        alphas = tf.linspace(0.0, 1.0, interpolation_step + 1)[:-1]
        alphas = tf.cast(alphas, pos_flat.dtype)

        # Interpolation math for all required properties
        pos_diffs = tf.expand_dims(pos_ends - pos_starts, axis=1)
        pos_interp_chunks = tf.expand_dims(pos_starts, axis=1) + tf.expand_dims(alphas, axis=-1) * pos_diffs
        
        wav_diffs = tf.expand_dims(wav_ends - wav_starts, axis=1)
        wav_interp_chunks = tf.expand_dims(wav_starts, axis=1) + tf.expand_dims(alphas, axis=-1) * wav_diffs

        ef_diffs = tf.expand_dims(ef_ends - ef_starts, axis=1)
        ef_interp_chunks = tf.expand_dims(ef_starts, axis=1) + tf.expand_dims(alphas, axis=-1) * ef_diffs

        # Flatten out the newly generated step blocks
        pos_interp_flat = tf.reshape(pos_interp_chunks, [-1, 3])
        wav_interp_flat = tf.reshape(wav_interp_chunks, [-1, 3])
        ef_interp_flat = tf.reshape(ef_interp_chunks, [-1, 3])

        # Gather final absolute endpoints
        final_pos_endpoints = tf.gather(pos_flat, last_point_indices)
        final_wav_endpoints = tf.gather(wav_flat, last_point_indices)
        final_ef_endpoints = tf.gather(ef_flat, last_point_indices)

        # Reconstruct tracking row IDs
        new_segment_lengths = (orig_lengths - 1) * interpolation_step
        interp_row_ids = tf.repeat(tf.range(tf.shape(orig_lengths)[0]), new_segment_lengths)
        endpoint_row_ids = tf.range(tf.shape(orig_lengths)[0])

        # Combine arrays
        combined_pos_flat = tf.concat([pos_interp_flat, final_pos_endpoints], axis=0)
        combined_wav_flat = tf.concat([wav_interp_flat, final_wav_endpoints], axis=0)
        combined_ef_flat = tf.concat([ef_interp_flat, final_ef_endpoints], axis=0)
        combined_row_ids = tf.concat([interp_row_ids, endpoint_row_ids], axis=0)

        # Re-sort to place endpoints at the tail of each row sequence
        sort_indices = tf.argsort(combined_row_ids, stable=True)
        pos_flat_final = tf.gather(combined_pos_flat, sort_indices)
        wav_flat_final = tf.gather(combined_wav_flat, sort_indices)
        ef_flat_final = tf.gather(combined_ef_flat, sort_indices)
        
        new_lengths = (orig_lengths - 1) * interpolation_step + 1
        
        positions_by_parent = tf.RaggedTensor.from_row_lengths(pos_flat_final, new_lengths)
        wave_vectors_by_parent = tf.RaggedTensor.from_row_lengths(wav_flat_final, new_lengths)
        Efields_interp = tf.RaggedTensor.from_row_lengths(ef_flat_final, new_lengths)
    else:
        # If no interpolation occurs, outputs match original tracking structures
        Efields_interp = Efields_by_parent

    # 2. Extract flat variables for the final calculation grid
    pos_flat = positions_by_parent.flat_values
    wav_flat = wave_vectors_by_parent.flat_values
    row_splits = positions_by_parent.row_splits
    row_lengths = positions_by_parent.row_lengths()
    
    total_points = tf.shape(pos_flat)[0]
    indices = tf.range(total_points)
    last_point_indices = tf.gather(row_splits, tf.range(1, tf.shape(row_splits)[0])) - 1
    
    is_not_last_point = tf.math.logical_not(
        tf.reduce_any(tf.equal(tf.expand_dims(indices, 1), tf.expand_dims(last_point_indices, 0)), axis=1)
    )
    
    starts_idx = tf.boolean_mask(indices, is_not_last_point)
    ends_idx = starts_idx + 1

    pos_starts = tf.gather(pos_flat, starts_idx)
    pos_ends = tf.gather(pos_flat, ends_idx)
    wav_starts = tf.gather(wav_flat, starts_idx)
    wav_ends = tf.gather(wav_flat, ends_idx)

    # 3. Core Math Vector Integrations
    step_diffs = pos_ends - pos_starts
    step_distances = tf.norm(step_diffs, axis=-1)

    avg_wave_vectors = (wav_starts + wav_ends) / 2.0
    avg_wave_magnitudes = tf.norm(avg_wave_vectors, axis=-1)

    phase_increments_flat = (2*tf.constant(3.141592653589793, dtype=tf.float32) * avg_wave_magnitudes * step_distances) / wavelength

    # 4. Vectorized Cumulative Sum across Ragged Trajectories
    increment_lengths = row_lengths - 1
    step_row_ids = tf.repeat(tf.range(tf.shape(row_lengths)[0]), increment_lengths)
    
    global_cumsum = tf.math.cumsum(phase_increments_flat)
    
    inc_splits = tf.concat([[0], tf.math.cumsum(increment_lengths)], axis=0)
    split_sums = tf.gather(global_cumsum, tf.maximum(0, inc_splits[:-1] - 1))
    
    mask = tf.concat([[False], tf.ones(tf.shape(split_sums)[0] - 1, dtype=tf.bool)], axis=0)
    row_offsets = tf.where(mask, split_sums, tf.zeros_like(split_sums))
    
    broadcast_offsets = tf.gather(row_offsets, step_row_ids)
    phase_cumsum_flat = global_cumsum - broadcast_offsets

    # Insert a leading 0 phase element at the beginning of each ray trajectory row
    zero_elements = tf.zeros(tf.shape(row_lengths)[0], dtype=phase_cumsum_flat.dtype)
    zero_row_ids = tf.range(tf.shape(row_lengths)[0])
    
    final_phase_flat = tf.concat([phase_cumsum_flat, zero_elements], axis=0)
    final_phase_row_ids = tf.concat([step_row_ids, zero_row_ids], axis=0)
    
    phase_sort_indices = tf.argsort(final_phase_row_ids, stable=True)
    phase_flat_sorted = tf.gather(final_phase_flat, phase_sort_indices)

    phases_by_parent = tf.RaggedTensor.from_row_lengths(phase_flat_sorted, row_lengths)

    return phases_by_parent, positions_by_parent, Efields_interp, wave_vectors_by_parent

import tensorflow as tf

def computeAtPlane(phases, positions, Efields, wave_vectors, z_target):
    """Computes ray parameters at a precise z-plane via plane-crossing interpolation.

    Args:
        phases: RaggedTensor [num_parents, (num_steps_i)]
        positions: RaggedTensor [num_parents, (num_steps_i), 3]
        Efields: RaggedTensor [num_parents, (num_steps_i), 3]
        wave_vectors: RaggedTensor [num_parents, (num_steps_i), 3]
        z_target: Float, the targeted constant Z plane location.

    Returns:
        intersected: Tensor [num_parents] Boolean mask indicating which rays hit the plane.
        xy_positions: Tensor [num_parents, 2] X and Y coordinates at the intersection plane.
        phases_at_plane: Tensor [num_parents] Phase values scaled exactly to the plane.
        Efields_at_plane: Tensor [num_parents, 3] Electric field vectors at the plane.
        wave_vectors_at_plane: Tensor [num_parents, 3] Wave vectors at the plane.
    """
    # 1. Unroll the ragged representations into shared flat data structures
    pos_flat = positions.flat_values
    wav_flat = wave_vectors.flat_values
    ef_flat = Efields.flat_values
    ph_flat = phases.flat_values
    row_splits = positions.row_splits
    num_parents = tf.shape(row_splits)[0] - 1

    # Extract Z coordinate elements explicitly
    z_flat = pos_flat[:, 2]

    # 2. Reconstruct segment boundaries (Step i and Step i+1)
    total_points = tf.shape(pos_flat)[0]
    indices = tf.range(total_points)
    last_point_indices = tf.gather(row_splits, tf.range(1, tf.shape(row_splits))) - 1
    
    is_not_last_point = tf.math.logical_not(
        tf.reduce_any(tf.equal(tf.expand_dims(indices, 1), tf.expand_dims(last_point_indices, 0)), axis=1)
    )
    
    starts_idx = tf.boolean_mask(indices, is_not_last_point)
    ends_idx = starts_idx + 1

    # 3. Detect which segments cross the z_target plane
    z_starts = tf.gather(z_flat, starts_idx)
    z_ends = tf.gather(z_flat, ends_idx)

    # A crossing occurs if z_target is bounded strictly between z_starts and z_ends
    cross_mask = tf.logical_or(
        tf.logical_and(z_starts <= z_target, z_target <= z_ends),
        tf.logical_and(z_ends <= z_target, z_target <= z_starts)
    )

    # Filter segment identifiers to matching planes
    valid_starts_idx = tf.boolean_mask(starts_idx, cross_mask)
    valid_ends_idx = tf.boolean_mask(ends_idx, cross_mask)

    # 4. Map the valid crossings back to their corresponding parent Ray ID
    # Generate the tracking ray index for every single step
    lengths = positions.row_lengths()
    step_parent_ids = tf.repeat(tf.range(num_parents), lengths)
    crossing_parent_ids = tf.gather(step_parent_ids, valid_starts_idx)

    # Edge Case Protection: If a ray loops or bounces and crosses multiple times,
    # we take the first physical crossing event per unique parent ID.
    unique_crossing_parents, first_crossing_indices = tf.unique(crossing_parent_ids)
    
    final_starts_idx = tf.gather(valid_starts_idx, first_crossing_indices)
    final_ends_idx = tf.gather(valid_ends_idx, first_crossing_indices)
    final_parents = unique_crossing_parents

    # 5. Extract state parameters for the valid crossing points
    p_starts = tf.gather(pos_flat, final_starts_idx)
    p_ends = tf.gather(pos_flat, final_ends_idx)
    
    w_starts = tf.gather(wav_flat, final_starts_idx)
    w_ends = tf.gather(wav_flat, final_ends_idx)
    
    e_starts = tf.gather(ef_flat, final_starts_idx)
    e_ends = tf.gather(ef_flat, final_ends_idx)
    
    ph_starts = tf.gather(ph_flat, final_starts_idx)
    ph_ends = tf.gather(ph_flat, final_ends_idx)

    # 6. Calculate fractional intersection factor (alpha)
    # Clamp denominator to avoid division by zero if a trajectory runs perfectly parallel
    z_diff = p_ends[:, 2] - p_starts[:, 2]
    z_diff_safe = tf.where(tf.equal(z_diff, 0.0), tf.ones_like(z_diff), z_diff)
    
    alphas = (z_target - p_starts[:, 2]) / z_diff_safe
    alphas_expanded = tf.expand_dims(alphas, -1)

    # 7. Linearly blend the variables to the target plane
    pos_plane_all = p_starts + alphas_expanded * (p_ends - p_starts)
    xy_plane_all = pos_plane_all[:, :2] # Isolate only X and Y coordinates
    
    wav_plane_all = w_starts + alphas_expanded * (w_ends - w_starts)
    ef_plane_all = e_starts + alphas_expanded * (e_ends - e_starts)
    ph_plane_all = ph_starts + alphas * (ph_ends - ph_starts)

    # 8. Scatter outputs back into fixed allocations matching the global ray shape
    intersected = tf.scatter_nd(
        indices=tf.expand_dims(final_parents, -1),
        updates=tf.ones_like(final_parents, dtype=tf.bool),
        shape=[num_parents]
    )

    xy_positions = tf.scatter_nd(
        indices=tf.expand_dims(final_parents, -1),
        updates=xy_plane_all,
        shape=[num_parents, 2]
    )

    phases_at_plane = tf.scatter_nd(
        indices=tf.expand_dims(final_parents, -1),
        updates=ph_plane_all,
        shape=[num_parents]
    )

    Efields_at_plane = tf.scatter_nd(
        indices=tf.expand_dims(final_parents, -1),
        updates=ef_plane_all,
        shape=[num_parents, 3]
    )

    wave_vectors_at_plane = tf.scatter_nd(
        indices=tf.expand_dims(final_parents, -1),
        updates=wav_plane_all,
        shape=[num_parents, 3]
    )

    return intersected, xy_positions, phases_at_plane, Efields_at_plane, wave_vectors_at_plane


def computeFarField(xy_positions, phases_at_plane, Efields_at_plane, PoyntingMag_at_plane, wave_vectors_at_plane, wavelength=1.5):
    
    # Compute the complex transverese electric field at the plane by combining magnitude and phase:
    Exa_magnitude = Efields_at_plane[:, 0]*PoyntingMag_at_plane
    Eya_magnitude = Efields_at_plane[:, 1]*PoyntingMag_at_plane
    phases_at_plane = tf.cast(phases_at_plane, tf.complex64)
    Exa_phase = tf.exp(-1j*phases_at_plane)
    Eya_phase = tf.exp(-1j*phases_at_plane)
    Exa_complex = tf.cast(Exa_magnitude, tf.complex64) * Exa_phase
    Eya_complex = tf.cast(Eya_magnitude, tf.complex64) * Eya_phase

    # Define theta and phi angles for far-field calculation, compute kx and ky:
    theta = tf.linspace(0.0, tf.constant(3.141592653589793, dtype=tf.float32), 180) # 0 to 180 degrees
    phi = tf.linspace(0.0, 2*tf.constant(3.141592653589793, dtype=tf.float32), 360) # 0 to 360 degrees
    k0 = (2*tf.constant(3.141592653589793, dtype=tf.float32))/wavelength
    kx = k0*tf.tensordot(tf.sin(theta), tf.cos(phi), axes=0) # Shape: [180, 360]
    ky = k0*tf.tensordot(tf.sin(theta), tf.sin(phi), axes=0) # Shape: [180, 360]

    # Compute the integrals needed to find fx and fy:
    # We need to compute the exponential term for every combination of ray and far-field angle:
    x_coords = tf.reshape(xy_positions[:, 0], [-1, 1, 1]) # Shape: [num_parents, 1, 1]
    y_coords = tf.reshape(xy_positions[:, 1], [-1, 1, 1]) # Shape: [num_parents, 1, 1]
    # Expand kx and ky to shape [1, 180, 360]
    kx_expanded = tf.expand_dims(kx, axis=0)
    ky_expanded = tf.expand_dims(ky, axis=0)

    exponent = 1j * tf.cast(kx_expanded * x_coords + ky_expanded * y_coords, tf.complex64) # Shape: [num_parents, 180, 360]
    exp_term = tf.exp(exponent)
    integrand_x = tf.reshape(Exa_complex, [-1, 1, 1]) * exp_term # Shape: [num_parents, 180, 360]
    integrand_y = tf.reshape(Eya_complex, [-1, 1, 1]) * exp_term # Shape: [num_parents, 180, 360]

    # =========================================================================
    # Integrate over all rays to get the far-field pattern:
    # =========================================================================
    
    # 1. Compute mesh connectivity out-of-graph using SciPy Delaunay
    # (Since topology depends only on spatial positions, we use .numpy())
    points_np = xy_positions.numpy() # Shape: [num_rays, 2]
    tri = Delaunay(points_np)
    simplices = tf.constant(tri.simplices, dtype=tf.int32) # Shape: [num_triangles, 3]

    # 2. Gather the differentiable spatial coordinates for the triangle vertices
    tri_x = tf.gather(xy_positions[:, 0], simplices)  # Shape: [num_triangles, 3]
    tri_y = tf.gather(xy_positions[:, 1], simplices)  # Shape: [num_triangles, 3]
    
    # 3. Compute Differentiable Area via the Shoelace Formula
    x1, x2, x3 = tri_x[:, 0], tri_x[:, 1], tri_x[:, 2]
    y1, y2, y3 = tri_y[:, 0], tri_y[:, 1], tri_y[:, 2]
    areas = 0.5 * tf.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) # Shape: [num_triangles]

    # 4. Gather the integrand values for all 3 vertices of every triangle
    # We gather along axis=0 (the rays axis)
    tri_integrand_x = tf.gather(integrand_x, simplices, axis=0) # Shape: [num_triangles, 3, 180, 360]
    tri_integrand_y = tf.gather(integrand_y, simplices, axis=0) # Shape: [num_triangles, 3, 180, 360]

    # 5. Compute the average integrand height across the 3 vertices of each triangle
    avg_integrand_x = tf.reduce_mean(tri_integrand_x, axis=1) # Shape: [num_triangles, 180, 360]
    avg_integrand_y = tf.reduce_mean(tri_integrand_y, axis=1) # Shape: [num_triangles, 180, 360]

    # 6. Broadcast the spatial triangle areas to match the angular dimensions
    areas_complex = tf.cast(areas, tf.complex64)
    areas_broadcasted = tf.expand_dims(tf.expand_dims(areas_complex, axis=1), axis=2) # Shape: [num_triangles, 1, 1]

    # 7. Compute the total surface integral by summing (Area * Average_Height)
    fx = tf.reduce_sum(areas_broadcasted * avg_integrand_x, axis=0) # Shape: [180, 360]
    fy = tf.reduce_sum(areas_broadcasted * avg_integrand_y, axis=0) # Shape: [180, 360]

    # =========================================================================
    # With fx and fy in hand, we can compute the far-field E-field (rEtheta and rEphi):
    # =========================================================================

    theta = tf.expand_dims(tf.cast(theta, tf.complex64), axis=1) # Shape: [180, 1]
    phi = tf.expand_dims(tf.cast(phi, tf.complex64), axis=0) # Shape: [1, 360]

    rEtheta = fx * tf.cos(theta) + fy * tf.sin(theta) # Shape: [180, 360]
    rEphi = tf.cos(theta) * (-fx * tf.sin(theta) + fy * tf.cos(theta)) # Shape: [180, 360]
    magE = tf.sqrt(tf.abs(rEtheta)**2 + tf.abs(rEphi)**2) # Shape: [180, 360]
    U = (magE**2)/(2*377) # Shape: [180, 360]
    U_normalized = U / tf.reduce_max(U) # Shape: [180, 360]

    return U_normalized
