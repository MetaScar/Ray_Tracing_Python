import tensorflow as tf
import tensorflow_probability as tfp
import DkDist as dk
import math as m
import matplotlib.pyplot as plt

# These functions are used by the new tensor-based ray-tracing algorithm.
# The functions are kept in this seperate Python file for clarity.

### ---------- Ray Update Equations ---------- ###

# This function computes a single time step for ALL rays at once, whether they are alive or not.
# Optical path length (OPL) computations are not currently included.
def rayPropagation(positions, wave_normals, Efields, alive, ordinary_constants, material_IDs, distance_step, dk_dist_flag):

    # Calculate e_perp, e_parallel, the director, and associated spatial derivatives for all rays:
    e_perp, deperp_dx, deperp_dy, deperp_dz = dk.getOrdinaryPermittivities(positions, ordinary_constants, dk_dist_flag)

    # Calculate k1-k6 for each ray:
    k1 = deperp_dx
    k2 = deperp_dy
    k3 = deperp_dz
    k4 = 2*wave_normals[:, 0]
    k5 = 2*wave_normals[:, 1]
    k6 = 2*wave_normals[:, 2]

    # Scale k4, k5, and k6 such that the ray travels a fixed distance:
    current_step = tf.math.sqrt(k4**2 + k5**2 + k6**2)
    current_step = current_step + 1e-12 # To prevent NaNs when the ray is stationary (i.e. current_step = 0)
    h = distance_step/current_step
    h = tf.expand_dims(h, axis=1)

    # Time-step using the first-order Runge-Kutta method:
    new_wave_normals = wave_normals + (h*(tf.stack([k1, k2, k3], axis=1)))
    new_positions = positions + (h*(tf.stack([k4, k5, k6], axis=1)))

    # Calculate the new polarization vectors for each ray:
    common_denominator = 2*e_perp*tf.norm(wave_normals, axis=1) + 1e-12
    k7 = -1.0*Efields[:, 0]*deperp_dx*wave_normals[:, 0]/common_denominator
    k8 = -1.0*Efields[:, 1]*deperp_dy*wave_normals[:, 1]/common_denominator
    k9 = -1.0*Efields[:, 2]*deperp_dz*wave_normals[:, 2]/common_denominator
    
    # Time-step and renormalize:
    E = Efields + (h*tf.stack([k7, k8, k9], axis=1))
    E = E/(tf.expand_dims(tf.norm(E, axis=1), axis=1) + 1e-12)

    # Only update rays that are "alive" (i.e. still being traced):
    alive_expanded = tf.expand_dims(alive, axis=1) # Turns alive from size N to size (N, 1)
    new_wave_normals = tf.where(alive_expanded, new_wave_normals, wave_normals)
    new_positions = tf.where(alive_expanded, new_positions, positions)
    new_Efields = tf.where(alive_expanded, E, Efields)
    
    return new_wave_normals, new_positions, new_Efields

### ---------- Geometry Related ---------- ###

# Given a tensor of positions and a boundingBox geometry, this function returns an "alive"
# tensor of booleans which determines whether each ray should continue to be traced.
# Before running the code, make sure to uncomment the block of code based on the boundingBox geometry type.
def checkBoundary(positions, boundingBox, current_alive, mat_type):
    
    # Cylindrical:
    if mat_type == 'cylinder':
        r = tf.math.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        z = positions[:, 2]
        alive = tf.logical_and(tf.logical_and(r<=boundingBox[1], r>=boundingBox[0]), tf.logical_and(z<=boundingBox[3], z>=boundingBox[2]))
        alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
        return alive
 
    # Spherical:
    if mat_type == 'sphere':
        r = tf.math.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
        alive = r<=boundingBox[4]
        alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
        return alive

    # Rectangular Prism Slabs:
    if mat_type == 'slab':
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        alive = tf.logical_and(tf.logical_and(x>=boundingBox[0], x<=boundingBox[1]), tf.logical_and(tf.logical_and(y>=boundingBox[2], y<=boundingBox[3]) , tf.logical_and(z>=boundingBox[4], z<=boundingBox[5])))
        alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
        return alive

# Given a tensor of positions and a tensor of material IDs (in the forms of integers), this function
# returns a tensor of booleans. True <==> New Material reached, False <==> same material.
def checkForHit(positions, material_IDs, geometry_vectors, mat_type):
    new_material_IDs = getMaterialsAtCoordinates(positions, geometry_vectors, mat_type)
    hit = tf.not_equal(material_IDs, new_material_IDs)
    return hit

# Given a tensor of positions (Nx3) and a list of geometry vectors, this function returns a tensor of integers corresponding 
# to the material associated with each position.
# The type of geometry used must be MANUALLY UNCOMMENTED when switches materials.
# Only one type of material is supported in a simulation.
# See the "MaterialClass.py" file for how to properly defined geometry vectors for each type of geometry.
def getMaterialsAtCoordinates(positions, geometry_vectors, mat_type):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Cylinders:
    # Note - This only works for two cylinders. The first cylinder must be at the origin.
    if mat_type == 'cylinder':
        r = tf.math.sqrt(x**2 + y**2)
        cond = tf.logical_and(tf.logical_and(r>=geometry_vectors[0][0], r<=geometry_vectors[0][1]), tf.logical_and(z>=geometry_vectors[0][2], z<=geometry_vectors[0][3]))
        cond = tf.cast(tf.logical_not(cond), dtype=tf.int32)
        return cond

    # Spheres:
    if mat_type == 'sphere':
        r = tf.math.sqrt(x**2 + y**2 + z**2)
        rmin = geometry_vectors[:, 3]
        rmax = geometry_vectors[:, 4]

        r = tf.expand_dims(r, axis=1)
        rmin = tf.expand_dims(rmin, axis=0)
        rmax = tf.expand_dims(rmax, axis=0)

        mask = tf.logical_and(r>=rmin, r<=rmax)
        indices = tf.argmax(tf.cast(mask, tf.int32), axis=1)
        return indices

    # Rectangular Prism Slabs:
    if mat_type == 'slab':
        x = tf.expand_dims(x, axis=1)
        y = tf.expand_dims(y, axis=1)
        z = tf.expand_dims(z, axis=1)

        xmin = tf.expand_dims(geometry_vectors[:, 0], axis=0)
        xmax = tf.expand_dims(geometry_vectors[:, 1], axis=0)
        ymin = tf.expand_dims(geometry_vectors[:, 2], axis=0)
        ymax = tf.expand_dims(geometry_vectors[:, 3], axis=0)
        zmin = tf.expand_dims(geometry_vectors[:, 4], axis=0)
        zmax = tf.expand_dims(geometry_vectors[:, 5], axis=0)

        mask = tf.logical_and(tf.logical_and(x>=xmin, x<=xmax), tf.logical_and(tf.logical_and(y>=ymin, y<=ymax) , tf.logical_and(z>=zmin, z<=zmax)))
        indices = tf.argmax(tf.cast(mask, tf.int32), axis=1)
        indices = tf.cast(indices, dtype=tf.int32)
        return indices

# This function takes in a tensors of current material_IDs and previous material_IDs and returns a tensor of surface normal unit vectors.
# This code works for rectangular slabs, concentric spheres, or "concentric" cylinders (all materials must be the same type!)    
def getSurfaceNormals(currMatIDs, prevMatIDs, geometry_vectors, positions, mat_type):

    ray_current_geometry_vectors = tf.gather(geometry_vectors, currMatIDs)
    ray_previous_geometry_vectors = tf.gather(geometry_vectors, prevMatIDs)

# Cylinders:
# Note - This only works for two cylinders. The first cylinder must be at the origin.
    if mat_type == 'cylinder':
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

# Spheres:
    if mat_type == 'cylinder':
        mask = ray_previous_geometry_vectors[:, 4] >= ray_current_geometry_vectors[:, 4]
        mask = tf.expand_dims(mask, axis=1)
        greater_than = positions/(-1.0*tf.expand_dims(tf.norm(positions, axis=1), axis=1))
        less_than = positions/tf.expand_dims(tf.norm(positions, axis=1), axis=1)
        surface_normal = tf.where(mask, greater_than, less_than)
        return surface_normal

# Rectangular Prism Slabs:
    if mat_type == 'cylinder':
        mask = ray_current_geometry_vectors[:, 5] >= ray_previous_geometry_vectors[:, 5]
        mask = tf.expand_dims(mask, axis=1)
        length_positions = tf.shape(positions)[0]
        zhat = tf.constant([[0.0, 0.0, 1.0]])
        zhat = tf.repeat(zhat, length_positions, axis=0)
        minus_zhat = tf.constant([[0.0, 0.0, -1.0]])
        minus_zhat = tf.repeat(minus_zhat, length_positions, axis=0)
        surface_normal = tf.where(mask, zhat, minus_zhat) # If currentMat.zmax > prevMat.zmax, return [0,0,1], otherwise return [0,0,-1]
        return surface_normal

### ---------- Interface Analysis ---------- ###

# This function calculates a square root, but ensures that the output is real by converting any negative inputs to 1e-12 before taking the square root.
# This is used in the Isotropic-Isotropic interface function to prevents nans when the incident wave is above the critical angle for total internal reflection.
def safe_sqrt(arg):
    mask = tf.math.less(arg, 1e-12)
    epsilon = 1e-12 + tf.zeros_like(arg)
    safe_arg = tf.where(mask, epsilon, arg)
    return tf.math.sqrt(safe_arg)

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
    p_r = p_tn - (safe_sqrt(no1**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*surface_normal

    # We now need to calculate the wave normal associated with the transmitted wave in medium 2.
    p_t = p_tn + (safe_sqrt(no2**2 - tf.norm(p_tn, axis=1, keepdims=True)**2))*surface_normal

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

### ---------- Miscellaneous ---------- ###

# Given a number of FS coefficients, the period (centered about zero), and constants describing the initial guess,
# this function calculates and returns an equivalent fourier series representation of the er distribution.
# Note: This function can be updated for any choice of parameterization described by 'consts'.
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

# This function returns a list of target trajectories given a set of initial feed locations
# created using the function getCircFeedPoints(), and a given spacing in theta.
# This function is useful if using the 'specificPlaneWave()' loss function.
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

