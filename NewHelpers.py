import tensorflow as tf

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


# Given a tensor of ray positions and coefficients described the ordinary profile distribution for each ray,
# this function returns four rank 1 tensors: e_perp, deperp_dx, deperp_dy, and deperp_dz.
# Important Note: Each material must have a basis function of the same form (the coefficients can be different)
# Another important note: The permittivity distriubtion is passed into a "softplus" function to ensure er >= 1, always.
def getOrdinaryPermittivities(positions, ordinary_constants):

    # 4th-degree polynomial as a function of x:
    x = positions[:, 0] # for readability
    g = ordinary_constants[:, 0] + ordinary_constants[:, 1]*x + ordinary_constants[:, 2]*x**2 + ordinary_constants[:, 3]*x**3 + ordinary_constants[:, 4]*x**4 # g is an intermediate parameterization
    e_perp = 1 + tf.math.softplus(g)
    deperp_dx = tf.math.sigmoid(g)*(ordinary_constants[:, 1] + 2.0*ordinary_constants[:, 2]*x + 3.0*ordinary_constants[:, 3]*x**2 + 4.0*ordinary_constants[:, 4]*x**3)
    deperp_dy = tf.zeros(tf.shape(positions)[0])
    deperp_dz = tf.zeros(tf.shape(positions)[0])
    return e_perp, deperp_dx, deperp_dy, deperp_dz

# Similar to getOrdinaryPermittivities but for the extraordinary case.
def getExtraordinaryPermittivities(positions, extraordinary_constants):

    # 4th-degree polynomial as a function of x:
    x = positions[:, 0] # for readability
    g = extraordinary_constants[:, 0] + extraordinary_constants[:, 1]*x + extraordinary_constants[:, 2]*x**2 + extraordinary_constants[:, 3]*x**3 + extraordinary_constants[:, 4]*x**4 # g is an intermediate parameterization
    e_para = 1 + tf.math.softplus(g)
    depara_dx = tf.math.sigmoid(g)*(extraordinary_constants[:, 1] + 2.0*extraordinary_constants[:, 2]*x + 3.0*extraordinary_constants[:, 3]*x**2 + 4.0*extraordinary_constants[:, 4]*x**3)
    depara_dy = tf.zeros(tf.shape(positions)[0])
    depara_dz = tf.zeros(tf.shape(positions)[0])
    return e_para, depara_dx, depara_dy, depara_dz

# Similar to getOrdinaryPermittivites, however there are 10 items returned:
# The director profile (size 3 tensor), as well as the nine spatial derivatives (three for each director component).
def getDirector(positions, director_constants):

    # Constant x:
    dx = tf.ones(tf.shape(positions)[0])
    dy = tf.zeros(tf.shape(positions)[0])
    dz = tf.zeros(tf.shape(positions)[0])
    director = tf.stack([dx, dy, dz], axis=1)
    
    ddx_x = tf.zeros(tf.shape(positions)[0])
    ddx_y = tf.zeros(tf.shape(positions)[0])
    ddx_z = tf.zeros(tf.shape(positions)[0])
    ddy_x = tf.zeros(tf.shape(positions)[0])
    ddy_y = tf.zeros(tf.shape(positions)[0])
    ddy_z = tf.zeros(tf.shape(positions)[0])
    ddz_x = tf.zeros(tf.shape(positions)[0])
    ddz_y = tf.zeros(tf.shape(positions)[0])
    ddz_z = tf.zeros(tf.shape(positions)[0])

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
    # r = tf.math.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    # z = positions[:, 2]
    # alive = tf.logical_and(tf.logical_and(r<=boundingBox[1], r>=boundingBox[0]), tf.logical_and(z<=boundingBox[3], z>=boundingBox[2]))
    # alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
    # return alive
 
    # Spherical:
    # r = tf.math.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    # alive = r<=boundingBox[4]
    # alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
    # return alive

    # Rectangular Prism Slabs:
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    alive = tf.logical_and(tf.logical_and(x>=boundingBox[0], x<=boundingBox[1]), tf.logical_and(tf.logical_and(y>=boundingBox[2], y<=boundingBox[3]) , tf.logical_and(z>=boundingBox[4], z<=boundingBox[5])))
    alive = tf.logical_and(alive, current_alive) # To ensure previously "dead" rays do not because "alive" again
    return alive


# Given a tensor of positions and a tensor of material IDs (in the forms of integers), this function
# returns a tensor of booleans. True <==> New Material reached, False <==> same material.
def checkForHit(positions, material_IDs, geometry_vectors):
    new_material_IDs = getMaterialsAtCoordinates(positions, geometry_vectors)
    hit = tf.not_equal(material_IDs, new_material_IDs)
    return hit

# Given a tensor of positions (Nx3) and a list of geometry vectors, this function returns a tensor of integers corresponding 
# to the material associated with each position.
def getMaterialsAtCoordinates(positions, geometry_vectors):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    # Rectangular Prism Slabs:
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

    # Cylinders:
    # Note - This only works for two cylinders. The first cylinder must be at the origin.
    # r = tf.math.sqrt(x**2 + y**2)
    # cond = tf.logical_and(tf.logical_and(r>=geometry_vectors[0][0], r<=geometry_vectors[0][1]), tf.logical_and(z>=geometry_vectors[0][2], z<=geometry_vectors[0][3]))
    # cond = tf.cast(tf.logical_not(cond), dtype=tf.int32)
    # return cond

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
    mask = ray_current_geometry_vectors[:, 5] >= ray_previous_geometry_vectors[:, 5]
    mask = tf.expand_dims(mask, axis=1)
    length_positions = tf.shape(positions)[0]
    zhat = tf.constant([[0.0, 0.0, 1.0]])
    zhat = tf.repeat(zhat, length_positions, axis=0)
    minus_zhat = tf.constant([[0.0, 0.0, -1.0]])
    minus_zhat = tf.repeat(minus_zhat, length_positions, axis=0)
    surface_normal = tf.where(mask, zhat, minus_zhat) # If currentMat.zmax > prevMat.zmax, return [0,0,1], otherwise return [0,0,-1]
    return surface_normal
    
# Spheres:
    # mask = ray_previous_geometry_vectors[:, 4] >= ray_current_geometry_vectors[:, 4]
    # mask = tf.expand_dims(mask, axis=1)
    # greater_than = positions/(-1.0*tf.expand_dims(tf.norm(positions, axis=1), axis=1))
    # less_than = positions/tf.expand_dims(tf.norm(positions, axis=1), axis=1)
    # surface_normal = tf.where(mask, greater_than, less_than)
    # return surface_normal
    
# Cylinders:
# Note - This only works for two cylinders. The first cylinder must be at the origin.
    # mask1 = ray_previous_geometry_vectors[:, 1] >= ray_current_geometry_vectors[:, 1]
    # mask1 = tf.expand_dims(mask1, axis=1)
    
    # min_zdistance1 = tf.minimum(tf.math.abs(ray_current_geometry_vectors[:, 3] - positions[:, 2]), tf.math.abs(ray_current_geometry_vectors[:, 2] - positions[:, 2]))
    # min_zdistance2 = tf.minimum(tf.math.abs(ray_previous_geometry_vectors[:, 3] - positions[:, 2]), tf.math.abs(ray_previous_geometry_vectors[:, 2] - positions[:, 2]))
    # min_zdistance = tf.minimum(min_zdistance1, min_zdistance2)
    # min_rdistance1 = ray_current_geometry_vectors[:, 1] - tf.math.sqrt(tf.square(positions[:, 0]) + tf.square(positions[:, 1]))
    # min_rdistance2 = ray_previous_geometry_vectors[:, 1] - tf.math.sqrt(tf.square(positions[:, 0]) + tf.square(positions[:, 1]))
    # min_rdistance = tf.minimum(min_rdistance1, min_rdistance2)

    # mask2 = min_zdistance < min_rdistance
    # mask2 = tf.expand_dims(mask2, axis=1)
    # zhat = tf.stack([0.0*positions[:, 0], 0.0*positions[:, 1], positions[:, 2]], axis=1)
    # zhat = zhat/tf.expand_dims(tf.math.abs(positions[:, 2]), axis=1)
    # rhat = tf.stack([positions[:, 0], positions[:, 1], 0.0*positions[:, 2]], axis=1)
    # rhat = rhat/tf.expand_dims(tf.norm(rhat, axis=1), axis=1)

    # unit_vector = tf.where(mask2, zhat, rhat)
    # surface_normal = tf.where(mask1, -1.0*unit_vector, unit_vector)
    # return surface_normal

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

# Given a starting and ending x position, a fixed z_position, a number of rays, an incident angle (in degrees), and an incoming polarization, 
# this function generates all the ray tensors that will be added to the global ray tensors in the main algorithm:

def createStartingRays(NumberOfRays, starting_x, ending_x, fixed_z, angle, Epol, mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts):
    angle = angle*(.0174532925) # Convert the angle from degrees to radians
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

### ----------------------------- Everything below this line may need to be rewritten -------------------------------------------------------- ###

# This functions takes in a tensor of ray positions and a target focal point, and returns a size N
# tensor where for each ray the element represents the minimum distance SQUARED to the specified focal point.
def minDistanceToPoint(positions, focal_point):

    # Reshape focal point to (1, 3, 1) for broadcasting:
    focal_point = tf.reshape(focal_point, (1, 3, 1))

    # Calculate squared difference: (x-x0)^2, (y-y0)^2, (z-z0)^2
    squared_diff = tf.square(positions - focal_point) # Shape: (N, 3, SIZE)

    # Sum across coordinate dimension to get squared Euclidean distance:
    distance_squared = tf.reduce_sum(squared_diff, axis=1) # Shape: (N, SIZE)

    # Find the minimum squared distance across time (axis 1):
    min_dist_sq = tf.reduce_min(distance_squared, axis=1) # Shape: (N,)

    return min_dist_sq

# This objective function rewards rays for passing nearby a specified focal point.
# It is converted to a loss function by making the reward negative.

def focusObjective(positions, focal_point, material_IDs):
    
    # Only consider rays propagating in the air past the lens.
    # A boolean mask must be applied to positions to achieve this:
    material_mask = material_IDs == 2

    # For each ray, calculate the minimum distance squared to the focal point:
    min_dist_sq = minDistanceToPoint(positions, focal_point)

    # Apply the mask (if material_ID is not 2, the min distance squared will be set to zero):
    masked_dist_sq = tf.where(material_mask, min_dist_sq, tf.zeros_like(min_dist_sq))

    # Compute loss factor by summing over the minimum distances squared (note that the optimal loss is zero):
    loss = 10000*tf.reduce_sum(masked_dist_sq, axis=0)

    return loss
