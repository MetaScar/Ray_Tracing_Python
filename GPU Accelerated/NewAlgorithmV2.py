import tensorflow as tf
import NewHelpers as nh
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()

# This ray-tracing algorithm attempts to perform all tracing of rays in parallel to leverage GPU acceleration.
# THIS VERSION (V2) REWRITES MUCH OF THE LOGIC TO ENSURE THE CODE IS DIFFERENTIABLE!!!
# THIS IS THE MOST UP-TO-DATE VERSION OF THE CODE.

# Static Constants:
Max_Rays = 450 # Variable to keep track of the max number of rays allowed.
SIZE = 200 # Constant size of arrays to store ray positions, wave normals, and electric fields.
distance_step = tf.constant(0.05) # Fixed distance step for propagating rays
minimum_Poynting = tf.constant(-1.0) # Any ray with a Poynting Vector magnitude less than this value will not be "launched" when creating new rays at an interface
Num_constants = tf.constant(56, dtype=tf.int32) # Number of constants needed to parameterize the materials (permittivity + director profile)
Num_Materials = tf.constant(1, dtype=tf.int32) # Size of the number of materials in the simulation (e.g. 2 ==> 3 materials)
IGNORE_REFLECTIONS = True # True causes no reflected rays to spawn, False causes reflected rays to spawn as normal.

# Variables with History Storage (tf.TensorArrays):
positions_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays, 3], clear_after_read=False)
wave_vectors_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays, 3], clear_after_read=False)
Efields_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays, 3], clear_after_read=False)

# Current State Tensors (standard tensors):
# The values of these tensors are overwritten during the loop.
alive = tf.zeros([Max_Rays], dtype=tf.bool)
PoyntingMag = tf.zeros([Max_Rays], dtype=tf.float32)
ordinary = tf.zeros([Max_Rays], dtype=tf.bool)
material_IDs = tf.zeros([Max_Rays], dtype=tf.int32)

# Material parameter tensors (also standard tensors):
ray_ordinary_consts = tf.zeros([Max_Rays, Num_constants], dtype=tf.float32)
ray_extraordinary_consts = tf.zeros([Max_Rays, Num_constants], dtype=tf.float32)
ray_director_consts = tf.zeros([Max_Rays, Num_constants], dtype=tf.float32)

# Loop counters (int32 tensors):
step = tf.constant(1, dtype=tf.int32) # This should always be initialized to one
Num_rays_active = tf.constant(100, dtype=tf.int32) # This should be intialized to the desired # of input starting rays
Num_Starting_Rays = Num_rays_active

# Initialize material tensors:
geometry_vectors = tf.constant([[0.0, 8.0, -2.5, 2.5], [8.0, 10.0, -5.0, 5.0]])
isotropic = tf.constant([True, True], dtype=tf.bool)

# Find materialCoef FS coeffcients from initial polynomial guess:
# materialCoefs_lens = tf.stack([nh.generateFourierCoefs(40, 16.0, [0.5, 10.0, 20.0]), nh.generateFourierCoefs(40, 5.0, [0.5, 5.0, 40.0])], axis=0) # (N, T0, [emax, rho1, alpha])
# materialCoefs_lens = tf.reshape(materialCoefs_lens, [-1])

# materialCoefs_air = tf.stack([nh.generateFourierCoefs(20, 8.0, [-100.0, 10.0, 0.0]), nh.generateFourierCoefs(20, 8.0, [-100.0, 10.0, 0.0])], axis=0)  # Results in a flat er = 1.
# materialCoefs_air = tf.reshape(materialCoefs_air, [-1]) # Results in a flat er = 1.

# Lens coefficients:
erb = 1.5
e_max = 5.8
rho_max = 8.0 # lens radius
alpha = 4.5
C = 17.4
zmin = -40.0
zmax = 40.0
T0 = 16.0

# FS_coeffs = tf.constant([-2.9000, 10.577966442033571,	-2.644517763045427,	1.175360600576376,	-0.661155594074306,	0.423152134319206,	-0.293866304750302,	0.215911027620145,	-0.165315054935490,	0.130626623097328,	-0.105814192325012,	0.087455800136030,	-0.073492737778857,	0.062626192441672,	-0.054003921860388,	0.047047914437841,	-0.041354932571561,	0.036636728995691,	-0.032682829012930,	0.029336644123710,	-0.026479726239681,	0.024021139533335,	-0.021890133631576,	0.020030996147535,	-0.018399374001172,	0.016959603367423,	-0.015682744145981,	0.014545114240363,	-0.013527183500485,	0.012612729890533,	-0.011788189165964,	0.011042148957168,	-0.010364951742390,	0.009748380726371,	-0.009185409413939,	0.008670000534393,	-0.008196943506961,	0.007761722230117,	-0.007360406897294,	0.006989564975535,	-0.006646187563613,	0.006327628166478,	-0.006031551549847,	0.005755890821855,	-0.005498811263561,	0.005258679721952,	-0.005034038608945,	0.004823583730891,	-0.004626145316923,	0.004440671729517,	-0.004266215432190])

# materialCoefs_lens = tf.concat([[erb, C, zmin, zmax, T0], FS_coeffs], axis=0)

materialCoefs_lens = tf.constant([1.7000000e+00,  1.7400000e+01, -4.0000000e+01,  4.0000000e+01,
        1.6000000e+01, -2.8304734e+00,  1.0496612e+01, -2.7222948e+00,
        1.0993505e+00, -7.3412341e-01,  3.7208858e-01, -2.7466777e-01,
        2.5886971e-01, -1.2727737e-01,  1.4478713e-01, -1.5068257e-01,
        5.5423234e-02, -8.0074303e-02,  8.6546898e-02, -1.9548669e-02,
        5.4926880e-02, -6.6027276e-02,  7.1996865e-03, -3.4953032e-02,
        5.1373281e-02, -9.2328498e-03,  2.9523453e-02, -4.0450893e-02,
        6.9585219e-03, -2.0507148e-02,  2.5666401e-02, -4.2328723e-03,
        2.0767948e-02, -2.0752003e-02,  3.8161612e-04, -1.6458375e-02,
        1.5035616e-02,  2.0771390e-03,  1.5684143e-02, -1.6479021e-02,
        2.5884548e-05, -1.2941909e-02,  1.1724184e-02,  1.5739516e-03,
        1.2016340e-02, -1.0893514e-02, -2.2598682e-03, -1.2370811e-02,
        1.4079228e-02,  1.9118408e-03,  7.6199574e-03, -1.2536882e-02,
       -4.7228960e-03,  2.2023106e-03,  1.1981796e-02, -7.9207700e-03])


materialCoefs_air = tf.concat([[1.0, C, zmin, zmax, T0, -1000.0], tf.zeros(50, dtype=tf.float32)], axis=0)

# materialCoefs_lens = [erb, e_max, rho_max, alpha, C, zmin, zmax]
# materialCoefs_air = [1.0, -1000.0, 8.0, 0.0, C, zmin, zmax]

mat_ordinary_consts = tf.stack([materialCoefs_lens, materialCoefs_air])
mat_extraordinary_consts = tf.stack([[3.5, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
mat_director_consts = tf.stack([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

# Initialize Bounding Box:
boundingBox = tf.constant([0.0, 10.0, -5.0, 5.0])

# Find circular feed locations:
center_location = tf.constant([0.0000001, 0.0000001, -3.5])
dx = 0.5
dy = 0.5
Nx = 1
Ny = 1
initialPoints = nh.getRectFeedPoints(center_location, dx, dy, Nx, Ny)
numFeeds = tf.shape(initialPoints)[0]

# Specify desired target beam trajectories for each feed (### CURRENTLY NOT USING THIS FEATURE)
# targetTrajects = nh.getTargetTrajects(center_location, dr, dphi, Nrings, theta_spacing_deg=15)

# Specify other variables that describe initial rays:
theta_max_deg = 35.0
theta_target_deg = 0.0
Epol = tf.constant([0.0, 1.0, 0.0])

# Vectorized versions of these variables:
theta_max_deg = tf.tile([theta_max_deg], [numFeeds])
theta_target_deg = tf.tile([theta_target_deg], [numFeeds])
Epol = tf.expand_dims(Epol, axis=0)
Epol = tf.tile(Epol, [numFeeds, 1])
Num_Rays = tf.tile([Num_Starting_Rays/numFeeds], [numFeeds])
group_IDs_initial = 1 + tf.range(numFeeds)
group_IDs_initial = tf.repeat(group_IDs_initial, repeats=tf.cast(Num_Starting_Rays/numFeeds, tf.int32)) # Repeats are the number of rays per feed
parent_IDs_initial = 1 + tf.range(Num_Starting_Rays) # This just assigns a unique ID to each ray, which is used for tracking rays through the system and visualizing ray trees.

# Create initial rays to launch into system (isotropic point source):
# Call createIsotropicRays():
positions_initial, wave_vectors_initial, PoyntingMag_initial, alive_initial, Efields_initial, ordinary_initial, material_IDs_initial, ray_ordinary_consts_initial, ray_extraordinary_consts_initial, ray_director_consts_initial = nh.createIsotropicRays(Num_Rays, theta_max_deg, theta_target_deg, initialPoints, Epol, mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts)

# positions_initial2, wave_vectors_initial2, PoyntingMag_initial2, alive_initial2, Efields_initial2, ordinary_initial2, material_IDs_initial2, ray_ordinary_consts_initial2, ray_extraordinary_consts_initial2, ray_director_consts_initial2, group_IDs_initial2 = nh.createIsotropicRays(Num_Rays[1], theta_max_degs[1], theta_target_degs[1], sphere_centers[1], Epol[1], mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts, group_ID=2)
# positions_initial3, wave_vectors_initial3, PoyntingMag_initial3, alive_initial3, Efields_initial3, ordinary_initial3, material_IDs_initial3, ray_ordinary_consts_initial3, ray_extraordinary_consts_initial3, ray_director_consts_initial3, group_IDs_initial3 = nh.createIsotropicRays(Num_Rays[2], theta_max_degs[2], theta_target_degs[2], sphere_centers[2], Epol[2], mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts, group_ID=3)
# positions_initial4, wave_vectors_initial4, PoyntingMag_initial4, alive_initial4, Efields_initial4, ordinary_initial4, material_IDs_initial4, ray_ordinary_consts_initial4, ray_extraordinary_consts_initial4, ray_director_consts_initial4, group_IDs_initial4 = nh.createIsotropicRays(Num_Rays[3], theta_max_degs[3], theta_target_degs[3], sphere_centers[3], Epol[3], mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts, group_ID=4)

# positions_initial = tf.concat([positions_initial1, positions_initial2, positions_initial3, positions_initial4], axis=0)
# wave_vectors_initial = tf.concat([wave_vectors_initial1, wave_vectors_initial2, wave_vectors_initial3, wave_vectors_initial4], axis=0)
# PoyntingMag_initial = tf.concat([PoyntingMag_initial1, PoyntingMag_initial2, PoyntingMag_initial3, PoyntingMag_initial4], axis=0)
# alive_initial = tf.concat([alive_initial1, alive_initial2, alive_initial3, alive_initial4], axis=0)
# Efields_initial = tf.concat([Efields_initial1, Efields_initial2, Efields_initial3, Efields_initial4], axis=0)
# ordinary_initial = tf.concat([ordinary_initial1, ordinary_initial2, ordinary_initial3, ordinary_initial4], axis=0)
# material_IDs_initial = tf.concat([material_IDs_initial1, material_IDs_initial2, material_IDs_initial3, material_IDs_initial4], axis=0)
# ray_ordinary_consts_initial = tf.concat([ray_ordinary_consts_initial1, ray_ordinary_consts_initial2, ray_ordinary_consts_initial3, ray_ordinary_consts_initial4], axis=0)
# ray_extraordinary_consts_initial = tf.concat([ray_extraordinary_consts_initial1, ray_extraordinary_consts_initial2, ray_extraordinary_consts_initial3, ray_extraordinary_consts_initial4], axis=0)
# ray_director_consts_initial = tf.concat([ray_director_consts_initial1, ray_director_consts_initial2, ray_director_consts_initial3, ray_director_consts_initial4], axis=0)
# group_IDs_initial = tf.concat([group_IDs_initial1, group_IDs_initial2, group_IDs_initial3, group_IDs_initial4], axis=0)

# Create initial rays to launch into system (plane wave):
# angle = tf.constant(0.0)
# starting_x = tf.constant(-1.0)
# ending_x = tf.constant(1.0)
# fixed_z = tf.constant(-18.0)
# Epol = tf.constant([0.0, 0.0, 1.0])

# positions_initial, wave_vectors_initial, PoyntingMag_initial, alive_initial, Efields_initial, ordinary_initial, material_IDs_initial, ray_ordinary_consts_initial, ray_extraordinary_consts_initial, ray_director_consts_initial = nh.createStartingRays(Num_rays_active, starting_x, ending_x, fixed_z, angle, Epol, mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts)

# PAD/INSERT into Max_Rays tensors:
# Use tf.pad() to take N active rays and fill the rest of the slots with zeros
paddings = [[0, Max_Rays - Num_rays_active], [0, 0]]
paddings_1d = [[0, Max_Rays - Num_rays_active]]

# Update state tensors:
alive = tf.pad(alive_initial, paddings_1d)
PoyntingMag = tf.pad(PoyntingMag_initial, paddings_1d)
ordinary = tf.pad(ordinary_initial, paddings_1d)
material_IDs = tf.pad(material_IDs_initial, paddings_1d)
ray_ordinary_consts = tf.pad(ray_ordinary_consts_initial, paddings)
ray_extraordinary_consts = tf.pad(ray_extraordinary_consts_initial, paddings)
ray_director_consts = tf.pad(ray_director_consts_initial, paddings)
group_IDs = tf.pad(group_IDs_initial, paddings_1d)
parent_IDs = tf.pad(parent_IDs_initial, paddings_1d)

# Initialize the TensorArrays using TensorArray.write():
positions_ta = positions_ta.write(0, tf.pad(positions_initial, paddings))
wave_vectors_ta = wave_vectors_ta.write(0, tf.pad(wave_vectors_initial, paddings))
Efields_ta = Efields_ta.write(0, tf.pad(Efields_initial, paddings))

# Main ray-tracing logic:
while tf.reduce_any(alive): # While there is at least one True in the "alive" tensor

    # Preserve the previous alive mask so rays that died on this step can keep their last position,
    # while rays that were already dead remain zeroed out in future history steps.
    alive_before = alive

    # 1) Perform a step of ray propagation:

    # Read the state of wave_vectors, positions, and Efields from the previous step (step - 1):
    curr_pos = positions_ta.read(step-1)
    curr_wave = wave_vectors_ta.read(step-1)
    curr_E = Efields_ta.read(step-1)

    # Call the rayPropagationStep function:
    new_wave_vectors, new_positions, new_Efields = nh.rayPropagation(curr_pos, curr_wave, curr_E, ordinary, alive, ray_ordinary_consts, ray_extraordinary_consts, ray_director_consts, isotropic, material_IDs, distance_step)

    # Zero out any rays that were already dead before this step, so their later history does not persist.
    already_dead = tf.logical_not(alive_before)
    already_dead_expanded = tf.expand_dims(already_dead, axis=1)
    new_positions = tf.where(already_dead_expanded, tf.zeros_like(new_positions), new_positions)
    new_wave_vectors = tf.where(already_dead_expanded, tf.zeros_like(new_wave_vectors), new_wave_vectors)
    new_Efields = tf.where(already_dead_expanded, tf.zeros_like(new_Efields), new_Efields)
    
    # Update the TensorArrays for positions, wave vectors, and Electric fields:
    positions_ta = positions_ta.write(step, new_positions)
    wave_vectors_ta = wave_vectors_ta.write(step, new_wave_vectors)
    Efields_ta = Efields_ta.write(step, new_Efields)

    # 2) Check for rays that have left the boundingBox... if yes, set their alive to False
    alive = nh.checkBoundary(new_positions, boundingBox, alive)

    # 3) Check for rays that have reached a new material, and assign their alive to False. Also, rays that have reached out of bounds should not be included in newMatMask.
    newMatMask = nh.checkForHit(new_positions, material_IDs, geometry_vectors) ### ERROR FOR CYLINDRICAL GEOMETRY HERE
    newMatMask = tf.logical_and(newMatMask, alive)
    alive = tf.logical_and(alive, tf.logical_not(newMatMask))

    # 4) Perform interface analysis:

    # Mask all variables required for interface analysis, corresponding to rays that have hit a new material.
    # Also, for positions, wave_vectors, and Efields, only the last entry of each ray is kept:
    positions_lastStep = positions_ta.read(step)
    positions_newMat = tf.boolean_mask(positions_lastStep, newMatMask)
    p_lastStep = wave_vectors_ta.read(step)
    p_newMat = tf.boolean_mask(p_lastStep, newMatMask)
    Efields_lastStep = Efields_ta.read(step)
    Efields_newMat = tf.boolean_mask(Efields_lastStep, newMatMask)
    PoyntingMag_newMat = tf.boolean_mask(PoyntingMag, newMatMask)
    ordinary_newMat = tf.boolean_mask(ordinary, newMatMask)
    material_IDs_newMat = tf.boolean_mask(material_IDs, newMatMask)
    group_IDs_newMat = tf.boolean_mask(group_IDs, newMatMask)
    parent_IDs_newMat = tf.boolean_mask(parent_IDs, newMatMask)
    ray_ordinary_consts_newMat = tf.boolean_mask(ray_ordinary_consts, newMatMask)
    ray_extraordinary_consts_newMat = tf.boolean_mask(ray_ordinary_consts, newMatMask)
    ray_director_consts_newMat = tf.boolean_mask(ray_director_consts, newMatMask)

    # Compute four seperate boolean masks for each possible type of interface,
    # including Isotropic-Isotropic, Isotropic-Anisotropic, Anisotropic-Isotropic, and Anisotropic-Anisotropic.
    newMat_IDs = nh.getMaterialsAtCoordinates(positions_newMat, geometry_vectors) # This is a list of the material IDs of the new materials that have just been reached.

    prev_Mat_types = tf.gather(isotropic, material_IDs_newMat)
    new_Mat_types = tf.gather(isotropic, newMat_IDs)

    Iso_Iso_mask = tf.logical_and(prev_Mat_types, new_Mat_types) # Recall true True ==> isotropic and False ==> anisotropic...
    Iso_Aniso_mask = tf.logical_and(prev_Mat_types, tf.logical_not(new_Mat_types))
    Aniso_Iso_mask = tf.logical_and(tf.logical_not(prev_Mat_types), new_Mat_types)
    Aniso_Aniso_mask = tf.logical_and(tf.logical_not(prev_Mat_types), tf.logical_not(new_Mat_types))

    # Performing Isotropic_Isotropic interface analysis...
    # Create a Iso-Iso masked version of all variables:
    positions_Iso_Iso = tf.boolean_mask(positions_newMat, Iso_Iso_mask)
    p_Iso_Iso = tf.boolean_mask(p_newMat, Iso_Iso_mask)
    Efields_Iso_Iso = tf.boolean_mask(Efields_newMat, Iso_Iso_mask)
    PoyntingMag_Iso_Iso = tf.boolean_mask(PoyntingMag_newMat, Iso_Iso_mask)
    ray_ordinary_consts_Iso_Iso = tf.boolean_mask(ray_ordinary_consts_newMat, Iso_Iso_mask)
    ray_extraordinary_consts_Iso_Iso = tf.boolean_mask(ray_extraordinary_consts_newMat, Iso_Iso_mask)
    ray_director_consts_Iso_Iso = tf.boolean_mask(ray_director_consts_newMat, Iso_Iso_mask)
    group_IDs_Iso_Iso = tf.boolean_mask(group_IDs_newMat, Iso_Iso_mask)
    parent_IDs_Iso_Iso = tf.boolean_mask(parent_IDs_newMat, Iso_Iso_mask)
    prev_Mat_types_Iso_Iso = tf.boolean_mask(prev_Mat_types, Iso_Iso_mask)
    new_Mat_types_Iso_Iso = tf.boolean_mask(new_Mat_types, Iso_Iso_mask)
    prev_MatIDs_Iso_Iso = tf.boolean_mask(material_IDs_newMat, Iso_Iso_mask)
    new_MatIDs_Iso_Iso = tf.boolean_mask(newMat_IDs, Iso_Iso_mask)

    # Setting up and calling the Isotropic-Isotropic interface analysis function:
    surface_normals = nh.getSurfaceNormals(new_MatIDs_Iso_Iso, prev_MatIDs_Iso_Iso, geometry_vectors, positions_Iso_Iso)
    e_perp1, _, _, _ = nh.getOrdinaryPermittivities(positions_Iso_Iso, ray_ordinary_consts_Iso_Iso)
    no1 = tf.math.sqrt(e_perp1)
    mat2_ordinary_consts = tf.gather(mat_ordinary_consts, new_MatIDs_Iso_Iso)
    e_perp2, _, _, _ = nh.getOrdinaryPermittivities(positions_Iso_Iso, mat2_ordinary_consts)
    no2 = tf.math.sqrt(e_perp2)

    p_rII, p_tII, E_rII, E_tII, S_rII, S_tII = nh.Isotropic_Isotropic(surface_normals, no1, no2, p_Iso_Iso, Efields_Iso_Iso)
    S_rII = S_rII*PoyntingMag_Iso_Iso # Multiply the normalized Poynting vectors by the Poynting Vector magnitude of the corresponding incident ray
    S_tII = S_tII*PoyntingMag_Iso_Iso

    # Initialize the rest of the variables for the new reflected and transmitted rays:
    new_positions_reflectedII = positions_Iso_Iso
    new_positions_transmittedII = positions_Iso_Iso
    # new_alive = tf.ones(2*tf.shape(new_positions)[0], dtype=tf.bool)
    new_materialIDs_reflectedII = prev_MatIDs_Iso_Iso
    new_materialIDs_transmittedII = new_MatIDs_Iso_Iso
    group_IDs_reflectedII = group_IDs_Iso_Iso
    group_IDs_transmittedII = group_IDs_Iso_Iso
    parent_IDs_reflectedII = parent_IDs_Iso_Iso
    parent_IDs_transmittedII = parent_IDs_Iso_Iso
    ray_ordinary_consts_reflectedII = tf.gather(mat_ordinary_consts, new_materialIDs_reflectedII)
    ray_ordinary_consts_transmittedII = tf.gather(mat_ordinary_consts, new_materialIDs_transmittedII)
    ray_extraordinary_consts_reflectedII = tf.gather(mat_extraordinary_consts, new_materialIDs_reflectedII)
    ray_extraordinary_consts_transmittedII= tf.gather(mat_extraordinary_consts, new_materialIDs_transmittedII)
    ray_director_consts_reflectedII = tf.gather(mat_director_consts, new_materialIDs_reflectedII)
    ray_director_consts_transmittedII = tf.gather(mat_director_consts, new_materialIDs_transmittedII)
    ordinary_reflectedII = tf.ones(tf.shape(new_positions_reflectedII)[0], dtype=tf.bool)
    ordinary_transmittedII = tf.ones(tf.shape(new_positions_reflectedII)[0], dtype=tf.bool)

    # Performing Isotropic_Anisotropic interface analysis...
    # Create an Iso_Aniso masked version of all variables:
    positions_Iso_Aniso = tf.boolean_mask(positions_newMat, Iso_Aniso_mask)
    p_Iso_Aniso = tf.boolean_mask(p_newMat, Iso_Aniso_mask)
    Efields_Iso_Aniso = tf.boolean_mask(Efields_newMat, Iso_Aniso_mask)
    PoyntingMag_Iso_Aniso = tf.boolean_mask(PoyntingMag_newMat, Iso_Aniso_mask)
    ray_ordinary_consts_Iso_Aniso = tf.boolean_mask(ray_ordinary_consts_newMat, Iso_Aniso_mask)
    ray_extraordinary_consts_Iso_Aniso = tf.boolean_mask(ray_extraordinary_consts_newMat, Iso_Aniso_mask)
    ray_director_consts_Iso_Aniso = tf.boolean_mask(ray_director_consts_newMat, Iso_Aniso_mask)
    group_IDs_Iso_Aniso = tf.boolean_mask(group_IDs_newMat, Iso_Aniso_mask)
    parent_IDs_Iso_Aniso = tf.boolean_mask(parent_IDs_newMat, Iso_Aniso_mask)
    prev_Mat_types_Iso_Aniso = tf.boolean_mask(prev_Mat_types, Iso_Aniso_mask)
    new_Mat_types_Iso_Aniso = tf.boolean_mask(new_Mat_types, Iso_Aniso_mask)
    prev_MatIDs_Iso_Aniso = tf.boolean_mask(material_IDs_newMat, Iso_Aniso_mask)
    new_MatIDs_Iso_Aniso = tf.boolean_mask(newMat_IDs, Iso_Aniso_mask)

    # Setting up and calling the Isotropic-Anisotropic interface analysis function:
    surface_normals = nh.getSurfaceNormals(new_MatIDs_Iso_Aniso, prev_MatIDs_Iso_Aniso, geometry_vectors, positions_Iso_Aniso)
    e_perp1, _, _, _ = nh.getOrdinaryPermittivities(positions_Iso_Aniso, ray_ordinary_consts_Iso_Aniso)
    no1 = tf.math.sqrt(e_perp1)
    mat2_ordinary_consts = tf.gather(mat_ordinary_consts, new_MatIDs_Iso_Aniso)
    e_perp2, _, _, _ = nh.getOrdinaryPermittivities(positions_Iso_Aniso, mat2_ordinary_consts)
    no2 = tf.math.sqrt(e_perp2)
    mat2_extraordinary_consts = tf.gather(mat_extraordinary_consts, new_MatIDs_Iso_Aniso)
    e_para2, _, _, _ = nh.getExtraordinaryPermittivities(positions_Iso_Aniso, mat2_extraordinary_consts)
    ne2 = tf.math.sqrt(e_para2)
    mat2_director_consts = tf.gather(mat_director_consts, new_MatIDs_Iso_Aniso)
    optical_axis2, _, _, _, _, _, _, _, _, _ = nh.getDirector(positions_Iso_Aniso, mat2_director_consts)

    p_rIA, p_toIA, p_teIA, E_rIA, E_toIA, E_teIA, S_rIA, S_toIA, S_teIA = nh.Isotropic_Anisotropic(surface_normals, optical_axis2, no1, no2, ne2, p_Iso_Aniso, Efields_Iso_Aniso)
    S_rIA = S_rIA*PoyntingMag_Iso_Aniso # Multiply the normalized Poynting vectors by the Poynting Vector magnitude of the corresponding incident ray
    S_toIA = S_toIA*PoyntingMag_Iso_Aniso
    S_teIA = S_teIA*PoyntingMag_Iso_Aniso

    # Initialize the rest of the variables for the new reflected and transmitted rays:
    new_positions_reflectedIA = positions_Iso_Aniso
    new_positions_transmitted_ordinaryIA = positions_Iso_Aniso
    new_positions_transmitted_extraordinaryIA = positions_Iso_Aniso
    new_materialIDs_reflectedIA = prev_MatIDs_Iso_Aniso
    new_materialIDs_transmitted_ordinaryIA = new_MatIDs_Iso_Aniso
    new_materialIDs_transmitted_extraordinaryIA = new_MatIDs_Iso_Aniso

    group_IDs_reflectedIA = group_IDs_Iso_Aniso
    group_IDs_transmitted_ordinaryIA = group_IDs_Iso_Aniso
    group_IDs_transmitted_extraordinaryIA = group_IDs_Iso_Aniso

    parent_IDs_reflectedIA = parent_IDs_Iso_Aniso
    parent_IDs_transmitted_ordinaryIA = parent_IDs_Iso_Aniso
    parent_IDs_transmitted_extraordinaryIA = parent_IDs_Iso_Aniso

    ray_ordinary_consts_reflectedIA = tf.gather(mat_ordinary_consts, new_materialIDs_reflectedIA)
    ray_ordinary_consts_transmitted_ordinaryIA = tf.gather(mat_ordinary_consts, new_materialIDs_transmitted_ordinaryIA)
    ray_ordinary_consts_transmitted_extraordinaryIA = tf.gather(mat_ordinary_consts, new_materialIDs_transmitted_extraordinaryIA)

    ray_extraordinary_consts_reflectedIA = tf.gather(mat_extraordinary_consts, new_materialIDs_reflectedIA)
    ray_extraordinary_consts_transmitted_ordinaryIA = tf.gather(mat_extraordinary_consts, new_materialIDs_transmitted_ordinaryIA)
    ray_extraordinary_consts_transmitted_extraordinaryIA = tf.gather(mat_extraordinary_consts, new_materialIDs_transmitted_extraordinaryIA)

    ray_director_consts_reflectedIA = tf.gather(mat_director_consts, new_materialIDs_reflectedIA)
    ray_director_consts_transmitted_ordinaryIA = tf.gather(mat_director_consts, new_materialIDs_transmitted_ordinaryIA)
    ray_director_consts_transmitted_extraordinaryIA = tf.gather(mat_director_consts, new_materialIDs_transmitted_extraordinaryIA)

    ordinary_reflectedIA = tf.ones(tf.shape(new_positions_reflectedIA)[0], dtype=tf.bool)
    ordinary_transmitted_ordinaryIA = tf.ones(tf.shape(new_positions_reflectedIA)[0], dtype=tf.bool)
    ordinary_transmitted_extraordinaryIA = tf.zeros(tf.shape(new_positions_reflectedIA)[0], dtype=tf.bool)

    # Performing Anistropic_Isotropic Interface analysis...
    # Creating an Aniso_Iso masked version of all variables:
    positions_Aniso_Iso = tf.boolean_mask(positions_newMat, Aniso_Iso_mask)
    p_Aniso_Iso = tf.boolean_mask(p_newMat, Aniso_Iso_mask)
    Efields_Aniso_Iso = tf.boolean_mask(Efields_newMat, Aniso_Iso_mask)
    PoyntingMag_Aniso_Iso = tf.boolean_mask(PoyntingMag_newMat, Aniso_Iso_mask)
    ray_ordinary_consts_Aniso_Iso = tf.boolean_mask(ray_ordinary_consts_newMat, Aniso_Iso_mask)
    ray_extraordinary_consts_Aniso_Iso = tf.boolean_mask(ray_extraordinary_consts_newMat, Aniso_Iso_mask)
    ray_director_consts_Aniso_Iso = tf.boolean_mask(ray_director_consts_newMat, Aniso_Iso_mask)
    group_IDs_Aniso_Iso = tf.boolean_mask(group_IDs_newMat, Aniso_Iso_mask)
    parent_IDs_Aniso_Iso = tf.boolean_mask(parent_IDs_newMat, Aniso_Iso_mask)
    prev_Mat_types_Aniso_Iso = tf.boolean_mask(prev_Mat_types, Aniso_Iso_mask)
    new_Mat_types_Aniso_Iso = tf.boolean_mask(new_Mat_types, Aniso_Iso_mask)
    prev_MatIDs_Aniso_Iso = tf.boolean_mask(material_IDs_newMat, Aniso_Iso_mask)
    new_MatIDs_Aniso_Iso = tf.boolean_mask(newMat_IDs, Aniso_Iso_mask)

    # Setting up and calling the Isotropic-Anisotropic interface analysis function:
    surface_normals = nh.getSurfaceNormals(new_MatIDs_Aniso_Iso, prev_MatIDs_Aniso_Iso, geometry_vectors, positions_Aniso_Iso)
    e_perp1, _, _, _ = nh.getOrdinaryPermittivities(positions_Aniso_Iso, ray_ordinary_consts_Aniso_Iso)
    no1 = tf.math.sqrt(e_perp1)
    e_para1, _, _, _ = nh.getExtraordinaryPermittivities(positions_Aniso_Iso, ray_extraordinary_consts_Aniso_Iso)
    ne1 = tf.math.sqrt(e_para1)
    optical_axis1, _, _, _, _, _, _, _, _, _ = nh.getDirector(positions_Aniso_Iso, ray_director_consts_Aniso_Iso)
    mat2_ordinary_consts = tf.gather(mat_ordinary_consts, new_MatIDs_Aniso_Iso)
    e_perp2, _, _, _ = nh.getOrdinaryPermittivities(positions_Aniso_Iso, mat2_ordinary_consts)
    no2 = tf.math.sqrt(e_perp2)

    p_roAI, p_reAI, p_tAI, E_roAI, E_reAI, E_tAI, S_roAI, S_reAI, S_tAI = nh.Anisotropic_Isotropic(surface_normals, optical_axis1, no1, ne1, no2, p_Aniso_Iso, Efields_Aniso_Iso)
    S_roAI = S_roAI*PoyntingMag_Aniso_Iso # Multiply the normalized Poynting vectors by the Poynting Vector magnitude of the corresponding incident ray
    S_reAI = S_reAI*PoyntingMag_Aniso_Iso
    S_tAI = S_tAI*PoyntingMag_Aniso_Iso

    # Initialize the rest of the variables for the new reflected and transmitted rays:
    new_positions_reflected_ordinaryAI = positions_Aniso_Iso
    new_positions_reflected_extraordinaryAI = positions_Aniso_Iso
    new_positions_transmittedAI = positions_Aniso_Iso
    
    new_materialIDs_reflected_ordinaryAI = prev_MatIDs_Aniso_Iso
    new_materialIDs_reflected_extraordinaryAI = prev_MatIDs_Aniso_Iso
    new_materialIDs_transmittedAI = new_MatIDs_Aniso_Iso

    group_IDs_reflected_ordinaryAI = group_IDs_Aniso_Iso
    group_IDs_reflected_extraordinaryAI = group_IDs_Aniso_Iso
    group_IDs_transmittedAI = group_IDs_Aniso_Iso

    parent_IDs_reflected_ordinaryAI = parent_IDs_Aniso_Iso
    parent_IDs_reflected_extraordinaryAI = parent_IDs_Aniso_Iso
    parent_IDs_transmittedAI = parent_IDs_Aniso_Iso
    
    ray_ordinary_consts_reflected_ordinaryAI = tf.gather(mat_ordinary_consts, new_materialIDs_reflected_ordinaryAI)
    ray_ordinary_consts_reflected_extraordinaryAI = tf.gather(mat_ordinary_consts, new_materialIDs_reflected_extraordinaryAI)
    ray_ordinary_consts_transmittedAI = tf.gather(mat_ordinary_consts, new_materialIDs_transmittedAI)

    ray_extraordinary_consts_reflected_ordinaryAI = tf.gather(mat_extraordinary_consts, new_materialIDs_reflected_ordinaryAI)
    ray_extraordinary_consts_reflected_extraordinaryAI = tf.gather(mat_extraordinary_consts, new_materialIDs_reflected_extraordinaryAI)
    ray_extraordinary_consts_transmittedAI = tf.gather(mat_extraordinary_consts, new_materialIDs_transmittedAI)

    ray_director_consts_reflected_ordinaryAI = tf.gather(mat_director_consts, new_materialIDs_reflected_ordinaryAI)
    ray_director_consts_reflected_extraordinaryAI = tf.gather(mat_director_consts, new_materialIDs_reflected_extraordinaryAI)
    ray_director_consts_transmittedAI = tf.gather(mat_director_consts, new_materialIDs_transmittedAI)

    ordinary_reflected_ordinaryAI = tf.ones(tf.shape(new_positions_reflected_ordinaryAI)[0], dtype=tf.bool)
    ordinary_reflected_extraordinaryAI = tf.zeros(tf.shape(new_positions_reflected_extraordinaryAI)[0], dtype=tf.bool)
    ordinary_transmittedAI = tf.ones(tf.shape(new_positions_transmittedAI)[0], dtype=tf.bool)

    # Performing Anistropic_Anisotropic Interface analysis...
    # Creating an Aniso_Aniso masked version of all variables:
    positions_Aniso_Aniso = tf.boolean_mask(positions_newMat, Aniso_Aniso_mask)
    p_Aniso_Aniso = tf.boolean_mask(p_newMat, Aniso_Aniso_mask)
    Efields_Aniso_Aniso = tf.boolean_mask(Efields_newMat, Aniso_Aniso_mask)
    PoyntingMag_Aniso_Aniso = tf.boolean_mask(PoyntingMag_newMat, Aniso_Aniso_mask)
    ray_ordinary_consts_Aniso_Aniso = tf.boolean_mask(ray_ordinary_consts_newMat, Aniso_Aniso_mask)
    ray_extraordinary_consts_Aniso_Aniso = tf.boolean_mask(ray_extraordinary_consts_newMat, Aniso_Aniso_mask)
    ray_director_consts_Aniso_Aniso = tf.boolean_mask(ray_director_consts_newMat, Aniso_Aniso_mask)
    group_IDs_Aniso_Aniso = tf.boolean_mask(group_IDs_newMat, Aniso_Aniso_mask)
    parent_IDs_Aniso_Aniso = tf.boolean_mask(parent_IDs_newMat, Aniso_Aniso_mask)
    prev_Mat_types_Aniso_Aniso = tf.boolean_mask(prev_Mat_types, Aniso_Aniso_mask)
    new_Mat_types_Aniso_Aniso = tf.boolean_mask(new_Mat_types, Aniso_Aniso_mask)
    prev_MatIDs_Aniso_Aniso = tf.boolean_mask(material_IDs_newMat, Aniso_Aniso_mask)
    new_MatIDs_Aniso_Aniso = tf.boolean_mask(newMat_IDs, Aniso_Aniso_mask)

    # Setting up and calling the Anisotropic-Anisotropic interface analysis function:
    surface_normals = nh.getSurfaceNormals(new_MatIDs_Aniso_Aniso, prev_MatIDs_Aniso_Aniso, geometry_vectors, positions_Aniso_Aniso)
    e_perp1, _, _, _ = nh.getOrdinaryPermittivities(positions_Aniso_Aniso, ray_ordinary_consts_Aniso_Aniso)
    no1 = tf.math.sqrt(e_perp1)
    e_para1, _, _, _ = nh.getExtraordinaryPermittivities(positions_Aniso_Aniso, ray_extraordinary_consts_Aniso_Aniso)
    ne1 = tf.math.sqrt(e_para1)
    optical_axis1, _, _, _, _, _, _, _, _, _ = nh.getDirector(positions_Aniso_Aniso, ray_director_consts_Aniso_Aniso)
    mat2_ordinary_consts = tf.gather(mat_ordinary_consts, new_MatIDs_Aniso_Aniso)
    e_perp2, _, _, _ = nh.getOrdinaryPermittivities(positions_Aniso_Aniso, mat2_ordinary_consts)
    no2 = tf.math.sqrt(e_perp2)
    mat2_extraordinary_consts = tf.gather(mat_extraordinary_consts, new_MatIDs_Aniso_Aniso)
    e_para2, _, _, _ = nh.getExtraordinaryPermittivities(positions_Aniso_Aniso, mat2_extraordinary_consts)
    ne2 = tf.math.sqrt(e_para2)
    mat2_director_consts = tf.gather(mat_director_consts, new_MatIDs_Aniso_Aniso)
    optical_axis2, _, _, _, _, _, _, _, _, _ = nh.getDirector(positions_Aniso_Aniso, mat2_director_consts)

    p_roAA, p_reAA, p_toAA, p_teAA, E_roAA, E_reAA, E_toAA, E_teAA, S_roAA, S_reAA, S_toAA, S_teAA = nh.Anisotropic_Anisotropic(surface_normals, optical_axis1, optical_axis2, no1, ne1, no2, ne2, p_Aniso_Aniso, Efields_Aniso_Aniso)
    S_roAA = S_roAA*PoyntingMag_Aniso_Aniso # Multiply the normalized Poynting vectors by the Poynting Vector magnitude of the corresponding incident ray
    S_reAA = S_reAA*PoyntingMag_Aniso_Aniso
    S_toAA = S_toAA*PoyntingMag_Aniso_Aniso
    S_teAA = S_teAA*PoyntingMag_Aniso_Aniso

    # Initialize the rest of the variables for the new reflected and transmitted rays:
    new_positions_reflected_ordinaryAA = positions_Aniso_Aniso
    new_positions_reflected_extraordinaryAA = positions_Aniso_Aniso
    new_positions_transmitted_ordinaryAA = positions_Aniso_Aniso
    new_positions_transmitted_extraordinaryAA = positions_Aniso_Aniso

    group_IDs_reflected_ordinaryAA = group_IDs_Aniso_Aniso
    group_IDs_reflected_extraordinaryAA = group_IDs_Aniso_Aniso
    group_IDs_transmitted_ordinaryAA = group_IDs_Aniso_Aniso
    group_IDs_transmitted_extraordinaryAA = group_IDs_Aniso_Aniso

    parent_IDs_reflected_ordinaryAA = parent_IDs_Aniso_Aniso
    parent_IDs_reflected_extraordinaryAA = parent_IDs_Aniso_Aniso
    parent_IDs_transmitted_ordinaryAA = parent_IDs_Aniso_Aniso
    parent_IDs_transmitted_extraordinaryAA = parent_IDs_Aniso_Aniso
    
    new_materialIDs_reflected_ordinaryAA = prev_MatIDs_Aniso_Aniso
    new_materialIDs_reflected_extraordinaryAA = prev_MatIDs_Aniso_Aniso
    new_materialIDs_transmitted_ordinaryAA = new_MatIDs_Aniso_Aniso
    new_materialIDs_transmitted_extraordinaryAA = new_MatIDs_Aniso_Aniso
    
    ray_ordinary_consts_reflected_ordinaryAA = tf.gather(mat_ordinary_consts, new_materialIDs_reflected_ordinaryAA)
    ray_ordinary_consts_reflected_extraordinaryAA = tf.gather(mat_ordinary_consts, new_materialIDs_reflected_extraordinaryAA)
    ray_ordinary_consts_transmitted_ordinaryAA = tf.gather(mat_ordinary_consts, new_materialIDs_transmitted_ordinaryAA)
    ray_ordinary_consts_transmitted_extraordinaryAA = tf.gather(mat_ordinary_consts, new_materialIDs_transmitted_extraordinaryAA)

    ray_extraordinary_consts_reflected_ordinaryAA = tf.gather(mat_extraordinary_consts, new_materialIDs_reflected_ordinaryAA)
    ray_extraordinary_consts_reflected_extraordinaryAA = tf.gather(mat_extraordinary_consts, new_materialIDs_reflected_extraordinaryAA)
    ray_extraordinary_consts_transmitted_ordinaryAA = tf.gather(mat_extraordinary_consts, new_materialIDs_transmitted_ordinaryAA)
    ray_extraordinary_consts_transmitted_extraordinaryAA = tf.gather(mat_extraordinary_consts, new_materialIDs_transmitted_extraordinaryAA)

    ray_director_consts_reflected_ordinaryAA = tf.gather(mat_director_consts, new_materialIDs_reflected_ordinaryAA)
    ray_director_consts_reflected_extraordinaryAA = tf.gather(mat_director_consts, new_materialIDs_reflected_extraordinaryAA)
    ray_director_consts_transmitted_ordinaryAA = tf.gather(mat_director_consts, new_materialIDs_transmitted_ordinaryAA)
    ray_director_consts_transmitted_extraordinaryAA = tf.gather(mat_director_consts, new_materialIDs_transmitted_extraordinaryAA)

    ordinary_reflected_ordinaryAA = tf.ones(tf.shape(new_positions_reflected_ordinaryAA)[0], dtype=tf.bool)
    ordinary_reflected_extraordinaryAA = tf.zeros(tf.shape(new_positions_reflected_extraordinaryAA)[0], dtype=tf.bool)
    ordinary_transmitted_ordinaryAA = tf.ones(tf.shape(new_positions_transmitted_ordinaryAA)[0], dtype=tf.bool)
    ordinary_transmitted_extraordinaryAA = tf.zeros(tf.shape(new_positions_transmitted_extraordinaryAA)[0], dtype=tf.bool)

    # If IGNORE_REFLECTIONS is True, ensure all variables corresponding to reflected rays are empty:
    if IGNORE_REFLECTIONS:
        new_positions_reflectedII = tf.zeros_like(new_positions_reflectedII)[:0]
        new_positions_reflectedIA = tf.zeros_like(new_positions_reflectedIA)[:0]
        new_positions_reflected_ordinaryAI = tf.zeros_like(new_positions_reflected_ordinaryAI)[:0]
        new_positions_reflected_extraordinaryAI = tf.zeros_like(new_positions_reflected_ordinaryAI)[:0]
        new_positions_reflected_ordinaryAA = tf.zeros_like(new_positions_reflected_ordinaryAI)[:0]
        new_positions_reflected_extraordinaryAA = tf.zeros_like(new_positions_reflected_ordinaryAI)[:0]
        p_rII = tf.zeros_like(p_rII)[:0]
        p_rIA = tf.zeros_like(p_rIA)[:0]
        p_roAI = tf.zeros_like(p_roAI)[:0]
        p_reAI = tf.zeros_like(p_reAI)[:0]
        p_roAA = tf.zeros_like(p_roAA)[:0]
        p_reAA = tf.zeros_like(p_reAA)[:0]
        S_rII = tf.zeros_like(S_rII)[:0]
        S_rIA = tf.zeros_like(S_rIA)[:0]
        S_roAI = tf.zeros_like(S_roAI)[:0]
        S_reAI = tf.zeros_like(S_reAI)[:0]
        S_roAA = tf.zeros_like(S_roAA)[:0]
        S_reAA = tf.zeros_like(S_reAA)[:0]
        E_rII = tf.zeros_like(E_rII)[:0]
        E_rIA = tf.zeros_like(E_rIA)[:0]
        E_roAI = tf.zeros_like(E_roAI)[:0]
        E_reAI = tf.zeros_like(E_reAI)[:0]
        E_roAA = tf.zeros_like(E_roAA)[:0]
        E_reAA = tf.zeros_like(E_reAA)[:0]
        ordinary_reflectedII = tf.zeros_like(ordinary_reflectedII)[:0]
        ordinary_reflectedIA = tf.zeros_like(ordinary_reflectedIA)[:0]
        ordinary_reflected_ordinaryAI = tf.zeros_like(ordinary_reflected_ordinaryAI)[:0]
        ordinary_reflected_extraordinaryAI = tf.zeros_like(ordinary_reflected_extraordinaryAI)[:0]
        ordinary_reflected_ordinaryAA = tf.zeros_like(ordinary_reflected_ordinaryAA)[:0]
        ordinary_reflected_extraordinaryAA = tf.zeros_like(ordinary_reflected_extraordinaryAA)[:0]

        group_IDs_reflectedII = tf.zeros_like(group_IDs_reflectedII)[:0]
        group_IDs_reflectedIA = tf.zeros_like(group_IDs_reflectedIA)[:0]
        group_IDs_reflected_ordinaryAI = tf.zeros_like(group_IDs_reflected_ordinaryAI)[:0]
        group_IDs_reflected_extraordinaryAI = tf.zeros_like(group_IDs_reflected_extraordinaryAI)[:0]
        group_IDs_reflected_ordinaryAA = tf.zeros_like(group_IDs_reflected_ordinaryAA)[:0]
        group_IDs_reflected_extraordinaryAA = tf.zeros_like(group_IDs_reflected_extraordinaryAA)[:0]

        parent_IDs_reflectedII = tf.zeros_like(parent_IDs_reflectedII)[:0]
        parent_IDs_reflectedIA = tf.zeros_like(parent_IDs_reflectedIA)[:0]
        parent_IDs_reflected_ordinaryAI = tf.zeros_like(parent_IDs_reflected_ordinaryAI)[:0]
        parent_IDs_reflected_extraordinaryAI = tf.zeros_like(parent_IDs_reflected_extraordinaryAI)[:0]
        parent_IDs_reflected_ordinaryAA = tf.zeros_like(parent_IDs_reflected_ordinaryAA)[:0]
        parent_IDs_reflected_extraordinaryAA = tf.zeros_like(parent_IDs_reflected_extraordinaryAA)[:0]

        new_materialIDs_reflectedII = tf.zeros_like(new_materialIDs_reflectedII)[:0]
        new_materialIDs_reflectedIA = tf.zeros_like(new_materialIDs_reflectedIA)[:0]
        new_materialIDs_reflected_ordinaryAI = tf.zeros_like(new_materialIDs_reflected_ordinaryAI)[:0]
        new_materialIDs_reflected_extraordinaryAI = tf.zeros_like(new_materialIDs_reflected_extraordinaryAI)[:0]
        new_materialIDs_reflected_ordinaryAA = tf.zeros_like(new_materialIDs_reflected_ordinaryAA)[:0]
        new_materialIDs_reflected_extraordinaryAA = tf.zeros_like(new_materialIDs_reflected_extraordinaryAA)[:0]
        ray_ordinary_consts_reflectedII = tf.zeros_like(ray_ordinary_consts_reflectedII)[:0]
        ray_ordinary_consts_reflectedIA = tf.zeros_like(ray_ordinary_consts_reflectedIA)[:0]
        ray_ordinary_consts_reflected_ordinaryAI = tf.zeros_like(ray_ordinary_consts_reflected_ordinaryAI)[:0]
        ray_ordinary_consts_reflected_extraordinaryAI = tf.zeros_like(ray_ordinary_consts_reflected_extraordinaryAI)[:0]
        ray_ordinary_consts_reflected_ordinaryAA = tf.zeros_like(ray_ordinary_consts_reflected_ordinaryAA)[:0]
        ray_ordinary_consts_reflected_extraordinaryAA = tf.zeros_like(ray_ordinary_consts_reflected_extraordinaryAA)[:0]
        ray_extraordinary_consts_reflectedII = tf.zeros_like(ray_extraordinary_consts_reflectedII)[:0]
        ray_extraordinary_consts_reflectedIA = tf.zeros_like(ray_extraordinary_consts_reflectedIA)[:0]
        ray_extraordinary_consts_reflected_ordinaryAI = tf.zeros_like(ray_extraordinary_consts_reflected_ordinaryAI)[:0]
        ray_extraordinary_consts_reflected_extraordinaryAI = tf.zeros_like(ray_extraordinary_consts_reflected_extraordinaryAI)[:0]
        ray_extraordinary_consts_reflected_ordinaryAA = tf.zeros_like(ray_extraordinary_consts_reflected_ordinaryAA)[:0]
        ray_extraordinary_consts_reflected_extraordinaryAA = tf.zeros_like(ray_extraordinary_consts_reflected_extraordinaryAA)[:0]
        ray_director_consts_reflectedII = tf.zeros_like(ray_director_consts_reflectedII)[:0]
        ray_director_consts_reflectedIA = tf.zeros_like(ray_director_consts_reflectedIA)[:0]
        ray_director_consts_reflected_ordinaryAI = tf.zeros_like(ray_director_consts_reflected_ordinaryAI)[:0]
        ray_director_consts_reflected_extraordinaryAI = tf.zeros_like(ray_director_consts_reflected_extraordinaryAI)[:0]
        ray_director_consts_reflected_ordinaryAA = tf.zeros_like(ray_director_consts_reflected_ordinaryAA)[:0]
        ray_director_consts_reflected_extraordinaryAA = tf.zeros_like(ray_director_consts_reflected_extraordinaryAA)[:0]

    # Next, concatenate all the new variables of each type together:
    positions_combined = tf.concat([new_positions_reflectedII, new_positions_transmittedII, new_positions_reflectedIA, new_positions_transmitted_ordinaryIA, new_positions_transmitted_extraordinaryIA, new_positions_reflected_ordinaryAI, new_positions_reflected_extraordinaryAI, new_positions_transmittedAI, new_positions_reflected_ordinaryAA, new_positions_reflected_extraordinaryAA, new_positions_transmitted_ordinaryAA, new_positions_transmitted_extraordinaryAA], axis=0)
    wave_vectors_combined = tf.concat([p_rII, p_tII, p_rIA, p_toIA, p_teIA, p_roAI, p_reAI, p_tAI, p_roAA, p_reAA, p_toAA, p_teAA], axis=0)
    PoyntingMag_combined = tf.concat([S_rII, S_tII, S_rIA, S_toIA, S_teIA, S_roAI, S_reAI, S_tAI, S_roAA, S_reAA, S_toAA, S_teAA], axis=0)
    Efields_combined = tf.concat([E_rII, E_tII, E_rIA, E_toIA, E_teIA, E_roAI, E_reAI, E_tAI, E_roAA, E_reAA, E_toAA, E_teAA], axis=0)
    ordinary_combined = tf.concat([ordinary_reflectedII, ordinary_transmittedII, ordinary_reflectedIA, ordinary_transmitted_ordinaryIA, ordinary_transmitted_extraordinaryIA, ordinary_reflected_ordinaryAI, ordinary_reflected_extraordinaryAI, ordinary_transmittedAI, ordinary_reflected_ordinaryAA, ordinary_reflected_extraordinaryAA, ordinary_transmitted_ordinaryAA, ordinary_transmitted_extraordinaryAA], axis=0)
    group_IDs_combined = tf.concat([group_IDs_reflectedII, group_IDs_transmittedII, group_IDs_reflectedIA, group_IDs_transmitted_ordinaryIA, group_IDs_transmitted_extraordinaryIA, group_IDs_reflected_ordinaryAI, group_IDs_reflected_extraordinaryAI, group_IDs_transmittedAI, group_IDs_reflected_ordinaryAA, group_IDs_reflected_extraordinaryAA, group_IDs_transmitted_ordinaryAA, group_IDs_transmitted_extraordinaryAA], axis=0)
    parent_IDs_combined = tf.concat([parent_IDs_reflectedII, parent_IDs_transmittedII, parent_IDs_reflectedIA, parent_IDs_transmitted_ordinaryIA, parent_IDs_transmitted_extraordinaryIA, parent_IDs_reflected_ordinaryAI, parent_IDs_reflected_extraordinaryAI, parent_IDs_transmittedAI, parent_IDs_reflected_ordinaryAA, parent_IDs_reflected_extraordinaryAA, parent_IDs_transmitted_ordinaryAA, parent_IDs_transmitted_extraordinaryAA], axis=0)
    material_IDs_combined = tf.concat([new_materialIDs_reflectedII, new_materialIDs_transmittedII, new_materialIDs_reflectedIA, new_materialIDs_transmitted_ordinaryIA, new_materialIDs_transmitted_extraordinaryIA, new_materialIDs_reflected_ordinaryAI, new_materialIDs_reflected_extraordinaryAI, new_materialIDs_transmittedAI, new_materialIDs_reflected_ordinaryAA, new_materialIDs_reflected_extraordinaryAA, new_materialIDs_transmitted_ordinaryAA, new_materialIDs_transmitted_extraordinaryAA], axis=0)
    ray_ordinary_consts_combined = tf.concat([ray_ordinary_consts_reflectedII, ray_ordinary_consts_transmittedII, ray_ordinary_consts_reflectedIA, ray_ordinary_consts_transmitted_ordinaryIA, ray_ordinary_consts_transmitted_extraordinaryIA, ray_ordinary_consts_reflected_ordinaryAI, ray_ordinary_consts_reflected_extraordinaryAI, ray_ordinary_consts_transmittedAI, ray_ordinary_consts_reflected_ordinaryAA, ray_ordinary_consts_reflected_extraordinaryAA, ray_ordinary_consts_transmitted_ordinaryAA, ray_ordinary_consts_transmitted_extraordinaryAA], axis=0)
    ray_extraordinary_consts_combined = tf.concat([ray_extraordinary_consts_reflectedII, ray_extraordinary_consts_transmittedII, ray_extraordinary_consts_reflectedIA, ray_extraordinary_consts_transmitted_ordinaryIA, ray_extraordinary_consts_transmitted_extraordinaryIA, ray_extraordinary_consts_reflected_ordinaryAI, ray_extraordinary_consts_reflected_extraordinaryAI, ray_extraordinary_consts_transmittedAI, ray_extraordinary_consts_reflected_ordinaryAA, ray_extraordinary_consts_reflected_extraordinaryAA, ray_extraordinary_consts_transmitted_ordinaryAA, ray_extraordinary_consts_transmitted_extraordinaryAA], axis=0)
    ray_director_consts_combined = tf.concat([ray_director_consts_reflectedII, ray_director_consts_transmittedII, ray_director_consts_reflectedIA, ray_director_consts_transmitted_ordinaryIA, ray_director_consts_transmitted_extraordinaryIA, ray_director_consts_reflected_ordinaryAI, ray_director_consts_reflected_extraordinaryAI, ray_director_consts_transmittedAI, ray_director_consts_reflected_ordinaryAA, ray_director_consts_reflected_extraordinaryAA, ray_director_consts_transmitted_ordinaryAA, ray_director_consts_transmitted_extraordinaryAA], axis=0)

    # Create appropriately-sized alive_combined tensor (all new rays are set to True):
    alive_combined = tf.ones(tf.shape(positions_combined)[0], dtype=tf.bool)

    # Create a mask to eliminate rays with a sufficiently small Poyning Vector magnitude:
    small_magnitude_mask = tf.greater(PoyntingMag_combined, minimum_Poynting)

    # Apply the mask to all variables:
    positions_combined = tf.boolean_mask(positions_combined, small_magnitude_mask)
    wave_vectors_combined = tf.boolean_mask(wave_vectors_combined, small_magnitude_mask)
    PoyntingMag_combined = tf.boolean_mask(PoyntingMag_combined, small_magnitude_mask)
    Efields_combined = tf.boolean_mask(Efields_combined, small_magnitude_mask)
    ordinary_combined = tf.boolean_mask(ordinary_combined, small_magnitude_mask)
    group_IDs_combined = tf.boolean_mask(group_IDs_combined, small_magnitude_mask)
    parent_IDs_combined = tf.boolean_mask(parent_IDs_combined, small_magnitude_mask)
    material_IDs_combined = tf.boolean_mask(material_IDs_combined, small_magnitude_mask)
    ray_ordinary_consts_combined = tf.boolean_mask(ray_ordinary_consts_combined, small_magnitude_mask)
    ray_extraordinary_consts_combined = tf.boolean_mask(ray_extraordinary_consts_combined, small_magnitude_mask)
    ray_director_consts_combined = tf.boolean_mask(ray_director_consts_combined, small_magnitude_mask)
    alive_combined = tf.boolean_mask(alive_combined, small_magnitude_mask)

    # Calculate indices to update based on "Num_rays_active", and number of new rays being created at the interface (e.g. tf.shape(positions_combined)[0]):
    target_rows = tf.range(Num_rays_active, Num_rays_active + tf.shape(positions_combined)[0])

    # Create the 3D indices update tensor (for (N,) tensors):
    indices1 = target_rows
    indices1 = tf.expand_dims(indices1, axis=1)

    # Create the 3D incidces update tensor for coefficients (NxNum_constants)
    indicesCoefs = target_rows
    indicesCoefs = tf.expand_dims(indicesCoefs, axis=1)
    
    # Update Num_rays_active to account for the fact that new rays have been added to the system:
    Num_rays_active = Num_rays_active + tf.shape(positions_combined)[0]

    # Update the "global" variables using scatterndupdate:
    pos_slice = positions_ta.read(step)
    pos_slice = tf.tensor_scatter_nd_update(pos_slice, indices1, positions_combined)
    positions_ta = positions_ta.write(step, pos_slice)

    wv_slice = wave_vectors_ta.read(step)
    wv_slice = tf.tensor_scatter_nd_update(wv_slice, indices1, wave_vectors_combined)
    wave_vectors_ta = wave_vectors_ta.write(step, wv_slice)

    Efields_slice = Efields_ta.read(step)
    Efields_slice = tf.tensor_scatter_nd_update(Efields_slice, indices1, Efields_combined)
    Efields_ta = Efields_ta.write(step, Efields_slice)

    # Update standard tensor variables using scatterndupdate:
    PoyntingMag = tf.tensor_scatter_nd_update(PoyntingMag, indices1, PoyntingMag_combined)
    ordinary = tf.tensor_scatter_nd_update(ordinary, indices1, ordinary_combined)
    group_IDs = tf.tensor_scatter_nd_update(group_IDs, indices1, group_IDs_combined)
    parent_IDs = tf.tensor_scatter_nd_update(parent_IDs, indices1, parent_IDs_combined)
    material_IDs = tf.tensor_scatter_nd_update(material_IDs, indices1, material_IDs_combined)
    ray_ordinary_consts = tf.tensor_scatter_nd_update(ray_ordinary_consts, indicesCoefs, ray_ordinary_consts_combined)
    ray_extraordinary_consts = tf.tensor_scatter_nd_update(ray_extraordinary_consts, indicesCoefs, ray_extraordinary_consts_combined)
    ray_director_consts = tf.tensor_scatter_nd_update(ray_director_consts, indicesCoefs, ray_director_consts_combined)
    alive = tf.tensor_scatter_nd_update(alive, indices1, alive_combined)

    # 5) Increment the "step" variable:
    step = step + 1

# Convert positions, wave_vectors, and Efields into rank 3 tensors for loss function computation:
positions_final = tf.transpose(positions_ta.stack(), perm=[1, 2, 0])
wave_vectors_final = tf.transpose(wave_vectors_ta.stack(), perm=[1, 2, 0])
Efields_final = tf.transpose(Efields_ta.stack(), perm=[1, 2, 0])

# Compute constructCompleteTensors() function:
unique_parent_IDs, group_IDs_by_parent, positions_by_parent, wave_vectors_by_parent, Efields_by_parent, PoyntingMags_by_parent = nh.constructCompleteTensors(positions_final, wave_vectors_final, Efields_final, PoyntingMag, group_IDs, parent_IDs)

# Interpolate the phase, position, E-field, and wave vector of each ray at regular intervals along its path:
phases_interp, positions_interp, Efields_interp, wave_vectors_interp = nh.computePhase(positions_by_parent, wave_vectors_by_parent, Efields_by_parent, wavelength=1.5, interpolation_step=10)

# Compute the phase, position, E-field, and wave vector at a specified z=constant plane:
intersected, xy_positions, phases_at_plane, Efields_at_plane, wave_vectors_at_plane = nh.computeAtPlane(phases_interp, positions_interp, Efields_interp, wave_vectors_interp, z_target=3.0)
# (Manually) compute the Poynting Magnitudes at the plane by knowing which material the plane is in:
dense_poynting = PoyntingMags_by_parent.to_tensor(default_value=0.0)
S_mags_at_plane = dense_poynting[:, 2] # Since the plane is past the lens (third ray per parent)

# Compute fx and fy (this is a test):
U_normalized = nh.computeFarField(xy_positions, phases_at_plane, Efields_at_plane, S_mags_at_plane, wave_vectors_at_plane, wavelength=1.5)

# Plot the normalized far field pattern (phi=0 cut):
U_phi0 = U_normalized[:, 0]
U_phi_dB = 10.0*tf.math.log(U_phi0)/tf.math.log(10.0) # Convert to dB scale
theta = tf.linspace(0.0, 180.0, tf.shape(U_phi0)[0])
plt.plot(theta.numpy(), U_phi_dB.numpy())
plt.xlabel("Theta (degrees)")
plt.ylabel("Normalized Directivity (dBi)")
plt.title("Far Field Radiation Pattern (Phi = 0)")
plt.grid()
plt.savefig("far_field_pattern.png", dpi=300)

# Calculate and print focus loss value:
loss = nh.planeWaveObjective(wave_vectors_final, group_IDs, material_IDs, Num_Starting_Rays)
# theta, phi = nh.calcBeamAngles(wave_vectors_final, group_IDs, material_IDs, Num_Starting_Rays, specificGroup=1)
print(f"Value of Loss: {loss.numpy()}")

# End timer:
end_time = time.perf_counter()

# Calculate and print runtime:
duration = end_time - start_time
print(f"Execution took {duration:.4f} seconds")

# Print number of steps and number of rays:
print(f"Total Numbers of Steps: {step}")
print(f"Total Number of Rays: {Num_rays_active}")

# Calculate and plot the rays from a particular feed on the aperture:
aperture_positions = nh.calcAperturePositions(positions_final, wave_vectors_final, material_IDs, focal_plane=2.5, group_IDs=group_IDs, groupID=1)
nh.plotAperturePositions(aperture_positions)

# Plotting results:

fig = plt.figure()
ax = plt.axes(projection='3d')

# Plotting semi-transparent cylinder to help visualize locaiton of lens:

# Cylinder parameters
radius = tf.constant(8.0, dtype=tf.float32)
height = tf.constant(5.0, dtype=tf.float32)
z_min = tf.constant(-2.5, dtype=tf.float32)

n_theta = 100
n_z = 100

# Create parameter ranges using TensorFlow
theta = tf.linspace(0.0, 2.0 * tf.constant(3.141592653589793, dtype=tf.float32), n_theta)
z = tf.linspace(z_min, z_min + height, n_z)

# Create meshgrid (TensorFlow version)
theta_grid, z_grid = tf.meshgrid(theta, z)

# Parametric cylinder
x_cyl = radius * tf.cos(theta_grid)
y_cyl = radius * tf.sin(theta_grid)

# Convert to numpy ONLY for plotting
x_cyl_np = x_cyl.numpy()
y_cyl_np = y_cyl.numpy()
z_cyl_np = z_grid.numpy()

# Plot surface
ax.plot_surface(
    x_cyl_np,
    y_cyl_np,
    z_cyl_np,
    color='gray',
    alpha=0.2,
    edgecolor='none'
)

# Number of samples
n_theta = 100
n_r = 50

# Constants
pi = tf.constant(3.141592653589793, dtype=tf.float32)

# Parameter ranges
theta = tf.linspace(0.0, 2.0 * pi, n_theta)
r = tf.linspace(0.0, radius, n_r)

# Meshgrid in polar coords
theta_grid_cap, r_grid = tf.meshgrid(theta, r)

# Convert to Cartesian
x_cap = r_grid * tf.cos(theta_grid_cap)
y_cap = r_grid * tf.sin(theta_grid_cap)

# Top cap (z = z_min + height)
z_top = tf.ones_like(x_cap) * (z_min + height)

# Bottom cap (z = z_min)
z_bottom = tf.ones_like(x_cap) * z_min

# Convert to numpy for plotting
x_cap_np = x_cap.numpy()
y_cap_np = y_cap.numpy()
z_top_np = z_top.numpy()
z_bottom_np = z_bottom.numpy()

# Plot caps
ax.plot_surface(
    x_cap_np, y_cap_np, z_top_np,
    color='gray', alpha=0.2, edgecolor='none'
)

ax.plot_surface(
    x_cap_np, y_cap_np, z_bottom_np,
    color='gray', alpha=0.2, edgecolor='none'
)

# Plotting only a single (complete) ray:
x_coords = positions_by_parent[0,:,0].numpy()
y_coords = positions_by_parent[0,:,1].numpy()
z_coords = positions_by_parent[0,:,2].numpy()

ax.plot3D(x_coords, y_coords, z_coords, color='blue', linewidth=2)

# # Plotting the ray paths:
# for i in range(Num_rays_active):
#     # Extract the trajectory for the current ray [3, step-1]
#     path = positions_final[i, :, :step-1]

#     # Only plotting rays from a particular feed:
#     if (group_IDs[i] == 1):
#         # Create a mask: True if any coordinate (x,y,z) at that step is non-zero
#         # We check across the coordinate axis (axis 0)
#         mask = tf.reduce_any(path != 0, axis=0).numpy()
        
#         # Apply the mask to x, y, and z coordinates
#         z_coords = path[2, :][mask]
#         y_coords = path[1, :][mask]
#         x_coords = path[0, :][mask]

#         color = 'blue' if ordinary[i] else 'red'
#         ax.scatter3D(x_coords, y_coords, z_coords, color=color, s=5)

# # Plot a vertical line to indicate the start of the lens:
# plt.axvline(x=0.05, color='k', linestyle='-', linewidth=2)

# # Plot a vertical line to indicate the end of the lens:
# plt.axvline(x=0.35, color='k', linestyle='-', linewidth=2)

# # Plot a vertical line to indicate the desired "extraordinary" focal point:
# plt.axvline(x=0.45, color='k', linestyle='--', linewidth=2)

plt.title("3D Ray Propagation")
plt.grid(True)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_zlim(-5, 5)
ax.set_box_aspect([1, 1, 1])

plt.savefig("3D_Ray_Propagation.png", dpi=300)