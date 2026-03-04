import tensorflow as tf
import NewHelpers as nh
import matplotlib.pyplot as plt
import time

# This file contains the ray-tracing computation and loss function calculation all wrapped
# in a single @tf.function.

@tf.function
def rayTrace(a0, a2, b0, b2):
    # This ray-tracing algorithm attempts to perform all tracing of rays in parallel to leverage GPU acceleration.

    # Static Constants:
    Max_Rays = 500 # Variable to keep track of the max number of rays allowed.
    SIZE = 1000 # Constant size of arrays to store ray positions, wave normals, and electric fields.
    distance_step = tf.constant(0.005) # Fixed distance step for propagating rays
    minimum_Poynting = tf.constant(0.1) # Any ray with a Poynting Vector magnitude less than this value will not be "launched" when creating new rays at an interface
    Num_constants = tf.constant(5, dtype=tf.int32) # Number of constants needed to parameterize the materials (permittivity + director profile)
    Num_Materials = tf.constant(2, dtype=tf.int32) # Size of the number of materials in the simulation (e.g. 2 ==> 3 materials)

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
    step = tf.constant(1, dtype=tf.int32)
    Num_rays_active = tf.constant(10, dtype=tf.int32)

    # Initialize material tensors:
    geometry_vectors = tf.constant([[-0.20, 0.20, -0.05, 0.05, 0.0, 0.05], [-0.20, 0.20, -0.05, 0.05, 0.05, 0.35], [-0.20, 0.20, -0.05, 0.05, 0.35, 0.50]])
    isotropic = tf.constant([True, False, True], dtype=tf.bool)

    mat_ordinary_consts = tf.stack([[-100.0, 0.0, 0.0, 0.0, 0.0], [a0, 0.0, a2, 0.0, 0.0], [-100.0, 0.0, 0.0, 0.0, 0.0]])
    mat_extraordinary_consts = tf.stack([[-100.0, 0.0, 0.0, 0.0, 0.0], [b0, 0.0, b2, 0.0, 0.0], [-100.0, 0.0, 0.0, 0.0, 0.0]])
    mat_director_consts = tf.stack([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])

    # Initialize Bounding Box:
    boundingBox = tf.constant([-0.20, 0.20, -0.05, 0.05, 0.0, 0.50])

    # Create initial rays to launch into system:
    starting_x = -0.01
    ending_x = 0.01
    fixed_z = 0.03
    angle = 0.001
    Epol = tf.constant([1.0, 0.0, 0.0])

    positions_initial, wave_vectors_initial, PoyntingMag_initial, alive_initial, Efields_initial, ordinary_initial, material_IDs_initial, ray_ordinary_consts_initial, ray_extraordinary_consts_initial, ray_director_consts_initial = nh.createStartingRays(Num_rays_active, starting_x, ending_x, fixed_z, angle, Epol, mat_ordinary_consts, mat_extraordinary_consts, mat_director_consts)

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

    # Initialize the TensorArrays using TensorArray.write():
    positions_ta = positions_ta.write(0, tf.pad(positions_initial, paddings))
    wave_vectors_ta = wave_vectors_ta.write(0, tf.pad(wave_vectors_initial, paddings))
    Efields_ta = Efields_ta.write(0, tf.pad(Efields_initial, paddings))

    # Main ray-tracing logic:
    while tf.reduce_any(alive): # While there is at least one True in the "alive" tensor

        # 1) Perform a step of ray propagation:

        # Read the state of wave_vectors, positions, and Efields from the previous step (step - 1):
        curr_pos = positions_ta.read(step-1)
        curr_wave = wave_vectors_ta.read(step-1)
        curr_E = Efields_ta.read(step-1)

        # Call the rayPropagationStep function:
        new_wave_vectors, new_positions, new_Efields = nh.rayPropagation(curr_pos, curr_wave, curr_E, ordinary, alive, ray_ordinary_consts, ray_extraordinary_consts, ray_director_consts, isotropic, material_IDs, distance_step)
        
        # Update the TensorArrays for positions, wave vectors, and Electric fields:
        positions_ta = positions_ta.write(step, new_positions)
        wave_vectors_ta = wave_vectors_ta.write(step, new_wave_vectors)
        Efields_ta = Efields_ta.write(step, new_Efields)

        # 2) Check for rays that have left the boundingBox... if yes, set their alive to False
        alive = nh.checkBoundary(new_positions, boundingBox, alive)

        # 3) Check for rays that have reached a new material, and assign their alive to False. Also, rays that have reached out of bounds should not be included in newMatMask.
        newMatMask = nh.checkForHit(new_positions, material_IDs, geometry_vectors)
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

        # Next, concatenate all the new variables of each type together:
        positions_combined = tf.concat([new_positions_reflectedII, new_positions_transmittedII, new_positions_reflectedIA, new_positions_transmitted_ordinaryIA, new_positions_transmitted_extraordinaryIA, new_positions_reflected_ordinaryAI, new_positions_reflected_extraordinaryAI, new_positions_transmittedAI, new_positions_reflected_ordinaryAA, new_positions_reflected_extraordinaryAA, new_positions_transmitted_ordinaryAA, new_positions_transmitted_extraordinaryAA], axis=0)
        wave_vectors_combined = tf.concat([p_rII, p_tII, p_rIA, p_toIA, p_teIA, p_roAI, p_reAI, p_tAI, p_roAA, p_reAA, p_toAA, p_teAA], axis=0)
        PoyntingMag_combined = tf.concat([S_rII, S_tII, S_rIA, S_toIA, S_teIA, S_roAI, S_reAI, S_tAI, S_roAA, S_reAA, S_toAA, S_teAA], axis=0)
        Efields_combined = tf.concat([E_rII, E_tII, E_rIA, E_toIA, E_teIA, E_roAI, E_reAI, E_tAI, E_roAA, E_reAA, E_toAA, E_teAA], axis=0)
        ordinary_combined = tf.concat([ordinary_reflectedII, ordinary_transmittedII, ordinary_reflectedIA, ordinary_transmitted_ordinaryIA, ordinary_transmitted_extraordinaryIA, ordinary_reflected_ordinaryAI, ordinary_reflected_extraordinaryAI, ordinary_transmittedAI, ordinary_reflected_ordinaryAA, ordinary_reflected_extraordinaryAA, ordinary_transmitted_ordinaryAA, ordinary_transmitted_extraordinaryAA], axis=0)
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

    # Calculate and print focus loss value:
    desired_focal = tf.constant([0.0, 0.0, 0.45])
    loss = nh.focusObjective(positions_final, desired_focal, material_IDs)

    return loss

######### --- Main Optimization Loop --- ###########

# Start timer:
start_time = time.perf_counter()

# Define optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1)

# Define tf.Variables to be optimized:
a0 = tf.Variable(16.0, dtype=tf.float32)
a2 = tf.Variable(-431.54, dtype=tf.float32)
b0 = tf.Variable(1.5, dtype=tf.float32)
b2 = tf.Variable(-80.0, dtype=tf.float32)

# Define a variable to store loss values:
loss_history = []
a0_history = []
a2_history = []
b0_history = []
b2_history = []

# Running the training loop:
for i in range(100):
    with tf.GradientTape() as tape:
        current_loss = rayTrace(a0, a2, b0, b2)
    
    # Calculate gradients using automatic differentiation:
    gradients = tape.gradient(current_loss, [a0, a2, b0, b2])

    # Apply optimization
    optimizer.apply_gradients(zip(gradients, [a0, a2, b0, b2]))

    loss_history.append(current_loss.numpy())
    a0_history.append(a0.numpy())
    a2_history.append(a2.numpy())
    b0_history.append(b0.numpy())
    b2_history.append(b2.numpy())

# End timer:
end_time = time.perf_counter()

# Calculate and print runtime:
duration = end_time - start_time
print(f"Execution took {duration:.4f} seconds")

# Plotting both loss and coefficients over iteration #:

# Plot loss
plt.figure(figsize=(6,4))
plt.plot(loss_history, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Optimization Steps')
plt.grid(True)
plt.legend()
plt.show()

# --- Plot 1: a0 and b0 ---
plt.figure(figsize=(6,4))
plt.plot(a0_history, label='a0', marker='o')
plt.plot(b0_history, label='b0', marker='x')
plt.xlabel('Step')
plt.ylabel('Variable Value')
plt.title('Evolution of a0 and b0')
plt.grid(True)
plt.legend()
plt.show()

# --- Plot 2: a2 and b2 ---
plt.figure(figsize=(6,4))
plt.plot(a2_history, label='a2', marker='o')
plt.plot(b2_history, label='b2', marker='x')
plt.xlabel('Step')
plt.ylabel('Variable Value')
plt.title('Evolution of a2 and b2')
plt.grid(True)
plt.legend()
plt.show()

