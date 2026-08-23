import tensorflow as tf
import HelperFuncs as nh
import DkDist as dk
import LossFuncs as lf
import GenerateRays as gr
import PlottingFuncs as pf
import time
import matplotlib.pyplot as plt

# This file contains the ray-tracing computation and loss function calculation all wrapped
# in a single @tf.function.

@tf.function
def rayTrace(materialCoefs_lens, materialCoefs_air):
    # Static Constants:
    # General Solver Settings:
    Max_Rays = 1000 # Variable to keep track of the max number of rays allowed.
    SIZE = 300 # Constant size of arrays to store ray positions, wave normals, and electric fields.

    minimum_Poynting = tf.constant(0.1) # Any ray with a Poynting Vector magnitude less than this value will not be "launched" when creating new rays at an interface
    IGNORE_REFLECTIONS = True # True causes no reflected rays to spawn, False causes reflected rays to spawn as normal.
    f_GHz = 20.0 # Frequency in GHz, used to calculate wavelength for phase calculations at the end.
    wavelength = 300.0/f_GHz # Wavelength in mm, used for phase calculations at the end.
    # interp_num = 10 # Number of linear interpolations used to compute the optical path length.
    distance_step = wavelength/15.0 # Fixed distance step for propagating rays

    # Specify geometry:
    geometry_type = 'cylinder' # String which selects the geometry of objects in the scene. Options are 'cylinder', 'sphere', and 'slab'.
    lens_radius = 48.5 # Radius of the cylindrical lens
    lens_height = 40.0 # Height of the cylindrical lens
    r_padding = 16.0 # Radial padding from edge of lens to bounding box
    z_padding = 33.0 # Vertical padding from top/bottom of lens to bounding box

    # Specify permittivity distribution:
    dk_dist_flag = 1 # Integer flag which selects the type of permittivity distribution being used. See DkDist.py for a mapping of flags to distributions.
    # Reminder: For this example, coefficients are [A1, A2, C1, C2, emax] (see DkDist.py and look under "if flag == 1")
    A1 = 1.035
    A2 = 8.865
    C1 = -0.0174
    C2 = -0.00228
    emax = 5.57

    # Specify starting rays (either point source feeds 'point_sources' or plane waves 'plane_waves'):
    excitation_type = 'point_sources'

    # Configure point sources:
    if excitation_type == 'point_sources':
        NumFeeds = 4 # Number of point sources
        Rays_per_feed = 50 # How many rays get launched for each feed
        theta_max_deg = 50.0 # Specifies the maximum half cone angle of the launched rays
        z_dist_to_lens = 7.0 # Specifies how far behind the lens the plane containing the feeds is
        x_locations = [0.0, -5.0, -10.0, -15.0] # Specifies x coordinates of the feeds. The length must be equal to NumFeeds.
        y_locations = [0.0, 0.0, 0.0, 0.0] # Specifies y coordinates of the feeds. The length must be equal to NumFeeds.

    # Configure plane waves:
    if excitation_type == 'plane_waves':
        NumWaves = 4 # Number of different plane waves
        Rays_per_wave = 51 # How many rays get launched for each plane wave (will be rounded down to a square number)
        theta_wave = [0.0, 10.0, 20.0, 30.0] # Specify the incoming angle for each plane wave
        illumination_side_length = 30.0 # Specify the side length of the square which defines the illumination area of the lens

    ### ---------- USER INPUT ENDS HERE ---------- ###

    # Variables with History Storage (tf.TensorArrays):
    positions_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays, 3], clear_after_read=False)
    wave_vectors_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays, 3], clear_after_read=False)
    Efields_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays, 3], clear_after_read=False)
    # OPL_ta = tf.TensorArray(dtype=tf.float32, size=SIZE, element_shape=[Max_Rays], clear_after_read=False)

    # Initialize material tensors:
    # Geometry:
    geometry_vectors = tf.constant([[0.0, lens_radius, -0.5*lens_height, 0.5*lens_height], [lens_radius, lens_radius + r_padding, -0.5*lens_height - z_padding, 0.5*lens_height + z_padding]])
    boundingBox = tf.constant([0.0, lens_radius + r_padding, -0.5*lens_height - z_padding, 0.5*lens_height + z_padding])
    # Permittivity distribution:
    mat_ordinary_consts = tf.stack([materialCoefs_lens, materialCoefs_air])

    # Current State Tensors (standard tensors):
    # The values of these tensors are overwritten during the loop.
    alive = tf.zeros([Max_Rays], dtype=tf.bool)
    PoyntingMag = tf.zeros([Max_Rays], dtype=tf.float32)
    material_IDs = tf.zeros([Max_Rays], dtype=tf.int32)

    # Material parameter tensors (also standard tensors):
    ray_ordinary_consts = tf.zeros([Max_Rays, tf.shape(materialCoefs_lens)[0]], dtype=tf.float32)

    # Loop counters (int32 tensors):
    step = tf.constant(1, dtype=tf.int32) # This should always be initialized to one. DO NOT CHANGE.
    # Num_rays_active = tf.constant(200, dtype=tf.int32) # This should be intialized to the desired # of input starting rays
    # Num_Starting_Rays = Num_rays_active

    # Create starting rays:
    if excitation_type == 'point_sources':

        Num_rays_active = tf.cast(NumFeeds*Rays_per_feed, dtype=tf.int32)
        Num_Starting_Rays = Num_rays_active

        # Create source of spherical rays:
        Rays_per_feed_vector = Rays_per_feed * tf.ones(NumFeeds, dtype=tf.int32)
        theta_max = tf.repeat(theta_max_deg, NumFeeds)
        theta_target_deg = tf.repeat(0.0, NumFeeds)
        # sphere_center = tf.constant([[0.000001, 0.0000001, -24.0], [-5.0, 0.00000001, -24.0], [-10.0, 0.00000001, -24.0], [-15.0, 0.00000001, -24.0]])
        z_locations = (-0.5*lens_height - z_dist_to_lens) * tf.ones(NumFeeds)
        sphere_center = tf.stack([x_locations, y_locations, z_locations], axis=1)
        Epol = tf.repeat([[0.0, 1.0, 0.0]], NumFeeds, axis=0)

        positions_initial, wave_vectors_initial, PoyntingMag_initial, alive_initial, Efields_initial, material_IDs_initial, ray_ordinary_consts_initial = gr.createIsotropicRays(Rays_per_feed_vector, theta_max_deg=theta_max, theta_target_deg=theta_target_deg, sphere_center=sphere_center, Epol=Epol, ordinary_consts=mat_ordinary_consts)

        group_IDs_initial = 1 + tf.range(NumFeeds)
        group_IDs_initial = tf.repeat(group_IDs_initial, tf.cast(Rays_per_feed, dtype=tf.int32))
        parent_IDs_initial = 1 + tf.range(Num_rays_active)

    if excitation_type == 'plane_waves':

        raysPerLength = tf.cast(tf.math.sqrt(tf.cast(Rays_per_wave, dtype=tf.float32)), dtype=tf.int32)

        # Compute total number of rays:
        Num_rays_active = tf.cast(NumWaves*raysPerLength**2, dtype=tf.int32)
        Num_Starting_Rays = Num_rays_active

        lensBottom = -0.5*lens_height
        distToLens = z_padding - 5.0
        positions_initial, wave_vectors_initial, PoyntingMag_initial, alive_initial, Efields_initial, material_IDs_initial, ray_ordinary_consts_initial = gr.createPlaneWaves(illumination_side_length, theta_wave, raysPerLength, lensBottom, distToLens, mat_ordinary_consts)

        group_IDs_initial = 1 + tf.range(NumWaves)
        group_IDs_initial = tf.repeat(group_IDs_initial, tf.cast(raysPerLength**2, dtype=tf.int32))
        parent_IDs_initial = 1 + tf.range(Num_rays_active)

    # PAD/INSERT into Max_Rays tensors:
    # Use tf.pad() to take N active rays and fill the rest of the slots with zeros
    paddings = [[0, Max_Rays - Num_rays_active], [0, 0]]
    paddings_1d = [[0, Max_Rays - Num_rays_active]]

    # Update state tensors:
    alive = tf.pad(alive_initial, paddings_1d)
    PoyntingMag = tf.pad(PoyntingMag_initial, paddings_1d)
    material_IDs = tf.pad(material_IDs_initial, paddings_1d)
    ray_ordinary_consts = tf.pad(ray_ordinary_consts_initial, paddings)
    group_IDs = tf.pad(group_IDs_initial, paddings_1d)
    parent_IDs = tf.pad(parent_IDs_initial, paddings_1d)

    # Initialize the TensorArrays using TensorArray.write():
    positions_ta = positions_ta.write(0, tf.pad(positions_initial, paddings))
    wave_vectors_ta = wave_vectors_ta.write(0, tf.pad(wave_vectors_initial, paddings))
    Efields_ta = Efields_ta.write(0, tf.pad(Efields_initial, paddings))
    # OPL_ta = OPL_ta.write(0, tf.zeros(Max_Rays, dtype=tf.float32))

    # Main ray-tracing logic:
    while tf.reduce_any(alive): # While there is at least one True in the "alive" tensor

        # Preserve the previous alive mask so rays that died on this step can keep their last position,
        # while rays that were already dead remain zeroed out in future history steps.
        alive_before = alive

        # 1) Perform a step of ray propagation:

        # Read the state of wave_vectors, positions, Efields, and OPL from the previous step (step - 1):
        curr_pos = positions_ta.read(step-1)
        curr_wave = wave_vectors_ta.read(step-1)
        curr_E = Efields_ta.read(step-1)
        # curr_OPL = OPL_ta.read(step-1)

        # Call the rayPropagationStep function:
        new_wave_vectors, new_positions, new_Efields = nh.rayPropagation(curr_pos, curr_wave, curr_E, alive, ray_ordinary_consts, material_IDs, distance_step, dk_dist_flag)

        # Zero out any rays that were already dead before this step, so their later history does not persist.
        already_dead = tf.logical_not(alive_before)
        already_dead_expanded = tf.expand_dims(already_dead, axis=1)
        new_positions = tf.where(already_dead_expanded, tf.zeros_like(new_positions), new_positions)
        new_wave_vectors = tf.where(already_dead_expanded, tf.zeros_like(new_wave_vectors), new_wave_vectors)
        new_Efields = tf.where(already_dead_expanded, tf.zeros_like(new_Efields), new_Efields)
        # new_OPL = tf.where(already_dead, tf.zeros_like(new_OPL), new_OPL)
        
        # Update the TensorArrays for positions, wave vectors, and Electric fields:
        positions_ta = positions_ta.write(step, new_positions)
        wave_vectors_ta = wave_vectors_ta.write(step, new_wave_vectors)
        Efields_ta = Efields_ta.write(step, new_Efields)
        # OPL_ta = OPL_ta.write(step, new_OPL)

        # 2) Check for rays that have left the boundingBox... if yes, set their alive to False
        alive = nh.checkBoundary(new_positions, boundingBox, alive, geometry_type)

        # 3) Check for rays that have reached a new material, and assign their alive to False. Also, rays that have reached out of bounds should not be included in newMatMask.
        newMatMask = nh.checkForHit(new_positions, material_IDs, geometry_vectors, geometry_type)
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
        # OPL_lastStep = OPL_ta.read(step)
        # OPL_newMat = tf.boolean_mask(OPL_lastStep, newMatMask)
        PoyntingMag_newMat = tf.boolean_mask(PoyntingMag, newMatMask)
        material_IDs_newMat = tf.boolean_mask(material_IDs, newMatMask)
        group_IDs_newMat = tf.boolean_mask(group_IDs, newMatMask)
        parent_IDs_newMat = tf.boolean_mask(parent_IDs, newMatMask)
        ray_ordinary_consts_newMat = tf.boolean_mask(ray_ordinary_consts, newMatMask)

        newMat_IDs = nh.getMaterialsAtCoordinates(positions_newMat, geometry_vectors, geometry_type) # This is a list of the material IDs of the new materials that have just been reached.

        # Since there is only one type of interface now:
        Iso_Iso_mask = tf.ones(tf.shape(positions_newMat)[0], dtype=tf.bool) # This mask corresponds to rays that have hit an Isotropic-Isotropic interface, which is the only type of interface in this simulation since there is only one material.

        # Performing Isotropic_Isotropic interface analysis...
        # Create a Iso-Iso masked version of all variables:
        positions_Iso_Iso = tf.boolean_mask(positions_newMat, Iso_Iso_mask)
        p_Iso_Iso = tf.boolean_mask(p_newMat, Iso_Iso_mask)
        Efields_Iso_Iso = tf.boolean_mask(Efields_newMat, Iso_Iso_mask)
        # OPL_Iso_Iso = tf.boolean_mask(OPL_newMat, Iso_Iso_mask)
        PoyntingMag_Iso_Iso = tf.boolean_mask(PoyntingMag_newMat, Iso_Iso_mask)
        ray_ordinary_consts_Iso_Iso = tf.boolean_mask(ray_ordinary_consts_newMat, Iso_Iso_mask)
        group_IDs_Iso_Iso = tf.boolean_mask(group_IDs_newMat, Iso_Iso_mask)
        parent_IDs_Iso_Iso = tf.boolean_mask(parent_IDs_newMat, Iso_Iso_mask)
        prev_MatIDs_Iso_Iso = tf.boolean_mask(material_IDs_newMat, Iso_Iso_mask)
        new_MatIDs_Iso_Iso = tf.boolean_mask(newMat_IDs, Iso_Iso_mask)

        # Setting up and calling the Isotropic-Isotropic interface analysis function:
        surface_normals = nh.getSurfaceNormals(new_MatIDs_Iso_Iso, prev_MatIDs_Iso_Iso, geometry_vectors, positions_Iso_Iso, geometry_type)
        e_perp1, _, _, _ = dk.getOrdinaryPermittivities(positions_Iso_Iso, ray_ordinary_consts_Iso_Iso, dk_dist_flag)
        no1 = tf.math.sqrt(e_perp1)
        mat2_ordinary_consts = tf.gather(mat_ordinary_consts, new_MatIDs_Iso_Iso)
        e_perp2, _, _, _ = dk.getOrdinaryPermittivities(positions_Iso_Iso, mat2_ordinary_consts, dk_dist_flag)
        no2 = tf.math.sqrt(e_perp2)

        p_rII, p_tII, E_rII, E_tII, S_rII, S_tII = nh.Isotropic_Isotropic(surface_normals, no1, no2, p_Iso_Iso, Efields_Iso_Iso)
        S_rII = S_rII*PoyntingMag_Iso_Iso # Multiply the normalized Poynting vectors by the Poynting Vector magnitude of the corresponding incident ray
        S_tII = S_tII*PoyntingMag_Iso_Iso

        # Initialize the rest of the variables for the new reflected and transmitted rays:
        new_positions_reflectedII = positions_Iso_Iso
        new_positions_transmittedII = positions_Iso_Iso
        # new_OPL_reflectedII = OPL_Iso_Iso
        # new_OPL_transmittedII = OPL_Iso_Iso
        new_materialIDs_reflectedII = prev_MatIDs_Iso_Iso
        new_materialIDs_transmittedII = new_MatIDs_Iso_Iso
        group_IDs_reflectedII = group_IDs_Iso_Iso
        group_IDs_transmittedII = group_IDs_Iso_Iso
        parent_IDs_reflectedII = parent_IDs_Iso_Iso
        parent_IDs_transmittedII = parent_IDs_Iso_Iso
        ray_ordinary_consts_reflectedII = tf.gather(mat_ordinary_consts, new_materialIDs_reflectedII)
        ray_ordinary_consts_transmittedII = tf.gather(mat_ordinary_consts, new_materialIDs_transmittedII)

        # If IGNORE_REFLECTIONS is True, ensure all variables corresponding to reflected rays are empty:
        if IGNORE_REFLECTIONS:
            new_positions_reflectedII = tf.zeros_like(new_positions_reflectedII)[:0]
            # new_OPL_reflectedII = tf.zeros_like(new_OPL_reflectedII)[:0]
            p_rII = tf.zeros_like(p_rII)[:0]
            S_rII = tf.zeros_like(S_rII)[:0]
            E_rII = tf.zeros_like(E_rII)[:0]
            group_IDs_reflectedII = tf.zeros_like(group_IDs_reflectedII)[:0]
            parent_IDs_reflectedII = tf.zeros_like(parent_IDs_reflectedII)[:0]
            new_materialIDs_reflectedII = tf.zeros_like(new_materialIDs_reflectedII)[:0]
            ray_ordinary_consts_reflectedII = tf.zeros_like(ray_ordinary_consts_reflectedII)[:0]

        # Next, concatenate all the new variables of each type together:
        positions_combined = tf.concat([new_positions_reflectedII, new_positions_transmittedII], axis=0)
        # OPL_combined = tf.concat([new_OPL_reflectedII, new_OPL_transmittedII], axis=0)
        wave_vectors_combined = tf.concat([p_rII, p_tII], axis=0)
        PoyntingMag_combined = tf.concat([S_rII, S_tII], axis=0)
        Efields_combined = tf.concat([E_rII, E_tII], axis=0)
        group_IDs_combined = tf.concat([group_IDs_reflectedII, group_IDs_transmittedII], axis=0)
        parent_IDs_combined = tf.concat([parent_IDs_reflectedII, parent_IDs_transmittedII], axis=0)
        material_IDs_combined = tf.concat([new_materialIDs_reflectedII, new_materialIDs_transmittedII], axis=0)
        ray_ordinary_consts_combined = tf.concat([ray_ordinary_consts_reflectedII, ray_ordinary_consts_transmittedII], axis=0)

        # Create appropriately-sized alive_combined tensor (all new rays are set to True):
        alive_combined = tf.ones(tf.shape(positions_combined)[0], dtype=tf.bool)

        # Create a mask to eliminate rays with a sufficiently small Poyning Vector magnitude:
        small_magnitude_mask = tf.greater(PoyntingMag_combined, minimum_Poynting)

        # Apply the mask to all variables:
        positions_combined = tf.boolean_mask(positions_combined, small_magnitude_mask)
        # OPL_combined = tf.boolean_mask(OPL_combined, small_magnitude_mask)
        wave_vectors_combined = tf.boolean_mask(wave_vectors_combined, small_magnitude_mask)
        PoyntingMag_combined = tf.boolean_mask(PoyntingMag_combined, small_magnitude_mask)
        Efields_combined = tf.boolean_mask(Efields_combined, small_magnitude_mask)
        group_IDs_combined = tf.boolean_mask(group_IDs_combined, small_magnitude_mask)
        parent_IDs_combined = tf.boolean_mask(parent_IDs_combined, small_magnitude_mask)
        material_IDs_combined = tf.boolean_mask(material_IDs_combined, small_magnitude_mask)
        ray_ordinary_consts_combined = tf.boolean_mask(ray_ordinary_consts_combined, small_magnitude_mask)
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

        # Update the "global" (TensorArray) variables using scatterndupdate:
        pos_slice = positions_ta.read(step)
        pos_slice = tf.tensor_scatter_nd_update(pos_slice, indices1, positions_combined)
        positions_ta = positions_ta.write(step, pos_slice)

        wv_slice = wave_vectors_ta.read(step)
        wv_slice = tf.tensor_scatter_nd_update(wv_slice, indices1, wave_vectors_combined)
        wave_vectors_ta = wave_vectors_ta.write(step, wv_slice)

        Efields_slice = Efields_ta.read(step)
        Efields_slice = tf.tensor_scatter_nd_update(Efields_slice, indices1, Efields_combined)
        Efields_ta = Efields_ta.write(step, Efields_slice)

        # OPL_slice = OPL_ta.read(step)
        # OPL_slice = tf.tensor_scatter_nd_update(OPL_slice, indices1, OPL_combined)
        # OPL_ta = OPL_ta.write(step, OPL_slice)

        # Update standard tensor variables using scatterndupdate:
        PoyntingMag = tf.tensor_scatter_nd_update(PoyntingMag, indices1, PoyntingMag_combined)
        group_IDs = tf.tensor_scatter_nd_update(group_IDs, indices1, group_IDs_combined)
        parent_IDs = tf.tensor_scatter_nd_update(parent_IDs, indices1, parent_IDs_combined)
        material_IDs = tf.tensor_scatter_nd_update(material_IDs, indices1, material_IDs_combined)
        ray_ordinary_consts = tf.tensor_scatter_nd_update(ray_ordinary_consts, indicesCoefs, ray_ordinary_consts_combined)
        alive = tf.tensor_scatter_nd_update(alive, indices1, alive_combined)

        # 5) Increment the "step" variable:
        step = step + 1

    ### POST-PROCESSING AND LOSS CALCULATION: ###

    # Convert positions, wave_vectors, and Efields into rank 3 tensors for loss function computation:
    positions_final = tf.transpose(positions_ta.stack(), perm=[1, 2, 0])
    wave_vectors_final = tf.transpose(wave_vectors_ta.stack(), perm=[1, 2, 0])
    Efields_final = tf.transpose(Efields_ta.stack(), perm=[1, 2, 0])

    # Calculate and print focus loss value:
    loss = lf.planeWaveObjective(wave_vectors_final, group_IDs, material_IDs, Num_Starting_Rays)
    # theta, phi = nh.calcBeamAngles(wave_vectors_final, group_IDs, material_IDs, Num_Starting_Rays, specificGroup=1)
    tf.print("Value of Loss:", loss)

    return loss

######### --- Main Optimization Loop --- ###########

# Start timer:
start_time = time.perf_counter()

# ---------- USER INPUT STARTS HERE ---------- ###

# Define optimizer:
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

# Specify number of iterations
num_iterations = 15

# Air and Lens Material Coefficients:
materialCoefs_lens = tf.Variable([1.016, 8.934, -0.0174, -0.00254, 5.53]) # This serves as the starting guess for the optimization loop.

### ---------- USER INPUT ENDS HERE ---------- ###

materialCoefs_air = tf.Variable([1.0*1e-12, 1.0*1e-12, 1.0*1e-12, 1.0*1e-12, -100.0])
N = tf.cast(tf.shape(materialCoefs_lens)[0], dtype=tf.int32)

# Define a variable to store loss values:
loss_history = []
coefficient_history = []

# Running the training loop:
for i in range(num_iterations): 
    with tf.GradientTape() as tape:
        current_loss = rayTrace(materialCoefs_lens, materialCoefs_air)
    
    # # Calculate gradients using automatic differentiation:
    full_gradient = tape.gradient(current_loss, materialCoefs_lens)

    print(f"Current Gradient: {full_gradient}")

    # # # # Do not update the first 5 coefficients:
    # target_grads = full_gradient[5:]
    
    # # # Create a zeroed gradient for the full variable to keep indexing intact.
    # # # This ensures only the specific indices are updated.
    # update_grads = tf.zeros_like(materialCoefs_lens)
    # indices = tf.range(5, N)[:, None]
    # update_grads = tf.tensor_scatter_nd_update(update_grads, indices, target_grads)
    
    # 4. Apply to the full variable
    optimizer.apply_gradients([(full_gradient, materialCoefs_lens)])

    loss_history.append(current_loss.numpy())
    coefficient_history.append(materialCoefs_lens.numpy())

    print(f"Current Coefficients: {materialCoefs_lens.numpy()}")

print(f"Optimized loss: {loss_history[-1]:.4f}")
print(f"Optimized Coefficients: {materialCoefs_lens.numpy()}")

# End timer:
end_time = time.perf_counter()

# Calculate and print runtime:
duration = end_time - start_time
print(f"Execution took {duration:.4f} seconds")

# Plotting both loss over iteration #:

# Plot 0: loss
plt.figure(figsize=(6,4))
plt.plot(loss_history, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over Optimization Steps')
plt.grid(True)
plt.legend()
plt.show()

