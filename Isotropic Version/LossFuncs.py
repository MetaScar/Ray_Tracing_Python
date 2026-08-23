import tensorflow as tf

# This file contains all loss functions that have been written for the ray-tracing algorithm.
# New loss functions can be added as needed.

# TEST Loss function. This is a dummy loss function to test issues with backpropagation through the ray tracing algorithm.
def dummyLoss(wave_vectors):
    return tf.reduce_sum(wave_vectors)

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
def focusObjective(positions, wave_vectors, material_IDs, group_IDs, Num_Starting_Rays, focal_plane):
    
    # First, we eliminate the starting rays from consideration:
    N = Num_Starting_Rays
    positions = positions[N:]
    wave_vectors = wave_vectors[N:]
    group_IDs = group_IDs[N:]
    material_IDs = material_IDs[N:]

    # Only consider rays propagating in the air past the lens.
    # A boolean mask must be applied to positions to achieve this:
    material_mask = material_IDs == 1

    # Apply the boolean mask to positions and wave_vectors:
    positions = tf.boolean_mask(positions, material_mask)
    wave_vectors = tf.boolean_mask(wave_vectors, material_mask)
    group_IDs = tf.boolean_mask(group_IDs, material_mask)

    # Identify all unique group IDs:
    unique_ids, _ = tf.unique(group_IDs)

    # Initialize total loss to zero:
    total_loss = 0.0

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

    for g_id in unique_ids:
        # Extract focal plane positions for specific group:
        xpos_group = tf.boolean_mask(xpos, group_IDs == g_id)
        ypos_group = tf.boolean_mask(ypos, group_IDs == g_id)
        # Calculate variance along x and y:
        var_x = tf.math.reduce_variance(xpos_group)
        var_y = tf.math.reduce_variance(ypos_group)
        total_loss = total_loss + var_x + var_y

    return total_loss