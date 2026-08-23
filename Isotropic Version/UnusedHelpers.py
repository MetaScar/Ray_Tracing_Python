import tensorflow as tf
import tensorflow_probability as tfp
import math as m
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import DkDist as dk

# These are helper functions I have written that are not currently being used by the ray-tracing algorithm.
# The functions are kept in this seperate Python file for clarity.

# This function computes a single time step for ALL rays at once, whether they are alive or not.
# This step includes the computation of the optical path length (OPL), which is not currently being using by the RT algorithm.
def rayPropagationOPL(positions, wave_normals, Efields, OPL, alive, ordinary_constants, material_IDs, distance_step, interpolation_step):

    # Calculate e_perp, e_parallel, the director, and associated spatial derivatives for all rays:
    e_perp, deperp_dx, deperp_dy, deperp_dz = dk.getOrdinaryPermittivities(positions, ordinary_constants)

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

    # Calculate the Optical Path Length (OPL) for each ray...
    # Interpolate positions and calculate er at these positions for improved accuracy:
    interpolated_positions = tf.linspace(positions, new_positions, interpolation_step, axis=1)
    interped_positions_flat = tf.reshape(interpolated_positions, [tf.shape(interpolated_positions)[0] * tf.shape(interpolated_positions)[1], 3])
    ord_consts_flat= tf.repeat(ordinary_constants, repeats=interpolation_step, axis=0)
    er_interped_flat, _, _, _ = dk.getOrdinaryPermittivities(interped_positions_flat, ord_consts_flat)
    er_interped = tf.reshape(er_interped_flat, tf.shape(interpolated_positions)[:2])
    # Calculate n = sqrt(er):
    index_interped = tf.math.sqrt(er_interped)
    # Calculate the interpolated distances (from starting position)
    og_positions_repeated = tf.repeat(positions, repeats=interpolation_step, axis=0)
    og_positions_flat = tf.reshape(og_positions_repeated, [tf.shape(interpolated_positions)[0], tf.shape(interpolated_positions)[1], 3])
    interpolated_distances = tf.norm(interpolated_positions - og_positions_flat, axis=2)
    # Perform trapezoidal integration of n*d to find differential OPL
    diff_OPL = tfp.math.trapz(y=index_interped, x=interpolated_distances, axis=1)
    # Add to previous OPL:
    new_OPL = OPL + diff_OPL

    # Calculate the updated E-field magnitude for each ray (to account for wavefront curvature):
    # DF = computeDF()

    # Only update rays that are "alive" (i.e. still being traced):
    alive_expanded = tf.expand_dims(alive, axis=1) # Turns alive from size N to size (N, 1)
    new_wave_normals = tf.where(alive_expanded, new_wave_normals, wave_normals)
    new_positions = tf.where(alive_expanded, new_positions, positions)
    new_Efields = tf.where(alive_expanded, E, Efields)
    new_OPL = tf.where(alive, new_OPL, OPL)
    
    return new_wave_normals, new_positions, new_Efields, new_OPL

# This function is identical to the rayPropagation() function, except it uses the 4th-order Runge-Kutta method instead of the first-order RK method.
# Note that polarization is not correctly accounted for yet.
# Note that OPL is not correctly accounted for yet.
def rayPropagation_RK4(positions, wave_normals, Efields, alive, ordinary_constants, material_IDs, distance_step):

    # Calculate e_perp and associated spatial derivatives for all rays:
    e, de_dx, de_dy, de_dz = dk.getOrdinaryPermittivities(positions, ordinary_constants)
    
    # RK4 method implementation...
    # Compute k1, y1, for all 6 variables:
    k1_px = de_dx
    k1_py = de_dy
    k1_pz = de_dz
    k1_rx = 2*wave_normals[:, 0]
    k1_ry = 2*wave_normals[:, 1]
    k1_rz = 2*wave_normals[:, 2]
    # Use the (fixed) distance step to come up with an h (which will be used for the entirely of the RK4 method):
    current_step = tf.math.sqrt(k1_rx**2 + k1_ry**2 + k1_rz**2)
    h = distance_step/(current_step + 1e-12)
    y1_px = wave_normals[:, 0] + (0.5*h*k1_px)
    y1_py = wave_normals[:, 1] + (0.5*h*k1_py)
    y1_pz = wave_normals[:, 2] + (0.5*h*k1_pz)
    y1_rx = positions[:, 0] + (0.5*h*k1_rx)
    y1_ry = positions[:, 1] + (0.5*h*k1_ry)
    y1_rz = positions[:, 2] + (0.5*h*k1_rz)
    # Find y1_e, y1_de_di:
    y1_e, y1_de_dx, y1_de_dy, y1_de_dz = dk.getOrdinaryPermittivities(tf.stack([y1_rx, y1_ry, y1_rz], axis=1), ordinary_constants)
    # Compute k2, y2, for all 6 variables:
    k2_px = y1_de_dx
    k2_py = y1_de_dy
    k2_pz = y1_de_dz
    k2_rx = 2*y1_px
    k2_ry = 2*y1_py
    k2_rz = 2*y1_pz
    y2_px = wave_normals[:, 0] + (0.5*h*k2_px)
    y2_py = wave_normals[:, 1] + (0.5*h*k2_py)
    y2_pz = wave_normals[:, 2] + (0.5*h*k2_pz)
    y2_rx = positions[:, 0] + (0.5*h*k2_rx)
    y2_ry = positions[:, 1] + (0.5*h*k2_ry)
    y2_rz = positions[:, 2] + (0.5*h*k2_rz)
    # Find y2_e, y2_de_di:
    y2_e, y2_de_dx, y2_de_dy, y2_de_dz = dk.getOrdinaryPermittivities(tf.stack([y2_rx, y2_ry, y2_rz], axis=1), ordinary_constants)
    # Compute k3, y3, for all 6 variables:
    k3_px = y2_de_dx
    k3_py = y2_de_dy
    k3_pz = y2_de_dz
    k3_rx = 2*y2_px
    k3_ry = 2*y2_py
    k3_rz = 2*y2_pz
    y3_px = wave_normals[:, 0] + h*k3_px
    y3_py = wave_normals[:, 1] + h*k3_py
    y3_pz = wave_normals[:, 2] + h*k3_pz
    y3_rx = positions[:, 0] + h*k3_rx
    y3_ry = positions[:, 1] + h*k3_ry
    y3_rz = positions[:, 2] + h*k3_rz
    # Find y3_e, y3_de_di:
    y3_e, y3_de_dx, y3_de_dy, y3_de_dz = dk.getOrdinaryPermittivities(tf.stack([y3_rx, y3_ry, y3_rz], axis=1), ordinary_constants)
    # Compute k4 for all variables:
    k4_px = y3_de_dx
    k4_py = y3_de_dy
    k4_pz = y3_de_dz
    k4_rx = 2*y3_px
    k4_ry = 2*y3_py
    k4_rz = 2*y3_pz
    # Compute m for all variables:
    m_px = (k1_px + 2*k2_px + 2*k3_px + k4_px)/6
    m_py = (k1_py + 2*k2_py + 2*k3_py + k4_py)/6
    m_pz = (k1_pz + 2*k2_pz + 2*k3_pz + k4_pz)/6
    m_rx = (k1_rx + 2*k2_rx + 2*k3_rx + k4_rx)/6
    m_ry = (k1_ry + 2*k2_ry + 2*k3_ry + k4_ry)/6
    m_rz = (k1_rz + 2*k2_rz + 2*k3_rz + k4_rz)/6

    # Compute h such that rays travel a fixed distance step:
    current_step = tf.math.sqrt(m_rx**2 + m_ry**2 + m_rz**2)
    current_step = current_step + 1e-12 # To prevent NaNs when the ray is stationary (i.e. current_step = 0)
    h = distance_step/current_step
    h = tf.expand_dims(h, axis=1)
    # Time-step using the first-order Runge-Kutta method:
    new_wave_normals = wave_normals + (h*(tf.stack([m_px, m_py, m_pz], axis=1)))
    new_positions = positions + (h*(tf.stack([m_rx, m_ry, m_rz], axis=1)))

    # Calculate the new E-field vectors for each ray:
    ## THIS NEEDS TO BE UPDATED!
    E = Efields

    # Only update rays that are "alive" (i.e. still being traced):
    alive_expanded = tf.expand_dims(alive, axis=1) # Turns alive from size N to size (N, 1)
    new_wave_normals = tf.where(alive_expanded, new_wave_normals, wave_normals)
    new_positions = tf.where(alive_expanded, new_positions, positions)
    new_Efields = tf.where(alive_expanded, E, Efields)
    
    return new_wave_normals, new_positions, new_Efields

# This function takes in a set of 3 ray positions (1 axial, 2 paraxial) along with their associated directions.
# It then computes and returns the associated principle curvaatures k1 and k2.
def computeCurvature(positions, wave_vectors, positions_parax1, wave_vectors_parax1, positions_parax2, wave_vectors_parax2):
    # First, all wave_vectors are normalized to make them unit vectors:
    t0 = wave_vectors/tf.norm(wave_vectors, axis=1)
    t1 = wave_vectors_parax1/tf.norm(wave_vectors_parax1, axis=1)
    t2 = wave_vectors_parax2/tf.norm(wave_vectors_parax2, axis=1)
    # We then compute the unit vectors u, v, tu, and tv:
    u = (positions_parax1-positions)/tf.norm(positions_parax1-positions, axis=1)
    v = (positions_parax2-positions)/tf.norm(positions_parax2-positions, axis=1)
    tu = (t1-t0)/tf.norm(positions_parax1-positions, axis=1)
    tv = (t2-t0)/tf.norm(positions_parax2-positions, axis=1)
    # Next, compute the parameters of the first and second fundamental form (F, e, f, g):
    F = tf.reduce_sum(u*v, axis=1)
    e = tf.reduce_sum(u*tu, axis=1)
    f = 0.5*tf.reduce_sum(u*tv, axis=1) + 0.5*tf.reduce_sum(v*tu, axis=1)
    g = tf.reduce_sum(v*tv, axis=1)
    # We then calculate the intermediate variables A, B, and q:
    A = (g + e - 2*f*F)/(2*(1-F**2))
    B = (e*g-f**2)/(1-F**2)
    q = A + tf.math.sign(A)*tf.math.sqrt(A**2 - B)
    # Finally, compute the principle curvatures k1 and k2:
    k1 = q
    k2 = B/q
    return k1, k2

### All functions below this point were written by the Github Coplit AI. They may need to be rewritten. ###
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

    # Extract the positions and trajectories corresponding to the STEPs at which the distance to the focal plane is a minimum:
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
    # Plot the radius of the lens for reference (assuming a circular lens with radius 8 cm):
    circle = plt.Circle((0, 0), 8, fill=False, color='black')
    plt.gca().add_patch(circle)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
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

    exponent = -1j * tf.cast(kx_expanded * x_coords + ky_expanded * y_coords, tf.complex64) # Shape: [num_parents, 180, 360]
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

### Post-Processing code that leverages the new post-processing functions:
# # Compute constructCompleteTensors() function:
# unique_parent_IDs, group_IDs_by_parent, positions_by_parent, wave_vectors_by_parent, Efields_by_parent, PoyntingMags_by_parent = nh.constructCompleteTensors(positions_final, wave_vectors_final, Efields_final, PoyntingMag, group_IDs, parent_IDs)

# # Interpolate the phase, position, E-field, and wave vector of each ray at regular intervals along its path:
# phases_interp, positions_interp, Efields_interp, wave_vectors_interp = nh.computePhase(positions_by_parent, wave_vectors_by_parent, Efields_by_parent, wavelength=wavelength, interpolation_step=interpolation_step)

# # Compute the phase, position, E-field, and wave vector at a specified z=constant plane:
# intersected, xy_positions, phases_at_plane, Efields_at_plane, wave_vectors_at_plane = nh.computeAtPlane(phases_interp, positions_interp, Efields_interp, wave_vectors_interp, z_target=6.0)
# # (Manually) compute the Poynting Magnitudes at the plane by knowing which material the plane is in:
# dense_poynting = PoyntingMags_by_parent.to_tensor(default_value=0.0)
# S_mags_at_plane = dense_poynting[:, 2] # Since the plane is past the lens (third ray per parent)

# # Compute fx and fy (this is a test):
# U_normalized = nh.computeFarField(xy_positions, phases_at_plane, Efields_at_plane, S_mags_at_plane, wave_vectors_at_plane, wavelength=wavelength)

# # Plot the normalized far field pattern (phi=0 cut):
# U_phi0 = U_normalized[:, 0]
# U_phi_dB = 10.0*tf.math.log(U_phi0)/tf.math.log(10.0) # Convert to dB scale
# theta = tf.linspace(0.0, 180.0, tf.shape(U_phi0)[0])
# plt.plot(theta.numpy(), U_phi_dB.numpy())
# plt.xlabel("Theta (degrees)")
# plt.ylabel("Normalized Directivity (dBi)")
# plt.title("Far Field Radiation Pattern (Phi = 0)")
# plt.xlim(0, 90)
# plt.grid()
# plt.savefig("far_field_pattern.png", dpi=300)

# # Save the far field data to a CSV file:
# with open("Directivity.csv", "w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file)
#     for item in U_phi_dB.numpy():
#         writer.writerow([item])  # Wraps item in a list to make it a column

# # Plotting only a single (complete) ray:
# x_coords = positions_by_parent[0,:,0].numpy()
# y_coords = positions_by_parent[0,:,1].numpy()
# z_coords = positions_by_parent[0,:,2].numpy()

# ax.plot3D(x_coords, y_coords, z_coords, color='blue', linewidth=2)