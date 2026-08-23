import tensorflow as tf
import math as m

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

# This function creates a number of incident rays corresponding to a plane wave at a particular angle.
# The plane wave is centered on the origin when it strikes the lens. The plane wave will hit the lens with a square area according to the "sideLength" parameter.
# Note that theta is defined as the angle from the z axis to the ray direction vector.
def createPlaneWave(sideLength, theta, raysPerLength, lensBottom, distToLens):
    # Convert theta from degrees to radians:
    theta = theta * (m.pi/180.0)
    # Create the initial ray positions:
    x = tf.linspace(-sideLength/2, sideLength/2, raysPerLength)
    x = x - distToLens*tf.math.tan(theta) # To account for incidence angle
    y = tf.linspace(-sideLength/2, sideLength/2, raysPerLength)
    X, Y = tf.meshgrid(x, y, indexing='ij')
    X = tf.reshape(X, [-1])
    Y = tf.reshape(Y, [-1])
    Z = lensBottom - distToLens + tf.zeros_like(X)
    positions = tf.stack([X, Y, Z], axis=1)
    # Create wave_vectors based on theta:
    p_inc = [[1e-9 + tf.math.sin(theta), 1e-9 + 0.0, 1e-9 + tf.math.cos(theta)]]
    wave_vectors = tf.repeat(p_inc, repeats=tf.shape(positions)[0], axis=0)

    return positions, wave_vectors

# Given the illumination side length, angles of incidence for each plane wave, # of rays per length, position of the bottom of the lens, and
# the z distance to the lens, this function returns all variables needed to initialize the plane waves.
def createPlaneWaves(sideLength, theta_wave, raysPerLength, lensBottom, distToLens, ordinary_consts):
    Num_waves = tf.shape(theta_wave)[0]
    raysPerWave = raysPerLength**2
    # Initialize position and wave_vectors as empty Python list:
    positions_list = []
    wave_vectors_list = []
    # Loop through theta_wave, creating appropriate positions and wave_vectors for each plane wave,
    # then appending them to the lists:
    for i in range(Num_waves):
        curr_positions, curr_wave_vectors = createPlaneWave(sideLength, theta_wave[i], raysPerLength, lensBottom, distToLens)
        positions_list.append(curr_positions)
        wave_vectors_list.append(curr_wave_vectors)

    # Combine all slices along the first axis:
    positions = tf.concat(positions_list, axis=0)
    wave_vectors = tf.concat(wave_vectors_list, axis=0)

    # Initialize other variables:
    Poynting_mag = tf.ones(Num_waves*raysPerWave, dtype=tf.float32)
    alive = tf.ones(Num_waves*raysPerWave, dtype=tf.bool)
    Efields = tf.repeat([[0.0, 1.0, 0.0]], Num_waves*raysPerWave, axis=0)
    material_IDs = tf.ones(Num_waves*raysPerWave, dtype=tf.int32)
    ray_ordinary_consts = tf.transpose(tf.tile(tf.expand_dims(ordinary_consts[1,:], axis=1), multiples=[1,Num_waves*raysPerWave]))

    return positions, wave_vectors, Poynting_mag, alive, Efields, material_IDs, ray_ordinary_consts

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
    Z = tf.ones_like(X) * center_location[2]
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

# Given a number of rays to create, the maximum theta angle, the position of the sphere center, and the polarization
# of rays, this function returns all the tensors needed to intialize the spherical cap of rays.
# UPDATE: Each input is now a tensor corrponding to the variable value for each feed location.
def createIsotropicRays(NumberOfRays, theta_max_deg, theta_target_deg, sphere_center, Epol, ordinary_consts):

    ### Step 1 - Calculate the wave vectors:
    N = NumberOfRays # Number of rays
    N = tf.cast(N, dtype=tf.float32)

    # Find number of feeds from length of theta_max_deg:
    numFeeds = tf.shape(theta_max_deg)[0] # Use this for one than one point source
    
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
    Efields = tf.expand_dims(Epol, axis=1)
    # Efields = tf.expand_dims(Efields, axis=0)
    Efields = tf.tile(Efields, [1, NumberOfRays[0], 1])
    Efields = tf.reshape(Efields, [-1, 3])

    ### Step 3 - Calculate all other quantities:
    sphere_center = tf.expand_dims(sphere_center, axis=1)
    positions = tf.tile(sphere_center, [1, NumberOfRays[0], 1])
    positions = tf.reshape(positions, [-1, 3])

    totalNumRays = tf.cast(NumberOfRays[0], tf.int32)*numFeeds
    PoyntingMag = tf.ones([totalNumRays], dtype=tf.float32)
    alive = tf.ones([totalNumRays], dtype=tf.bool)
    material_IDs = tf.ones([totalNumRays], dtype=tf.int32)
    ray_ordinary_consts = tf.transpose(tf.tile(tf.expand_dims(ordinary_consts[1,:], axis=1), multiples=[1,totalNumRays]))

    return positions, wave_vectors, PoyntingMag, alive, Efields, material_IDs, ray_ordinary_consts