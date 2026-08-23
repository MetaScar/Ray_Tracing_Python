import tensorflow as tf
import matplotlib.pyplot as plt

# This function plots all ray paths, as well as a transparent cylinder to visualize the location of the lens.
def plotRaysAndCyl(radius, height, z_min, Num_rays_active, step, positions_final, group_IDs, specificGroupID, xy_max, z_max):

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plotting semi-transparent cylinder to help visualize locaiton of lens:

    # Cylinder parameters
    radius = tf.constant(48.5, dtype=tf.float32)
    height = tf.constant(34.0, dtype=tf.float32)
    z_min = tf.constant(-17.0, dtype=tf.float32)

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

    # Plotting the ray paths:
    for i in range(Num_rays_active):
        # Extract the trajectory for the current ray [3, step-1]
        path = positions_final[i, :, :step-1]

        # Only plotting rays from a particular feed:
        if (group_IDs[i] == specificGroupID):
            # Create a mask: True if any coordinate (x,y,z) at that step is non-zero
            # We check across the coordinate axis (axis 0)
            mask = tf.reduce_any(path != 0, axis=0).numpy()
            
            # Apply the mask to x, y, and z coordinates
            z_coords = path[2, :][mask]
            y_coords = path[1, :][mask]
            x_coords = path[0, :][mask]

            color = 'blue'
            ax.plot3D(x_coords, y_coords, z_coords, color=color)

    plt.title("3D Ray Propagation")
    plt.grid(True)
    ax.set_xlim(-xy_max, xy_max)
    ax.set_ylim(-xy_max, xy_max)
    ax.set_zlim(-z_max, z_max)
    ax.set_box_aspect([1, 1, 1])

    plt.show()
    plt.savefig("3D_Ray_Propagation.png", dpi=300)