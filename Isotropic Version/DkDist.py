import tensorflow as tf
import math as m

# Given a tensor of ray positions and coefficients described the ordinary profile distribution for each material (lens and air),
# this function returns four rank 1 tensors: e_perp, deperp_dx, deperp_dy, and deperp_dz.
# Important Note: Each material must have a basis function of the same form (the coefficients can be different).
# Another important note: The permittivity distriubtion is sometimes passed into a "softplus" function to ensure er >= 1.
# Note: The 'flag' input denotes the type of distribution selected by the user.
def getOrdinaryPermittivities(positions, ordinary_constants, flag):

    if flag == 1:
        # Unique Cylindrical Paramerization:
        # Coordinates:
        x = positions[:, 0] + 1e-12
        y = positions[:, 1] + 1e-12
        z = positions[:, 2] + 1e-12
        r = tf.math.sqrt(x**2 + y**2)
        # Optimzable parameters:
        A1 = ordinary_constants[:, 0] + 1e-12
        A2 = ordinary_constants[:, 1] + 1e-12
        C1 = ordinary_constants[:, 2] + 1e-12
        C2 = ordinary_constants[:, 3] + 1e-12
        emax = ordinary_constants[:, 4] + 1e-12
        # Permittivity + derivates:
        g = tf.math.sqrt(A1*r**2 + A2*z**2)
        h = emax + C1*g + C2*g**2
        e_perp = 1 + tf.math.softplus(h)
        de_dh = tf.math.sigmoid(h)
        dh_dg = C1 + 2*C2*g
        dg_dr = A1*r/g
        dg_dz = A2*z/g
        dr_dx = x/r
        dr_dy = y/r
        deperp_dx = de_dh * dh_dg * dg_dr * dr_dx
        deperp_dy = de_dh * dh_dg * dg_dr * dr_dy
        deperp_dz = de_dh * dh_dg * dg_dz
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    if flag == 2:
        # Quadractic as a function of r (spherical coordinates):
        erb = ordinary_constants[:, 0]
        alpha = ordinary_constants[:, 1]
        r = tf.math.sqrt(x**2 + y**2 + z**2)
        e_perp = erb*(1-alpha**2 * r**2)
        deperp_dr = -2.0*erb*alpha**2*r
        deperp_dx = deperp_dr*(x/r)
        deperp_dy = deperp_dr*(y/r)
        deperp_dz = deperp_dr*(z/r)
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    if flag == 3:
        # Mikaelian Lens formula:
        x = positions[:, 0]
        y = positions[:, 1]
        rho = tf.math.sqrt(x**2 + y**2)
        a = ordinary_constants[:, 0] # Lens radius
        d = ordinary_constants[:, 1] # Lens thickness
        no = tf.math.cosh(0.5*m.pi*a/d) # Maximum refractive index
        e_perp = (no/tf.math.cosh(0.5*m.pi*rho/d))**2
        # Derivatives:
        de_dn = 2*(no/tf.math.cosh(0.5*m.pi*rho/d))
        dn_drho = no * (-0.5*m.pi/d) * tf.math.sinh(0.5*m.pi*rho/d) / (tf.math.cosh(0.5*m.pi*rho/d))**2
        drho_dx = x/rho
        drho_dy = y/rho
        deperp_dx = de_dn * dn_drho * drho_dx
        deperp_dy = de_dn * dn_drho * drho_dy
        deperp_dz = tf.zeros_like(deperp_dy)
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    if flag == 4:
        # 4th-degree polynomial as a function of x:
        x = positions[:, 0] # for readability
        g = ordinary_constants[:, 0] + ordinary_constants[:, 1]*x + ordinary_constants[:, 2]*x**2 + ordinary_constants[:, 3]*x**3 + ordinary_constants[:, 4]*x**4 # g is an intermediate parameterization
        e_perp = 1 + tf.math.softplus(g)
        deperp_dx = tf.math.sigmoid(g)*(ordinary_constants[:, 1] + 2.0*ordinary_constants[:, 2]*x + 3.0*ordinary_constants[:, 3]*x**2 + 4.0*ordinary_constants[:, 4]*x**3)
        deperp_dy = tf.zeros(tf.shape(positions)[0])
        deperp_dz = tf.zeros(tf.shape(positions)[0])
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    if flag == 5:
        # Quadractic as a function of rho:
        x = positions[:, 0]
        y = positions[:, 1] 
        rho = x**2 + y**2 
        e_max = ordinary_constants[:, 0] 
        rho_max = ordinary_constants[:, 1]
        alpha = ordinary_constants[:, 2] # for readability
        g = e_max*(1 - alpha*(rho/rho_max)**2) # g is an intermediate variable
        e_perp = 1 + tf.math.softplus(g)
        deperp_dx = tf.math.sigmoid(g)*(-2.0*e_max*alpha*x/rho_max)
        deperp_dy = tf.math.sigmoid(g)*(-2.0*e_max*alpha*y/rho_max)
        deperp_dz = tf.zeros(tf.shape(positions)[0])
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    if flag == 6:
        # Quadractic as a function of rho multiplied by a double sigmoid as a function of z:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        rho = tf.math.sqrt(x**2 + y**2)
        erb = ordinary_constants[:, 0]
        e_max = ordinary_constants[:, 1] 
        rho_max = ordinary_constants[:, 2]
        alpha = ordinary_constants[:, 3] 
        C = ordinary_constants[:, 4]
        zmin = ordinary_constants[:, 5]
        zmax = ordinary_constants[:, 6] # for readability
        g_rho = e_max*(1 - alpha*(rho/rho_max)**2) # g_rho and g_z are intermediate variables
        g_z = tf.math.sigmoid(C*z - zmin) - tf.math.sigmoid(C*z - zmax)
        e_perp = erb + g_z*tf.math.softplus(g_rho)
        deperp_dx = g_z*tf.math.sigmoid(g_rho)*(-2.0*e_max*alpha*x/rho_max**2)
        deperp_dy = g_z*tf.math.sigmoid(g_rho)*(-2.0*e_max*alpha*y/rho_max**2)
        deperp_dz = tf.math.softplus(g_rho)*C*(tf.math.sigmoid(C*z - zmin)*(1 - tf.math.sigmoid(C*z - zmin)) - (tf.math.sigmoid(C*z - zmax)*(1 - tf.math.sigmoid(C*z - zmax))))
        return e_perp, deperp_dx, deperp_dy, deperp_dz

    if flag == 7:
        # N-harmonic Fourier series as a function of rho multiplied by a double sigmoid as a function of z:
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        rho = tf.math.sqrt(x**2 + y**2)
        rho = rho + 1e-12 # To prevent NaNs in the derivative when rho = 0
        erb = ordinary_constants[:, 0] 
        C = ordinary_constants[:, 1]
        zmin = ordinary_constants[:, 2]
        zmax = ordinary_constants[:, 3] # for readability
        w0 = ordinary_constants[:, 4]
        a0 = ordinary_constants[:, 5]
        an = ordinary_constants[:, 6:]
        N = tf.shape(an)[1]
        N = tf.cast(tf.squeeze(N), dtype=tf.int32)
        n_vec = tf.cast(tf.range(1, N+1)[:, tf.newaxis], dtype=tf.float32)

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

    if flag == 8:
        # N-harmonic Fourier series as a function of rho multiplied by an N-harmonic FS as a function of z (cosines only):

        # Fourier Series in rho:
        x = positions[:, 0]
        y = positions[:, 1]
        rho = tf.math.sqrt(x**2 + y**2)
        rho_constants = ordinary_constants[:, :tf.cast(tf.shape(ordinary_constants)[1]/2, dtype=tf.int32)]
        N = tf.shape(rho_constants)[1] - 2
        N = tf.cast(tf.squeeze(N), dtype=tf.int32)
        n_vec = tf.cast(tf.range(1, N+1)[:, tf.newaxis], dtype=tf.float32)
        T0 = rho_constants[:, 0]
        a0 = rho_constants[:, 1]
        an = rho_constants[:, 2:N+2]
        # bn = ordinary_constants[:, N+2:2*N + 2] # Don't need since we want an even function

        # Generate the cos and sin matrices:
        w0 = 2*m.pi/T0

        # Use broadcasting to get shape (batch, N)
        angle = tf.transpose(n_vec * tf.transpose(tf.expand_dims(w0*rho, axis=1)))

        cos_terms = an * tf.cos(angle) # (batch, N)
        # sin_terms = bn * tf.sin(angle) # (batch, N) # Don't need since we want an even function

        a_sum = tf.reduce_sum(cos_terms, axis=1) # (batch,)
        # b_sum = tf.reduce_sum(sin_terms, axis=1) # (batch,) # Don't need since we want an even function

        g_rho = a0 + a_sum # g is an intermediate variable

        # Pre-calculate the frequencies for the derivative: (n * w0)
        # Shape: (batch, N)
        frequencies = tf.expand_dims(w0, axis=1)*tf.transpose(n_vec)

        # Compute the derivative of the sum element-wise
        # d/drho [an * cos(n*w0*rho)] = -an * n*w0 * sin(n*w0*rho)
        da_drho = an * frequencies * tf.sin(angle)
        # db_drho =  bn * frequencies * tf.cos(angle) # Don't need since we want an even function

        # Sum across the N components to get (batch,)
        dg_drho = tf.reduce_sum(da_drho, axis=1)

        # Fourier Series in z:
        z = positions[:, 2]
        z_constants = ordinary_constants[:, tf.cast(tf.shape(ordinary_constants)[1]/2, dtype=tf.int32):]
        N = tf.shape(z_constants)[1] - 2
        N = tf.cast(tf.squeeze(N), dtype=tf.int32)
        n_vec = tf.cast(tf.range(1, N+1)[:, tf.newaxis], dtype=tf.float32)
        T0 = z_constants[:, 0]
        a0 = z_constants[:, 1]
        an = z_constants[:, 2:N+2]
        # bn = ordinary_constants[:, N+2:2*N + 2] # Don't need since we want an even function

        # Generate the cos and sin matrices:
        w0 = 2*m.pi/T0

        # Use broadcasting to get shape (batch, N)
        angle = tf.transpose(n_vec * tf.transpose(tf.expand_dims(w0*rho, axis=1)))

        cos_terms = an * tf.cos(angle) # (batch, N)
        # sin_terms = bn * tf.sin(angle) # (batch, N) # Don't need since we want an even function

        a_sum = tf.reduce_sum(cos_terms, axis=1) # (batch,)
        # b_sum = tf.reduce_sum(sin_terms, axis=1) # (batch,) # Don't need since we want an even function

        g_z = a0 + a_sum # g is an intermediate variable

        # Pre-calculate the frequencies for the derivative: (n * w0)
        # Shape: (batch, N)
        frequencies = tf.expand_dims(w0, axis=1)*tf.transpose(n_vec)

        # Compute the derivative of the sum element-wise
        # d/drho [an * cos(n*w0*rho)] = -an * n*w0 * sin(n*w0*rho)
        da_dz = an * frequencies * tf.sin(angle)
        # db_drho =  bn * frequencies * tf.cos(angle) # Don't need since we want an even function

        # Sum across the N components to get (batch,)
        dg_dz = tf.reduce_sum(da_drho, axis=1)

        e_perp = 1.0 + tf.math.softplus(g_rho)*tf.math.softplus(g_z) # Minimum er of 4 (based on substrate)

        # Chain rule for x, y, and z
        # Note: Softplus derivative is sigmoid

        deperp_dx = tf.math.softplus(g_z)*tf.math.sigmoid(g_rho)*dg_drho*(x/rho)
        deperp_dy = tf.math.softplus(g_z)*tf.math.sigmoid(g_rho)*dg_drho*(y/rho)
        deperp_dz = tf.math.softplus(g_rho)*tf.math.sigmoid(g_z)*dg_dz

        return e_perp, deperp_dx, deperp_dy, deperp_dz