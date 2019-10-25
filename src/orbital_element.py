"""
Harvard IACS Masters Thesis
Orbital Elements

Michael S. Emanuel
Wed Jul 10 13:44:54 2019
"""

# Library imports
import tensorflow as tf
from tensorflow import keras
import rebound
import numpy as np

# Local imports
from utils import range_inc
from tf_utils import Identity

# ********************************************************************************************************************* 
# Set autograph logging verbosity
tf.autograph.set_verbosity(0)

# ********************************************************************************************************************* 
# The gravitational constant in unit system (years, AU, Msun)
# numerical value close to 4 pi^2; see rebound documentation for exact value        
G_ = 39.476926421373

# ********************************************************************************************************************* 
# Data sets for testing orbital element conversions.
# Simple approach, just wraps calls to rebound library
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_data_orb_elt(n, a_min, a_max, e_max, inc_max, seed=42):
    """Data with mapping between orbital elements and configuration space."""
    # Set random seed
    np.random.seed(seed=seed)

    # Initialize orbital element by sampling according to the inputs
    a = np.random.uniform(low=a_min, high=a_max, size=n).astype(np.float32)
    e = np.random.uniform(low=0.0, high=e_max, size=n).astype(np.float32)
    inc = np.random.uniform(low=0.0, high=inc_max, size=n).astype(np.float32)
    Omega = np.random.uniform(low=-np.pi, high=np.pi, size=n).astype(np.float32)
    omega = np.random.uniform(low=-np.pi, high=np.pi, size=n).astype(np.float32)
    f = np.random.uniform(low=-np.pi, high=np.pi, size=n).astype(np.float32)
    
    # Initialize cartesian entries to zero vectors; these are placeholders
    qx = np.zeros(n, dtype=np.float32)
    qy = np.zeros(n, dtype=np.float32)
    qz = np.zeros(n, dtype=np.float32)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    vz = np.zeros(n, dtype=np.float32)
    accx = np.zeros(n, dtype=np.float32)
    accy = np.zeros(n, dtype=np.float32)
    accz = np.zeros(n, dtype=np.float32)
    
    # Create a simulation
    sim = rebound.Simulation()

    # Set units
    sim.units = ('yr', 'AU', 'Msun')

    # Add primary with 1 solar mass at origin with 0 velocity
    sim.add(m=1.0)
    
    # The gravitational constant mu as a scalar; assume the small particles have mass 0
    mu0 = sim.G * sim.particles[0].m
    # The gravitaional constant as a vector
    mu = mu0 * np.ones(n, dtype=np.float32)

    # Create particles with these orbital elements
    for i in range(n):
        # Create the new particle
        sim.add(m=0.0, a=a[i], e=e[i], inc=inc[i], Omega=Omega[i], omega=omega[i], f=f[i])
        # The coordinates of the new particle
        p = sim.particles[i+1]
        # Save coordinates of new particle
        qx[i], qy[i], qz[i] = p.x, p.y, p.z
        vx[i], vy[i], vz[i] = p.vx, p.vy, p.vz
    
    # Integrate forward, then backwards; this will get the acceleration primed
    sim.integrate(-1E-6, exact_finish_time=1)
    sim.integrate(1E-6, exact_finish_time=1)

    for i in range(n):
        p = sim.particles[i+1]
        accx[i], accy[i], accz[i] = p.ax, p.ay, p.az
    
    # Stack the position and velocity vectors
    q = np.stack([qx, qy, qz], axis=1)
    v = np.stack([vx, vy, vz], axis=1)
    acc = np.stack([accx, accy, accz], axis=1)
    
    # Dictionaries with elements and cartesian coordinates
    elts = {
        'a': a,
        'e': e,
        'inc': inc,
        'Omega': Omega,
        'omega': omega,
        'f': f,
        'mu': mu,
    }
    
    cart = {
        'q': q,
        'v': v,
        'acc': acc,
        'mu': mu,
    }
    
    return elts, cart

# ********************************************************************************************************************* 
def make_dataset_elt_to_cfg(n, a_min, a_max, e_max, inc_max, seed=42, batch_size=64):
    """Dataset with mapping from orbital elements to configuration space."""
    # Build data set as dictionaries of numpy arrays
    elts, cart = make_data_orb_elt(n=n, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed)
    
    # Wrap these into a Dataset object
    ds = tf.data.Dataset.from_tensor_slices((elts, cart))

    # Set shuffle buffer size
    buffer_size = batch_size * 256

    # Shuffle and batch data sets
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    return ds

# ********************************************************************************************************************* 
def make_dataset_cfg_to_elt(n, a_min, a_max, e_max, inc_max, seed=42, batch_size=64):
    """Dataset with mapping from configuration space to orbital elements."""
    # Build data set as dictionaries of numpy arrays
    elts, cart = make_data_orb_elt(n=n, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed)
    
    # Wrap these into a Dataset object
    ds = tf.data.Dataset.from_tensor_slices((cart, elts))

    # Set shuffle buffer size
    buffer_size = batch_size * 256

    # Shuffle and batch data sets
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    return ds

# ********************************************************************************************************************* 
# Custom layers for converting between configurations (position and velocity) and orbital elements.
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class ArcCos2(keras.layers.Layer):
    """
    Variant of arc cosine taking three inputs: x, r, and y
    Returns an angle theta such that r * cos(theta) = x and r * sin(theta) matches the sign of y
    Follows function acos2 in rebound tools.c
    """
    
    # @tf.function
    def call(self, inputs):
        # Unpack inputs
        x, r, y = inputs
        # Return the arc cosine with the appropriate sign
        cosine = tf.clip_by_value(x / r, -1.0, 1.0)
        return tf.acos(cosine) * tf.math.sign(y)

    def get_config(self):
        return dict()    
    
# ********************************************************************************************************************* 
class OrbitalElementToPosition(keras.layers.Layer):
    """
    Convert 6 orbital elements to a position.  
    There is a similar layer ElementToPosition in asteroid_model.py.
    It's separate for historical reasons to avoid breaking dependencies.
    This version outputs 3 separate components; that version outputs a vector with 3 in its last axis.
    """
    def call(self, inputs):
        """Compute position q from orbital elements (a, e, inc, Omega, omega, f)"""
        # Unpack inputs
        # a: semimajor axis
        # e: eccentricity
        # inc: inclination
        # Omega: longitude of ascending node
        # omega: argument of pericenter
        # f: true anomaly
        # mu: gravitational field strength mu = G * (m0 + m1)
        a, e, inc, Omega, omega, f, mu = inputs

        # See rebound library tools.c, reb_tools_orbit_to_particle_err
        
        # Distance from center
        one_minus_e2 = 1.0 - tf.square(e)
        one_plus_e_cos_f = 1.0 + e * tf.cos(f)
        r = a * one_minus_e2 / one_plus_e_cos_f
        
        # sine and cosine of the angles inc, Omega, omega, and f
        ci = keras.layers.Activation(activation=tf.cos, name='cos_inc')(inc)
        si = keras.layers.Activation(activation=tf.sin, name='sin_inc')(inc)
        cO = keras.layers.Activation(activation=tf.cos, name='cos_Omega')(Omega)
        sO = keras.layers.Activation(activation=tf.sin, name='sin_Omega')(Omega)
        co = keras.layers.Activation(activation=tf.cos, name='cos_omega')(omega)
        so = keras.layers.Activation(activation=tf.sin, name='sin_omega')(omega)
        cf = keras.layers.Activation(activation=tf.cos, name='cos_f')(f)
        sf = keras.layers.Activation(activation=tf.sin, name='sin_f')(f)

        # Position
        # qx = r*(cO*(co*cf-so*sf) - sO*(so*cf+co*sf)*ci)
        # qy = r*(sO*(co*cf-so*sf) + cO*(so*cf+co*sf)*ci)
        # qz = r*(so*cf+co*sf)*si
        # the term cos_omega*cos_f - sin_omega*sin_f appears 2 times
        # the term sin_omega*cos_f + cos_omega*sin_f appears 3 times
        cocf_sosf = co*cf-so*sf
        socf_cosf = so*cf+co*sf
        qx = r*(cO*cocf_sosf - sO*socf_cosf*ci)
        qy = r*(sO*cocf_sosf + cO*socf_cosf*ci)
        qz = r*socf_cosf*si
        
        return (qx, qy, qz,)

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class OrbitalElementToConfig(keras.layers.Layer):
    """
    Convert 6 orbital elements to a configuration (position, velocity); acceleration is an optional output.  
    """
    def __init__(self, include_accel:bool = False, **kwargs):
        super(OrbitalElementToConfig, self).__init__(**kwargs)
        self.include_accel = include_accel

    def call(self, inputs):
        """Compute configuration (q, v) from orbital elements (a, e, inc, Omega, omega, f)"""
        # Unpack inputs
        # a: semimajor axis
        # e: eccentricity
        # inc: inclination
        # Omega: longitude of ascending node
        # omega: argument of pericenter
        # f: true anomaly
        # mu: gravitational field strength mu = G * (m0 + m1)
        a, e, inc, Omega, omega, f, mu = inputs

        # See rebound library tools.c, reb_tools_orbit_to_particle_err
        
        # Distance from center
        one_minus_e2 = tf.constant(1.0) - tf.square(e)
        one_plus_e_cos_f = tf.constant(1.0) + e * tf.cos(f)
        r = a * one_minus_e2 / one_plus_e_cos_f
        
        # Current speed
        v0 = tf.sqrt(mu / a / one_minus_e2)
        
        # sine and cosine of the angles inc, Omega, omega, and f
        ci = keras.layers.Activation(activation=tf.cos, name='cos_inc')(inc)
        si = keras.layers.Activation(activation=tf.sin, name='sin_inc')(inc)
        cO = keras.layers.Activation(activation=tf.cos, name='cos_Omega')(Omega)
        sO = keras.layers.Activation(activation=tf.sin, name='sin_Omega')(Omega)
        co = keras.layers.Activation(activation=tf.cos, name='cos_omega')(omega)
        so = keras.layers.Activation(activation=tf.sin, name='sin_omega')(omega)
        cf = keras.layers.Activation(activation=tf.cos, name='cos_f')(f)
        sf = keras.layers.Activation(activation=tf.sin, name='sin_f')(f)

        # Position
        # qx = r*(cO*(co*cf-so*sf) - sO*(so*cf+co*sf)*ci)
        # qy = r*(sO*(co*cf-so*sf) + cO*(so*cf+co*sf)*ci)
        # qz = r*(so*cf+co*sf)*si
        # the term cos_omega*cos_f - sin_omega*sin_f appears 2 times
        # the term sin_omega*cos_f + cos_omega*sin_f appears 3 times
        cocf_sosf = co*cf-so*sf
        socf_cosf = so*cf+co*sf
        qx = r*(cO*cocf_sosf - sO*socf_cosf*ci)
        qy = r*(sO*cocf_sosf + cO*socf_cosf*ci)
        qz = r*socf_cosf*si
        
        # Velocity
        # vx = v0*((e+cf)*(-ci*co*sO - cO*so) - sf*(co*cO - ci*so*sO))
        # vy = v0*((e+cf)*(ci*co*cO - sO*so)  - sf*(co*sO + ci*so*cO))
        # vz = v0*((e+cf)*co*si - sf*si*so)
        # The term e+cf appears three times
        epcf = e + cf
        # vx = v0*(epcf*(-ci*co*sO - cO*so) - sf*(co*cO - ci*so*sO))
        # vy = v0*(epcf*(ci*co*cO - sO*so)  - sf*(co*sO + ci*so*cO))
        # vz = v0*(epcf*co*si - sf*si*so)
        # The term cocO appears twice
        cocO = co * cO
        # The term cosO appears twice
        cosO = co * sO
        # The term so*sO appears twice
        sosO = so * sO
        # The terms socO appears twice
        socO = so*cO

        # Simplified expression for velocity with substitutions
        vx = v0*(epcf*(-ci*cosO - socO) - sf*(cocO - ci*sosO))
        vy = v0*(epcf*(ci*cocO - sosO)  - sf*(cosO + ci*socO))
        vz = v0*(epcf*co*si - sf*si*so)
        
        # Acceleration - Compute this only if it was requested
        # Magnitude is mu / r^2
        # Components are ax = -mu / r^3 * qx, etc.
        if self.include_accel:
            mu_over_r3 = -mu / (r*r*r)
            ax = mu_over_r3 * qx
            ay = mu_over_r3 * qy
            az = mu_over_r3 * qz            
            return qx, qy, qz, vx, vy, vz, ax, ay, az
        else:
            return qx, qy, qz, vx, vy, vz

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class ConfigToOrbitalElement(keras.layers.Layer):
    def call(self, inputs):
        """Compute orbital elements (a, e, inc, Omega, omega, f) from configuration (qx, qy, qz, vx, vy, vz, mu)"""
        # Unpack inputs
        qx, qy, qz, vx, vy, vz, mu = inputs
        
        # Promote inputs to double precision to minimize roundoff problems
        qx = tf.dtypes.cast(qx, dtype=tf.float64, name='qx')
        qy = tf.dtypes.cast(qy, dtype=tf.float64, name='qy')
        qz = tf.dtypes.cast(qz, dtype=tf.float64, name='qz')
        vx = tf.dtypes.cast(vx, dtype=tf.float64, name='vx')
        vy = tf.dtypes.cast(vy, dtype=tf.float64, name='vx')
        vz = tf.dtypes.cast(vz, dtype=tf.float64, name='vx')
        mu = tf.dtypes.cast(mu, dtype=tf.float64, name='mu')

        # See rebound library tools.c, reb_tools_particle_to_orbit_err
        
        # The distance from the primary
        # r = tf.sqrt(tf.square(qx) + tf.square(qy) + tf.square(qz))
        r = tf.sqrt(tf.math.add_n(
                [tf.square(qx) + tf.square(qy) + tf.square(qz)]), 
                name='r')
        
        # The speed and its square
        # v2 = tf.square(vx) + tf.square(vy) + tf.square(vz)
        v2 = tf.math.add_n(
                [tf.square(vx) + tf.square(vy) + tf.square(vz)], 
                name='v2')
        # v = tf.sqrt(v2)
        
        # The speed squared of a circular orbit
        v2_circ = mu / r
        
        # The semi-major axis
        two = tf.constant(2.0, dtype=tf.float64)
        a = -mu / (v2 - two * v2_circ)
        
        # The specific angular momentum vector and its magnitude
        # hx = qy*vz - qz*vy
        # hy = qz*vx - qx*vz
        # hz = qx*vy - qy*vx
        # h = tf.sqrt(tf.square(hx) + tf.square(hy) + tf.square(hz))
        hx = tf.subtract(qy*vz, qz*vy, name='hx')
        hy = tf.subtract(qz*vx, qx*vz, name='hy')
        hz = tf.subtract(qx*vy, qy*vx, name='hz')
        h = tf.sqrt(tf.math.add_n(
                [tf.square(hx) + tf.square(hy) + tf.square(hz)]), 
                name='h')
        
        # The excess squared speed vs. a circular orbit
        # v2_diff = v2 - v2_circ
        v2_diff = tf.subtract(v2, v2_circ, name='v2_diff')
        
        # The dot product of v and r; same as r times the radial speed vr
        # rvr = (qx * vx + qy*vy + qz*vz)
        rvr = tf.add_n([qx*vx, qy*vy, qz*vz], name='rvr')
        # The radial speed
        # vr = rvr / r
        vr = tf.divide(rvr, r, name='vr')
        # Inverse of mu
        one = tf.constant(1.0, dtype=tf.float64)
        mu_inv = one / mu
        
        # Eccentricity vector
        ex = mu_inv * (v2_diff * qx - rvr * vx)
        ey = mu_inv * (v2_diff * qy - rvr * vy)
        ez = mu_inv * (v2_diff * qz - rvr * vz)
        # The eccentricity is the magnitude of this vector
        # e = tf.sqrt(tf.square(ex) + tf.square(ey) + tf.square(ez))
        e = tf.sqrt(tf.math.add_n(
                [tf.square(ex) + tf.square(ey) + tf.square(ez)]),
                name='e')
        
        # The mean motion
        N = tf.sqrt(tf.abs(mu / (a*a*a)), name='N')
        
        # The inclination; zero when h points along z axis, i.e. hz = h
        # inc = tf.acos(hz / h, name='inc')
        inc = ArcCos2(name='inc_fp64')((hz, h, one))

        # Vector pointing along the ascending node = zhat cross h
        nx = -hy
        ny = hx
        n = tf.sqrt(tf.square(nx) + tf.square(ny), name='n')
        
        # Longitude of ascending node
        # Omega = tf.acos(nx / n) * tf.math.sign(ny)
        Omega = ArcCos2(name='Omega_fp64')((nx, n, ny))
        
        # Compute the eccentric anomaly for elliptical orbits (e < 1)
        ea = ArcCos2(name='eccentric_anomaly')((one - r / a, e, vr))
        
        # Compute the mean anomaly from the eccentric anomaly using Kepler's equation
        M = ea - e * tf.sin(ea)
        
        # Sum of omega + f is always defined in the orbital plane when i != 0
        omega_f = ArcCos2(name='omega_plus_f')((nx*qx + ny*qy, n*r, qz))

        # The argument of pericenter
        omega = ArcCos2(name='omega_fp64')((nx*ex + ny*ey, n*e, ez))
                
        # The true anomaly; may be larger than pi
        f = omega_f - omega
        
        # Shift f to the interval [-pi, +pi]
        pi = tf.constant(np.pi, dtype=tf.float64)
        two_pi = tf.constant(2.0 * np.pi, dtype=tf.float64)
        f = tf.math.floormod(f+pi, two_pi) - pi
        
        # Convert the outputs to single precision
        a = tf.dtypes.cast(a, dtype=tf.float32, name='a')
        e = tf.dtypes.cast(e, dtype=tf.float32, name='e')
        inc = tf.dtypes.cast(inc, dtype=tf.float32, name='inc')
        Omega = tf.dtypes.cast(Omega, dtype=tf.float32, name='Omega')
        omega = tf.dtypes.cast(omega, dtype=tf.float32, name='omega')
        f = tf.dtypes.cast(f, dtype=tf.float32, name='f')
        M = tf.dtypes.cast(M, dtype=tf.float32, name='M')
        N = tf.dtypes.cast(N, dtype=tf.float32, name='N')
        
        return a, e, inc, Omega, omega, f, M, N

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class MeanToEccentricAnomalyIteration(keras.layers.Layer):
    """
    Perform a single iteration in solving Kepler's Equation for E given M and e
    Inputs are (M, E, f, e)
    """
    
    def __init__(self, shape, **kwargs):
        super(MeanToEccentricAnomalyIteration, self).__init__(**kwargs)
        self.shape = shape

    # @tf.function
    def call(self, M, e, E, F):
        """Inputs: (M, e, E, F)"""
        # Unpack inputs
        # M, e, E, F = inputs
        
        # One step of Newton's Method
        # E = E - F / (1.0 - e * tf.cos(E))
        cos_E = keras.layers.Activation(activation=tf.cos, name='cos_E')(E)
        # e_cos_E = keras.layers.multiply([e, cos_E], name='e_cos_E')
        e_cos_E = tf.multiply(e, cos_E, name='e_cos_E')
        one = tf.broadcast_to(1.0, self.shape)
        # one_minus_e_cos_E = keras.layers.subtract([one, e_cos_E], name='one_minus_e_cos_E')
        one_minus_e_cos_E = tf.subtract(one, e_cos_E, name='one_minus_e_cos_E')
        delta_E = tf.divide(F, one_minus_e_cos_E, name='delta_E')
        # E = keras.layers.subtract([E, delta_E], name='E')
        E = tf.subtract(E, delta_E, name='E')

        # The new error term
        # F = E - e * tf.sin(E) - M
        sin_E = keras.layers.Activation(activation=tf.sin, name='sin_E')(E)
        # e_sin_E = keras.layers.multiply([e, sin_E], name='e_sin_E')
        e_sin_E = tf.multiply(e, sin_E, name='e_sin_E')
        # e_sin_E_plus_M = keras.layers.add([e_sin_E, M], name='e_sin_E_plus_M')
        e_sin_E_plus_M = tf.add(e_sin_E, M, name='e_sin_E_plus_M')
        # F = keras.layers.subtract([E, e_sin_E_plus_M], name='F')
        F = tf.subtract(E, e_sin_E_plus_M, name='F')

        return E, F
        
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class MeanToEccentricAnomaly(keras.layers.Layer):
    """
    Convert the mean anomaly M to the eccentric anomaly E given the eccentricity e.
    Use iterative method with 10 steps of Newton-Raphson.
    """
    
    # @tf.function
    def call(self, inputs):
        # Unpack inputs
        M, e = inputs
        
        # Initialize E with M
        E = M
        
        # Initial error; from Kepler's equation M = E - e sin(E)
        # F = E - e * tf.sin(E) - M
        sin_E = keras.layers.Activation(activation=tf.sin, name='sin_E')(E)
        # e_sin_E = keras.layers.multiply([e, sin_E], name='e_sin_E')
        e_sin_E = tf.multiply(e, sin_E, name='e_sin_E')
        # e_sin_E_plus_M = keras.layers.add([e_sin_E, M], name='e_sin_E_plus_M')
        e_sin_E_plus_M = tf.add(e_sin_E, M, name='e_sin_E_plus_M')
        # F = keras.layers.subtract([E, e_sin_E_plus_M], name='F')
        F = tf.subtract(E, e_sin_E_plus_M, name='F')

        # Iterate to improve E; trial and error shows 10 iterations enough for single precision convergence
        # Put the iterations inline so AutoGraph can convert it to a graph
        shape = M.shape
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_1')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_2')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_3')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_4')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_5')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_6')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_7')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_8')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_9')(M, e, E, F)
        E, F = MeanToEccentricAnomalyIteration(shape=shape, name='M2E_it_10')(M, e, E, F)

        return E


    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class MeanToTrueAnomaly(keras.layers.Layer):
    """
    Convert the mean anomaly M to the true anomaly f given the eccentricity e.
    """
    
    # @tf.function
    def call(self, inputs):
        # Unpack inputs
        M, e = inputs
        
        # Compute the eccentric anomaly E
        E = MeanToEccentricAnomaly(name='ecc_anomaly')((M, e))
        
        # Compute the true anomaly from E
        # return 2.0*tf.math.atan(tf.sqrt((1.0+e)/(1.0-e))*tf.math.tan(0.5*E))
        one = tf.broadcast_to(1.0, M.shape)
        # one_plus_e = keras.layers.add([one, e], name='one_plus_e')
        one_plus_e = tf.add(one, e, name='one_plus_e')
        # one_minus_e = keras.layers.subtract([one, e], name='one_minus_e')
        one_minus_e = tf.subtract(one, e, name='one_minus_e')
        ecc_ratio = tf.divide(one_plus_e, one_minus_e, name='ecc_ratio')
        sqrt_ecc_ratio = keras.layers.Activation(activation=tf.sqrt, name='sqrt_ecc_ratio') (ecc_ratio)
        half = tf.broadcast_to(0.5, M.shape)
        # half_E = keras.layers.multiply([half, E], name='half_E')
        half_E = tf.multiply(half, E, name='half_E')
        tan_half_E = keras.layers.Activation(activation=tf.math.tan, name='tan_half_E')(half_E)
        # tan_half_theta = keras.layers.multiply([sqrt_ecc_ratio, tan_half_E], name='tan_half_theta')
        tan_half_theta = tf.multiply(sqrt_ecc_ratio, tan_half_E, name='tan_half_theta')
        half_theta = keras.layers.Activation(activation=tf.math.atan, name='half_theta')(tan_half_theta)
        two = tf.broadcast_to(2.0, M.shape)
        # theta = keras.layers.multiply([two, half_theta], name='theta')
        theta = tf.multiply(two, half_theta, name='theta')
        return theta

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class EccentricToMeanAnomaly(keras.layers.Layer):
    """
    Convert the eccentric anomaly E to the true anomaly M given the eccentricity e.
    """
    
    # @tf.function
    def call(self, inputs):
        # Unpack inputs
        E, e = inputs
        
        # Compute the mean anomaly from E using Kepler's Equation
        # M = E - e sin(E)
        sin_E = keras.layers.Activation(activation=tf.sin, name='sin_E')(E)
        # e_sin_E = keras.layers.multiply([e, sin_E], name='e_sin_E')
        e_sin_E = tf.multiply(e, sin_E, name='e_sin_E')
        # M = keras.layers.subtract([E, e_sin_E], name='M')
        M = tf.subtract(E, e_sin_E, name='M')
        return M
        
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class TrueToEccentricAnomaly(keras.layers.Layer):
    """
    Convert the true anomaly f to the eccentric anomaly E given the eccentricity e.
    """
    
    # @tf.function
    def call(self, inputs):
        # Unpack inputs
        f, e = inputs
        # Compute the eccentric anomaly E
        # https://en.wikipedia.org/wiki/Eccentric_anomaly
        cos_f = keras.layers.Activation(activation=tf.cos, name='cos_f')(f)
        # e_cos_f = keras.layers.multiply([e, cos_f], name = 'e_cos_f')
        e_cos_f = tf.multiply(e, cos_f, name = 'e_cos_f')
        one = tf.broadcast_to(1.0, f.shape)
        # denom = keras.layers.add([one, e_cos_f], name = 'denom')
        denom = tf.add(one, e_cos_f, name = 'denom')

        # cos_E_num = keras.layers.add([e, cos_f], name = 'cos_E_num')
        cos_E_num = tf.add(e, cos_f, name = 'cos_E_num')
        cos_E = tf.divide(cos_E_num, denom, name = 'cos_E')
        sin_f = keras.layers.Activation(activation=tf.sin, name='sin_f')(f)
        # e2 = keras.layers.multiply([e,e], name='e2')
        e2 = tf.square(e, name='e2')
        # one_minus_e2 = keras.layers.subtract([one, e2], name='one_minus_e2')
        one_minus_e2 = tf.subtract(one, e2, name='one_minus_e2')
        sqrt_one_minus_e2 = keras.layers.Activation(activation=tf.sqrt, name = 'sqrt_one_minus_e2')(one_minus_e2)
        # sin_E_num = keras.layers.multiply([sqrt_one_minus_e2, sin_f], name='sin_E_num')
        sin_E_num = tf.multiply(sqrt_one_minus_e2, sin_f, name='sin_E_num')
        sin_E = tf.divide(sin_E_num, denom, name='sin_E')
        E = tf.atan2(y=sin_E, x=cos_E, name='E')
        return E
        
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class TrueToMeanAnomaly(keras.layers.Layer):
    """
    Convert the true anomaly f to the mean anomaly M given the eccentricity e.
    """
    
    # @tf.function
    def call(self, inputs):
        # Unpack inputs
        f, e = inputs
        # Compute the eccentric anomaly E
        E = TrueToEccentricAnomaly(name='TrueToEccentricAnomaly')([f, e])
        # Compute mean anomaly M from E using Kepler's Equation
        # https://en.wikipedia.org/wiki/Mean_anomaly
        # M = E - e * tf.sin(E)
        # sin_E = tf.sin(E, name='sin_E')
        sin_E = keras.layers.Activation(activation=tf.sin, name='sin_E_')(E)
        # e_sin_E = keras.layers.multiply([e, sin_E], name='e_sin_E')
        e_sin_E = tf.multiply(e, sin_E, name='e_sin_E')
        # M = keras.layers.subtract([E, e_sin_E], name='M')
        M = tf.subtract(E, e_sin_E, name='M')
        return M

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
# Models wrapping the layers performing the conversions
# Makes it more convenient to test them using e.g. model.evaluate()
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_elt_to_pos(batch_size: int=64, name=None):
    """Model that transforms from orbital elements to cartesian coordinates (position only)"""
    # The shape shared by all the inputs
    input_shape = (1,)

    # Create input layers    
    a = keras.Input(shape=input_shape, batch_size=batch_size, name='a')
    e = keras.Input(shape=input_shape, batch_size=batch_size, name='e')
    inc = keras.Input(shape=input_shape, batch_size=batch_size, name='inc')
    Omega = keras.Input(shape=input_shape, batch_size=batch_size, name='Omega')
    omega = keras.Input(shape=input_shape, batch_size=batch_size, name='omega')
    f = keras.Input(shape=input_shape, batch_size=batch_size, name='f')
    mu = keras.Input(shape=input_shape, batch_size=batch_size, name='mu')
    
    # Wrap these up into one tuple of inputs
    inputs = (a, e, inc, Omega, omega, f, mu,)
    
    # Calculations are in one layer that does all the work...
    qx, qy, qz = OrbitalElementToPosition(name='orb_elt_to_pos')(inputs)
    
    # Assemble the position and velocity vectors
    q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=-1, name='q')

    # Wrap up the outputs
    # outputs = q

    # Create a model from inputs to outputs
    name = name or 'orbital_element_to_cartesian'
    model = keras.Model(inputs=inputs, outputs=q, name=name)
    return model

# ********************************************************************************************************************* 
def make_model_elt_to_cfg(include_accel: bool = False, batch_size: int=64, name=None):
    """Model that transforms from orbital elements to cartesian coordinates (position and velocity)"""
    # The shape shared by all the inputs
    input_shape = (1,)

    # Create input layers    
    a = keras.Input(shape=input_shape, batch_size=batch_size, name='a')
    e = keras.Input(shape=input_shape, batch_size=batch_size, name='e')
    inc = keras.Input(shape=input_shape, batch_size=batch_size, name='inc')
    Omega = keras.Input(shape=input_shape, batch_size=batch_size, name='Omega')
    omega = keras.Input(shape=input_shape, batch_size=batch_size, name='omega')
    f = keras.Input(shape=input_shape, batch_size=batch_size, name='f')
    mu = keras.Input(shape=input_shape, batch_size=batch_size, name='mu')
    
    # Wrap these up into one tuple of inputs
    inputs = (a, e, inc, Omega, omega, f, mu,)
    
    # Calculations are in one layer that does all the work...
    outputs = OrbitalElementToConfig(include_accel=include_accel, name='orbital_element_to_config')(inputs)
    
    if include_accel:
        qx, qy, qz, vx, vy, vz, ax, ay, az = outputs
    else:
        qx, qy, qz, vx, vy, vz = outputs
    
    # Assemble the position and velocity vectors
    q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=-1, name='q')
    v = keras.layers.concatenate(inputs=[vx, vy, vz], axis=-1, name='v')
    # Assemble the acceleration vector if it was requested
    if include_accel:
        acc = keras.layers.concatenate(inputs=[ax, ay, az], axis=-1, name='acc')

    # Wrap up the outputs
    outputs = (q, v, acc) if include_accel else (q, v)

    # Create a model from inputs to outputs
    name = name or 'orbital_element_to_cartesian'
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

# ********************************************************************************************************************* 
def make_model_cfg_to_elt(batch_size: int=64, name=None):
    """Model that transforms from orbital elements to cartesian coordinates"""
    # Create input layers    
    q = keras.Input(shape=(3,), batch_size=batch_size, name='q')
    v = keras.Input(shape=(3,), batch_size=batch_size, name='v')
    mu = keras.Input(shape=(1,), batch_size=batch_size, name='mu')
    
    # Wrap these up into one tuple of inputs for the model
    inputs_model = (q, v, mu,)
    
    # Unpack coordinates from inputs
    qx = keras.layers.Reshape(target_shape=(1,), name='qx')(q[:,0])
    qy = keras.layers.Reshape(target_shape=(1,), name='qy')(q[:,1])
    qz = keras.layers.Reshape(target_shape=(1,), name='qz')(q[:,2])
    vx = keras.layers.Reshape(target_shape=(1,), name='vx')(v[:,0])
    vy = keras.layers.Reshape(target_shape=(1,), name='vy')(v[:,1])
    vz = keras.layers.Reshape(target_shape=(1,), name='vz')(v[:,2])

    # Tuple of inputs for the layer
    inputs_layer = (qx, qy, qz, vx, vy, vz, mu,)

    # Calculations are in one layer that does all the work...
    a, e, inc, Omega, omega, f, M, N = ConfigToOrbitalElement(name='config_to_orbital_element')(inputs_layer)

    # Name the outputs of the layer
    a = Identity(name='a')(a)
    e = Identity(name='e')(e)
    inc = Identity(name='inc')(inc)
    Omega = Identity(name='Omega')(Omega)
    omega = Identity(name='omega')(omega)
    f = Identity(name='f')(f)
    # "Bonus outputs" - mean anomaly and mean motion
    M = Identity(name='M')(M)
    N = Identity(name='N')(N)

    # Wrap up the outputs
    outputs = (a, e, inc, Omega, omega, f, M, N)

    # Create a model from inputs to outputs
    name = name or 'config_to_orbital_element'
    model = keras.Model(inputs=inputs_model, outputs=outputs, name=name)
    return model

