"""
Harvard IACS Masters Thesis
Oribtal Elements

Michael S. Emanuel
Wed Jul 10 13:44:54 2019
"""

# Library imports
import tensorflow as tf
from tensorflow import keras
import rebound
import numpy as np

# Local imports
from tf_utils import Identity

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
    
    # Stack the position and velocity vectors
    q = np.stack([qx, qy, qz], axis=1)
    v = np.stack([vx, vy, vz], axis=1)
    
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
    def call(self, inputs):
        # Unpack inputs
        x, r, y = inputs
        # Return the arc cosine with the appropriate sign
        # return tf.acos(x / r) * tf.math.sign(y)
        cosine = tf.clip_by_value(x / r, -1.0, 1.0)
        return tf.acos(cosine) * tf.math.sign(y)

    def get_config(self):
        return dict()    
    
# ********************************************************************************************************************* 
class OrbitalElementToConfig(keras.layers.Layer):
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
        vx = v0*(epcf*(-ci*co*sO - cO*so) - sf*(co*cO - ci*so*sO))
        vy = v0*(epcf*(ci*co*cO - sO*so)  - sf*(co*sO + ci*so*cO))
        vz = v0*(epcf*co*si - sf*si*so)
        
        return qx, qy, qz, vx, vy, vz

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class ConfigToOrbitalElement(keras.layers.Layer):
    def call(self, inputs):
        """Compute orbital elements (a, e, inc, Omega, omega, f) from configuration (qx, qy, qz, vx, vy, vz, mu)"""
        # Unpack inputs
        qx, qy, qz, vx, vy, vz, mu = inputs

        # See rebound library tools.c, reb_tools_particle_to_orbit_err
        
        # The distance from the primary
        r = tf.sqrt(tf.square(qx) + tf.square(qy) + tf.square(qz))
        
        # The speed and its square
        v2 = tf.square(vx) + tf.square(vy) + tf.square(vz)
        # v = tf.sqrt(v2)
        
        # The speed squared of a circular orbit
        v2_circ = mu / r
        
        # The semi-major axis
        a = -mu / (v2 - tf.constant(2.0) * v2_circ)
        
        # The specific angular momentum vector and its magnitude
        hx = qy*vz - qz*vy
        hy = qz*vx - qx*vz
        hz = qx*vy - qy*vx
        h = tf.sqrt(tf.square(hx) + tf.square(hy) + tf.square(hz))
        
        # The excess squared speed vs. a circular orbit
        v2_diff = v2 - v2_circ
        
        # The dot product of v and r; same as r times the radial speed vr
        rvr = (qx * vx + qy*vy + qz*vz)
        # The radial speed
        vr = rvr / r
        # Inverse of mu
        mu_inv = tf.constant(1.0) / mu
        
        # Eccentricity vector
        ex = mu_inv * (v2_diff * qx - rvr * vx)
        ey = mu_inv * (v2_diff * qy - rvr * vy)
        ez = mu_inv * (v2_diff * qz - rvr * vz)
        # The eccentricity is the magnitude of this vector
        e = tf.sqrt(tf.square(ex) + tf.square(ey) + tf.square(ez))
        
        # The mean motion
        N = tf.sqrt(tf.abs(mu / (a*a*a)))
        
        # The inclination; zero when h points along z axis, i.e. hz = h
        inc = tf.acos(hz / h)

        # Vector pointing along the ascending node = zhat cross h
        nx = -hy
        ny = hx
        n = tf.sqrt(tf.square(nx) + tf.square(ny))
        
        # Longitude of ascending node
        # Omega = tf.acos(nx / n) * tf.math.sign(ny)
        Omega = ArcCos2(name='Omega')((nx, n, ny))
        
        # Compute the eccentric anomaly for elliptical orbits (e < 1)
        ea = ArcCos2(name='eccentric_anomaly')((tf.constant(1.0) - r / a, e, vr))
        
        # Compute the mean anomaly from the eccentric anomaly using Kepler's equation
        M = ea - e * tf.sin(ea)
        
        # Sum of omega + f is always defined in the orbital plane when i != 0
        omega_f = ArcCos2(name='omega_plus_f')((nx*qx + ny*qy, n*r, qz))

        # The argument of pericenter
        omega = ArcCos2(name='omega')((nx*ex + ny*ey, n*e, ez))
                
        # The true anomaly; may be larger than pi
        f = omega_f - omega
        
        # Shift f to the interval [-pi, +pi]
        pi = tf.constant(np.pi)
        two_pi = tf.constant(2.0 * np.pi)
        f = tf.math.floormod(f+pi, two_pi) - pi
        
        return a, e, inc, Omega, omega, f, M, N

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class MeanToEccentricAnomaly(keras.layers.Layer):
    """
    Convert the mean anomaly M to the eccentric anomly E given the eccentricity E.
    """
    def call(self, inputs):
        # Unpack inputs
        M, e = inputs
        
        # Initialize E; guess M when eccentricity is small, otherwise guess pi
        # E = tf.where(condition= e < 0.8, x=M, y=tf.constant(np.pi))
        E = M
        
        # Initial error; from Kepler's equation M = E - e sin(E)
        F = E - e * tf.sin(E) - M
        
        # Iterate to improve E; trial and error shows 10 enough for single precision convergence
        for i in range(10):
            # One step of Newton's Method
            E = E - F / (1.0 - e * tf.cos(E))
            # The new error term
            F = E - e * tf.sin(E) - M

        return E
        
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class MeanToTrueAnomaly(keras.layers.Layer):
    """
    Convert the mean anomaly M to the true anomly f given the eccentricity E.
    """
    def call(self, inputs):
        # Unpack inputs
        M, e = inputs
        
        # Compute the eccentric anomaly E
        E = MeanToEccentricAnomaly()((M, e))
        
        # Compute the true anomaly from E
        return 2.0*tf.math.atan(tf.sqrt((1.0+e)/(1.0-e))*tf.math.tan(0.5*E))
        
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
# Models wrapping the layers performing the conversions
# Makes it more convenient to test them using e.g. model.evaluate()
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_elt_to_cfg():
    """Model that transforms from orbital elements to cartesian coordinates"""
    # The shape shared by all the inputs
    input_shape = (1,)

    # Create input layers    
    a = keras.Input(shape=input_shape, name='a')
    e = keras.Input(shape=input_shape, name='e')
    inc = keras.Input(shape=input_shape, name='inc')
    Omega = keras.Input(shape=input_shape, name='Omega')
    omega = keras.Input(shape=input_shape, name='omega')
    f = keras.Input(shape=input_shape, name='f')
    mu = keras.Input(shape=input_shape, name='mu')
    
    # Wrap these up into one tuple of inputs
    inputs = (a, e, inc, Omega, omega, f, mu,)
    
    # Calculations are in one layer that does all the work...
    qx, qy, qz, vx, vy, vz = OrbitalElementToConfig(name='orbital_element_to_config')(inputs)
    
    # Assemble the position and velocity vectors
    q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=-1, name='q')
    v = keras.layers.concatenate(inputs=[vx, vy, vz], axis=-1, name='v')

    # Wrap up the outputs
    outputs = (q, v)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='orbital_element_to_cartesian')
    return model

# ********************************************************************************************************************* 
def make_model_cfg_to_elt():
    """Model that transforms from orbital elements to cartesian coordinates"""
    # Create input layers    
    q = keras.Input(shape=(3,), name='q')
    v = keras.Input(shape=(3,), name='v')
    mu = keras.Input(shape=(1,), name='mu')
    
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
    model = keras.Model(inputs=inputs_model, outputs=outputs, name='config_to_orbital_element')
    return model

