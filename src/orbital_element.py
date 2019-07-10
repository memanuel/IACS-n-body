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
    
    # The graviational constant mu as a scalar; assume the small particles have mass 0
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
    q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=1, name='q')
    v = keras.layers.concatenate(inputs=[vx, vy, vz], axis=1, name='v')

    # Wrap up the outputs
    outputs = (q, v)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='orbital_element_to_cartesian')
    return model

# ********************************************************************************************************************* 
def make_model_cfg_to_elt():
    """Model that transforms from orbital elements to cartesian coordinates"""
    # The shape shared by all the inputs
    input_shape = (1,)

    # Create input layers    
    qx = keras.Input(shape=input_shape, name='qx')
    qy = keras.Input(shape=input_shape, name='qy')
    qz = keras.Input(shape=input_shape, name='qz')
    vx = keras.Input(shape=input_shape, name='vx')
    vy = keras.Input(shape=input_shape, name='vy')
    vz = keras.Input(shape=input_shape, name='vz')
    mu = keras.Input(shape=input_shape, name='mu')
    
    # Wrap these up into one tuple of inputs
    inputs = (qx, qy, qz, vx, vy, vz, mu,)

    # Calculations are in one layer that does all the work...
    a, e, inc, Omega, omega, f = ConfigToOrbitalElement(name='config_to_orbital_element')(inputs)

    # Wrap up the outputs
    outputs = (a, e, inc, Omega, omega, f)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='cartesian_to_orbital_element')
    return model

