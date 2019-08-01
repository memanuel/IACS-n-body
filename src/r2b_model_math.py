"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Models for restricted two body problem using math (closed form)

Michael S. Emanuel
Tue Jul 30 14:52:00 2019
"""

# Library imports
import tensorflow as tf
# import numpy as np

# Aliases
keras = tf.keras

# Local imports
from orbital_element import make_model_cfg_to_elt
from orbital_element import OrbitalElementToConfig, MeanToTrueAnomaly
from r2b import make_physics_model_r2b
# from tf_utils import Identity
# from r2b import KineticEnergy_R2B, PotentialEnergy_R2B, AngularMomentum_R2B

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_r2b_math(traj_size = 731):
    """
    Compute orbit positions for the restricted two body problem from 
    the initial orbital elements with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    # Create input layers 
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(3,), name='q0')
    v0 = keras.Input(shape=(3,), name='v0')
    mu = keras.Input(shape=(1,), name='mu')
    
    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, mu)
    
    # The gravitational constant; give this shape (1,1) for compatibility with RepeatVector
    # The numerical value mu0 is close to 4 pi^2; see rebound documentation for exact value
    # mu0 = tf.constant([[39.476924896240234]])
    
    # Reshape the gravitational field strength from (batch_size,) to (batch_size, 1,)
    mu0 = keras.layers.Reshape((1,))(mu)

    # Tuple of inputs for the model converting from configuration to orbital elements
    inputs_c2e = (q0, v0, mu0)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e = make_model_cfg_to_elt()
    
    # Extract the orbital elements of the initial conditions
    a0, e0, inc0, Omega0, omega0, f0, M0, N0 = model_c2e(inputs_c2e)

    # Check shapes of initial orbital elements
    batch_size = t.shape[0]
    tf.debugging.assert_shapes(shapes={
        a0: (batch_size, 1),
        e0: (batch_size, 1),
        inc0: (batch_size, 1),
        Omega0: (batch_size, 1),
        omega0: (batch_size, 1),
        mu0: (batch_size, 1),
    }, message='make_position_model_r2b_math / initial orbital elements')
    
    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a = keras.layers.RepeatVector(n=traj_size, name='a')(a0)
    e = keras.layers.RepeatVector(n=traj_size, name='e')(e0)
    inc = keras.layers.RepeatVector(n=traj_size, name='inc')(inc0)
    Omega = keras.layers.RepeatVector(n=traj_size, name='Omega')(Omega0)
    omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)
    mu = keras.layers.RepeatVector(n=traj_size, name='mu_vec')(mu0)

    # Check shapes of orbital element calculations
    tf.debugging.assert_shapes(shapes={
        a: (batch_size, traj_size, 1),
        e: (batch_size, traj_size, 1),
        inc: (batch_size, traj_size, 1),
        Omega: (batch_size, traj_size, 1),
        omega: (batch_size, traj_size, 1),
        # f: (batch_size, traj_size, 1),
        mu: (batch_size, traj_size, 1),
    }, message='make_position_model_r2b_math / orbital element calcs')

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M0_vec = keras.layers.RepeatVector(n=traj_size, name='M0_vec')(M0)
    N0_vec = keras.layers.RepeatVector(n=traj_size, name='N0_vec')(N0)
    # Compute the mean anomaly M(t) as a function of time
    N_t = keras.layers.multiply(inputs=[N0_vec, t_vec])
    M = keras.layers.add(inputs=[M0_vec, N_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M, e])
    
    # Check shapes of calculations for true anomaly f
    tf.debugging.assert_shapes(shapes={
        M0_vec: (batch_size, traj_size, 1),
        N0_vec: (batch_size, traj_size, 1),
        N_t: (batch_size, traj_size, 1),
        M: (batch_size, traj_size, 1),
    }, message='make_position_model_r2b_math / orbital element calcs')
    
    
    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    inputs_e2c = (a, e, inc, Omega, omega, f, mu,)
    
    # Convert from orbital elements to cartesian
    qx, qy, qz, vx, vy, vz = OrbitalElementToConfig(name='orbital_element_to_config')(inputs_e2c)
    
    # Wrap up the outputs
    outputs = (qx, qy, qz, vx, vy, vz)
    
    # Check shapes
    tf.debugging.assert_shapes(shapes={
        qx: (batch_size, traj_size, 1),
        qy: (batch_size, traj_size, 1),
        qz: (batch_size, traj_size, 1),
        vx: (batch_size, traj_size, 1),
        vy: (batch_size, traj_size, 1),
        vz: (batch_size, traj_size, 1),
    }, message='make_position_model_r2b_math / outputs')
    
    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_r2b_position_math')
    return model

# ********************************************************************************************************************* 
def make_model_r2b_math(traj_size: int = 731):
    """Create a math model for the restricted two body circular problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_r2b_math(traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_r2b(position_model=position_model, traj_size=traj_size)

