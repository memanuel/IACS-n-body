"""
Harvard IACS Masters Thesis
General Two Body Problem
Models for general two body problem using math (closed form)

Michael S. Emanuel
Tue Aug 06 14:54:00 2019
"""

# Library imports
import tensorflow as tf
# import numpy as np

# Aliases
keras = tf.keras

# Local imports
from orbital_element import make_model_cfg_to_elt, make_model_elt_to_cfg
from orbital_element import G_, MeanToTrueAnomaly
from g2b import make_physics_model_g2b

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_g2b_math(traj_size = 731):
    """
    Compute orbit positions for the general two body problem from 
    the initial orbital elements with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    num_particles = 2
    space_dims = 3
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(num_particles, space_dims,), name='q0')
    v0 = keras.Input(shape=(num_particles, space_dims,), name='v0')
    m = keras.Input(shape=(num_particles,), name='m')

    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, m)

    # The gravitational constant; numerical value close to 4 pi^2; see rebound documentation for exact value        
    G = tf.constant(G_)

    # Gravitational field strength; shape (batch_size,)
    m1 = m[:, 0]
    m2 = m[:, 1]
    M_tot = m1 + m2
    mu = G * M_tot

    # Reshape the gravitational field strength from (batch_size,) to (batch_size, 1,)
    mu0 = keras.layers.Reshape((1,))(mu)

    # Extract the relative position and relative velocity in Jacobi coordinates
    q0_jac_2 = q0[:, 1, :] - q0[:, 0, :]
    v0_jac_2 = v0[:, 1, :] - v0[:, 0, :]

    # Tuple of inputs for the model converting from configuration to orbital elements
    inputs_c2e = (q0_jac_2, v0_jac_2, mu0)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e = make_model_cfg_to_elt()

    # Extract the orbital elements of the initial conditions
    a0, e0, inc0, Omega0, omega0, f0, M0, N0 = model_c2e(inputs_c2e)

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)

    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a = keras.layers.RepeatVector(n=traj_size, name='a')(a0)
    e = keras.layers.RepeatVector(n=traj_size, name='e')(e0)
    inc = keras.layers.RepeatVector(n=traj_size, name='inc')(inc0)
    Omega = keras.layers.RepeatVector(n=traj_size, name='Omega')(Omega0)
    omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)
    mu = keras.layers.RepeatVector(n=traj_size, name='mu_vec')(mu0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M0_vec = keras.layers.RepeatVector(n=traj_size, name='M0_vec')(M0)
    N0_vec = keras.layers.RepeatVector(n=traj_size, name='N0_vec')(N0)
    # Compute the mean anomaly M(t) as a function of time
    N_t = keras.layers.multiply(inputs=[N0_vec, t_vec])
    M = keras.layers.add(inputs=[M0_vec, N_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M, e])

    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    inputs_e2c = (a, e, inc, Omega, omega, f, mu,)

    # Model mapping orbital elements to cartesian coordinates
    model_e2c = make_model_elt_to_cfg()

    # Convert from orbital elements to cartesian coordinates
    # This is the position and velocity of the Jacobi coordinate r2 = q2 - q1
    r2_q, r2_v = model_e2c(inputs_e2c)

    # Reshape coefficients for q1 and q2 from r2
    coeff_shape = (1,1,)
    coeff_shape_layer = keras.layers.Reshape(target_shape=coeff_shape, name='coeff_shape')
    coeff1 = coeff_shape_layer(-m2 / M_tot)
    coeff2 = coeff_shape_layer( m1 / M_tot)

    # Compute the position and velocity of the individual particles from the Jacobi coordinates
    q1 = coeff1 * r2_q
    q2 = coeff2 * r2_q
    v1 = coeff1 * r2_v
    v2 = coeff2 * r2_v

    # Assemble the position and velocity
    particle_traj_shape = (-1, 1, 3)
    particle_traj_shape_layer = keras.layers.Reshape(target_shape=particle_traj_shape, name='particle_traj_shape')
    q1_vec = particle_traj_shape_layer(q1)
    q2_vec = particle_traj_shape_layer(q2)
    v1_vec = particle_traj_shape_layer(v1)
    v2_vec = particle_traj_shape_layer(v2)
    q = keras.layers.concatenate(inputs=[q1_vec, q2_vec], axis=-2)
    v = keras.layers.concatenate(inputs=[v1_vec, v2_vec], axis=-2)

    # Wrap up the outputs
    outputs = (q, v)

    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_g2b_position_math')
    return model

# ********************************************************************************************************************* 
def make_model_g2b_math(traj_size: int = 731):
    """Create a math model for the restricted two body circular problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_g2b_math(traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_g2b(position_model=position_model, traj_size=traj_size)

