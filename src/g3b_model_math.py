"""
Harvard IACS Masters Thesis
General Three Body Problem
Models for general three body problem using math (closed form)
NOT EXACT - Approximation using Jacobi coordinates where
- body 1 orbits around the primary (body 0)
- body 2 orbits around the center of mass of body 0 and body 1

Michael S. Emanuel
Tue Aug 06 14:54:00 2019
"""

# Library imports
import tensorflow as tf

# Aliases
keras = tf.keras

# Local imports
from orbital_element import make_model_cfg_to_elt, make_model_elt_to_cfg
from orbital_element import MeanToTrueAnomaly, G_
from g3b import make_physics_model_g3b

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_g3b_math(traj_size = 731):
    """
    Compute orbit positions for the general two body problem from 
    the initial orbital elements with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    num_particles = 3
    space_dims = 3
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(num_particles, space_dims,), name='q0')
    v0 = keras.Input(shape=(num_particles, space_dims,), name='v0')
    m = keras.Input(shape=(num_particles,), name='m')

    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, m)

    # The gravitational constant; numerical value close to 4 pi^2; see rebound documentation for exact value        
    G = tf.constant(G_)

    # Unpack masses
    m0 = m[:, 0]
    m1 = m[:, 1]
    m2 = m[:, 2]
    # Calculate cumulative masses
    M0 = m0
    M1 = M0 + m1
    M2 = M1 + m2

    # Gravitational field strength; shape (batch_size,)
    r1_mu = G * M1
    r2_mu = G * M2
    
    # Reshape the gravitational field strength from (batch_size,) to (batch_size, 1,)
    r1_mu = keras.layers.Reshape((1,))(r1_mu)
    r2_mu = keras.layers.Reshape((1,))(r2_mu)

    # Extract the relative position and relative velocity in Jacobi coordinates of body 1
    R0_q0 = q0[:, 0, :]
    R0_v0 = v0[:, 0, :]
    r1_q0 = q0[:, 1, :] - R0_q0
    r1_v0 = v0[:, 1, :] - R0_v0

    # Extract the relative position and relative velocity in Jacobi coordinates of body 2
    R1_q0 = (m0 * q0[:, 0, :] + m1 * q0[:, 1, :]) / M1
    R1_v0 = (m0 * v0[:, 0, :] + m1 * v0[:, 1, :]) / M1
    r2_q0 = q0[:, 2, :] - R1_q0
    r2_v0 = v0[:, 2, :] - R1_v0

    # Tuple of inputs for the model converting from configuration to orbital elements
    r1_cfg = (r1_q0, r1_v0, r1_mu)
    r2_cfg = (r2_q0, r2_v0, r2_mu)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e = make_model_cfg_to_elt()

    # Extract the orbital elements of the initial conditions
    a1_0, e1_0, inc1_0, Omega1_0, omega1_0, f1_0, M1_0, N1_0 = model_c2e(r1_cfg)
    a2_0, e2_0, inc2_0, Omega2_0, omega2_0, f2_0, M2_0, N2_0 = model_c2e(r2_cfg)

    # Alias ri_mu for naming consistency
    mu1_0 = r1_mu
    mu2_0 = r2_mu

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)

    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a1 = keras.layers.RepeatVector(n=traj_size, name='a')(a1_0)
    e1 = keras.layers.RepeatVector(n=traj_size, name='e')(e1_0)
    inc1 = keras.layers.RepeatVector(n=traj_size, name='inc')(inc1_0)
    Omega1 = keras.layers.RepeatVector(n=traj_size, name='Omega')(Omega1_0)
    omega1 = keras.layers.RepeatVector(n=traj_size, name='omega')(omega1_0)
    mu1 = keras.layers.RepeatVector(n=traj_size, name='mu')(mu1_0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M1_0_vec = keras.layers.RepeatVector(n=traj_size, name='M1_0_vec')(M1_0)
    N1_0_vec = keras.layers.RepeatVector(n=traj_size, name='N1_0_vec')(N1_0)
    # Compute the mean anomaly M(t) as a function of time
    N1_t = keras.layers.multiply(inputs=[N1_0_vec, t_vec])
    M1 = keras.layers.add(inputs=[M1_0_vec, N1_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f1 = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M1, e1])

    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    r1_elt = (a1, e1, inc1, Omega1, omega1, f1, mu1,)

    # Model mapping orbital elements to cartesian coordinates
    model_e2c = make_model_elt_to_cfg()

    # Convert from orbital elements to cartesian coordinates
    # This is the position and velocity of the Jacobi coordinate r2 = q2 - q1
    r1_q, r1_v = model_e2c(r1_elt)

#    # Reshape coefficients for q1 and q2 from r2
#    coeff_shape = (1,1,)
#    coeff_shape_layer = keras.layers.Reshape(target_shape=coeff_shape, name='coeff_shape')
#    coeff1 = coeff_shape_layer(-m2 / M2)
#    coeff2 = coeff_shape_layer( m1 / M2)

    # Compute the position and velocity of the individual particles from the Jacobi coordinates
    q0 = 0
    q1 = 0
    q2 = 0
    v0 = 0
    v1 = 0
    v2 = 0

    # Assemble the position and velocity
    particle_traj_shape = (-1, 1, 3)
    particle_traj_shape_layer = keras.layers.Reshape(target_shape=particle_traj_shape, name='particle_traj_shape')
    q1 = particle_traj_shape_layer(q1)
    q2 = particle_traj_shape_layer(q2)
    v1 = particle_traj_shape_layer(v1)
    v2 = particle_traj_shape_layer(v2)
    q = keras.layers.concatenate(inputs=[q1, q2], axis=-2)
    v = keras.layers.concatenate(inputs=[v1, v2], axis=-2)

    # Wrap up the outputs
    outputs = (q, v)

    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_g2b_position_math')
    return model

# ********************************************************************************************************************* 
def make_model_g3b_math(traj_size: int = 731):
    """Create a math model for the restricted three body problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_g3b_math(traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_g3b(position_model=position_model, traj_size=traj_size)

