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
from tf_utils import Identity
from orbital_element import make_model_cfg_to_elt, make_model_elt_to_cfg
from orbital_element import MeanToTrueAnomaly
from jacobi import CartesianToJacobi, JacobiToCartesian
from g3b import make_physics_model_g3b

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_g3b_math(traj_size:int =1001, batch_size:int =64, num_gpus:int = 1):
    """
    Compute orbit positions for the general two body problem from 
    the initial orbital elements with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    num_particles = 3
    space_dims = 3
    full_batch_size=batch_size*num_gpus
    t = keras.Input(shape=(traj_size,), batch_size=full_batch_size, name='t')
    q0 = keras.Input(shape=(num_particles, space_dims,), batch_size=full_batch_size, name='q0')
    v0 = keras.Input(shape=(num_particles, space_dims,), batch_size=full_batch_size, name='v0')
    m = keras.Input(shape=(num_particles,), batch_size=full_batch_size, name='m')

    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, m)

    # Compute the Jacobi coordinates of the initial conditions
    qj0, vj0, mu0 = CartesianToJacobi()([m, q0, v0])

    # Extract Jacobi coordinates of p1 and p2
    qj0_1 = qj0[:, 1, :]
    qj0_2 = qj0[:, 2, :]
    vj0_1 = vj0[:, 1, :]
    vj0_2 = vj0[:, 2, :]
    
    # Extract gravitational field strength for orbital element conversion of p1 and p2
    mu0_1 = mu0[:, 1:2]
    mu0_2 = mu0[:, 2:3]

    # Manually set the shapes to work around documented bug on slices losing shape info
    jacobi_shape = (batch_size, space_dims)
    qj0_1.set_shape(jacobi_shape)
    qj0_2.set_shape(jacobi_shape)
    vj0_1.set_shape(jacobi_shape)
    vj0_1.set_shape(jacobi_shape)
    mu_shape = (batch_size, 1)
    mu0_1.set_shape(mu_shape)
    mu0_2.set_shape(mu_shape)
    
    # Tuple of inputs for the model converting from configuration to orbital elements
    cfg_1 = (qj0_1, vj0_1, mu0_1)
    cfg_2 = (qj0_2, vj0_2, mu0_2)

    # Model mapping cartesian coordinates to orbital elements
    # model_c2e = make_model_cfg_to_elt()
    model_c2e_1 = make_model_cfg_to_elt(name='orbital_element_1')
    model_c2e_2 = make_model_cfg_to_elt(name='orbital_element_2')

    # Extract the orbital elements of the initial conditions
    a1_0, e1_0, inc1_0, Omega1_0, omega1_0, f1_0, M1_0, N1_0 = model_c2e_1(cfg_1)
    a2_0, e2_0, inc2_0, Omega2_0, omega2_0, f2_0, M2_0, N2_0 = model_c2e_2(cfg_2)

    # Alias mu0_i for naming consistency
    mu1_0 = mu0_1
    mu2_0 = mu0_2
    
    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # ******************************************************************
    # Predict orbital elements for Jacobi coordinates of body 1
    
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a1 = keras.layers.RepeatVector(n=traj_size, name='a1')(a1_0)
    e1 = keras.layers.RepeatVector(n=traj_size, name='e1')(e1_0)
    inc1 = keras.layers.RepeatVector(n=traj_size, name='inc1')(inc1_0)
    Omega1 = keras.layers.RepeatVector(n=traj_size, name='Omega1')(Omega1_0)
    omega1 = keras.layers.RepeatVector(n=traj_size, name='omega1')(omega1_0)
    mu1 = keras.layers.RepeatVector(n=traj_size, name='mu1')(mu1_0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M1_0_vec = keras.layers.RepeatVector(n=traj_size, name='M1_0_vec')(M1_0)
    N1_0_vec = keras.layers.RepeatVector(n=traj_size, name='N1_0_vec')(N1_0)
    # Compute the mean anomaly M(t) as a function of time
    N1_t = keras.layers.multiply(inputs=[N1_0_vec, t_vec])
    M1 = keras.layers.add(inputs=[M1_0_vec, N1_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f1 = MeanToTrueAnomaly(name='mean_to_true_anomaly_f1')([M1, e1])

    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    elt1 = (a1, e1, inc1, Omega1, omega1, f1, mu1,)

    # ******************************************************************
    # Predict orbital elements for Jacobi coordinates of body 2 
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a2 = keras.layers.RepeatVector(n=traj_size, name='a2')(a2_0)
    e2 = keras.layers.RepeatVector(n=traj_size, name='e2')(e2_0)
    inc2 = keras.layers.RepeatVector(n=traj_size, name='inc2')(inc2_0)
    Omega2 = keras.layers.RepeatVector(n=traj_size, name='Omega2')(Omega2_0)
    omega2 = keras.layers.RepeatVector(n=traj_size, name='omega2')(omega2_0)
    mu2 = keras.layers.RepeatVector(n=traj_size, name='mu2')(mu2_0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M2_0_vec = keras.layers.RepeatVector(n=traj_size, name='M2_0_vec')(M2_0)
    N2_0_vec = keras.layers.RepeatVector(n=traj_size, name='N2_0_vec')(N2_0)
    # Compute the mean anomaly M(t) as a function of time
    N2_t = keras.layers.multiply(inputs=[N2_0_vec, t_vec])
    M2 = keras.layers.add(inputs=[M2_0_vec, N2_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f2 = MeanToTrueAnomaly(name='mean_to_true_anomaly_f2')([M2, e2])

    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    elt2 = (a2, e2, inc2, Omega2, omega2, f2, mu2,)

    # ******************************************************************
    # Convert orbital elements to cartesian Jacobi coordinates 
    
    # Model mapping orbital elements to cartesian coordinates
    model_e2c = make_model_elt_to_cfg(include_accel=True, batch_size=batch_size)

    # The position of Jacobi coordinate 0 over time comes from the average velocity
    # We always use center of momentum coordinates, so this is zero
    qjt_0 = keras.backend.zeros(shape=[batch_size, traj_size, space_dims])
    vjt_0 = keras.backend.zeros(shape=[batch_size, traj_size, space_dims])
    ajt_0 = keras.backend.zeros(shape=[batch_size, traj_size, space_dims])
    
    # Convert from orbital elements to cartesian coordinates
    # This is the position and velocity of the Jacobi coordinate 
    qjt_1, vjt_1, ajt_1 = model_e2c(elt1)
    qjt_2, vjt_2, ajt_2 = model_e2c(elt2)
    
    # Reshape the Jacobi coordinates to include an axis for body number
    particle_traj_shape = (-1, 1, 3)
    particle_traj_shape_layer = keras.layers.Reshape(target_shape=particle_traj_shape, name='particle_traj_shape')
    qjt_0 = particle_traj_shape_layer(qjt_0)
    qjt_1 = particle_traj_shape_layer(qjt_1)
    qjt_2 = particle_traj_shape_layer(qjt_2)
    vjt_0 = particle_traj_shape_layer(vjt_0)
    vjt_1 = particle_traj_shape_layer(vjt_1)
    vjt_2 = particle_traj_shape_layer(vjt_2)
    ajt_0 = particle_traj_shape_layer(ajt_0)
    ajt_1 = particle_traj_shape_layer(ajt_1)
    ajt_2 = particle_traj_shape_layer(ajt_2)

    # Assemble the Jacobi coordinates over time
    qj = keras.layers.concatenate(inputs=[qjt_0, qjt_1, qjt_2], axis=-2, name='qj')
    vj = keras.layers.concatenate(inputs=[vjt_0, vjt_1, vjt_2], axis=-2, name='vj')
    aj = keras.layers.concatenate(inputs=[ajt_0, ajt_1, ajt_2], axis=-2, name='aj')

    # Convert the Jacobi coordinates over time to Cartesian coordinates
    q, v, a = JacobiToCartesian(include_accel=True, batch_size=batch_size)([m, qj, vj, aj])
    
    # Name the outputs
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)
    
    # Wrap up the outputs
    outputs = (q, v, a)

    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_g3b_position_math')
    return model

# ********************************************************************************************************************* 
def make_model_g3b_math(traj_size: int = 1001, batch_size:int = 64, num_gpus:int=1):
    """Create a math model for the restricted three body problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_g3b_math(traj_size=traj_size, 
                                                  batch_size=batch_size, num_gpus=num_gpus)
    
    # Set use_autodiff to false because in the math model, there the kepler velocity and accelartion are exact
    use_autodiff=False
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_g3b(position_model=position_model, use_autodiff=use_autodiff,
                                  traj_size=traj_size, batch_size=batch_size)

