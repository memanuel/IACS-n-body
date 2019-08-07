"""
Harvard IACS Masters Thesis
General Two Body Problem
Models for general two body problem - neural network

Michael S. Emanuel
Wed Aug  7 13:31:04 2019
"""

# Library imports
import tensorflow as tf

# Aliases
keras = tf.keras

# Local imports
from orbital_element import make_model_cfg_to_elt, make_model_elt_to_cfg
from orbital_element import MeanToTrueAnomaly, G_
from g2b import make_physics_model_g2b

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_g2b_nn(hidden_sizes, skip_layers=True, traj_size = 731):
    """
    Compute orbit positions for the general two body problem from 
    the initial orbital elements with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    # Create input layers 
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

    # Unpack masses and calculate total mass
    m1 = m[:, 0]
    m2 = m[:, 1]
    m_tot = m1 + m2

    # Gravitational field strength; shape (batch_size,)
    r2_mu = G * m_tot

    # Reshape the gravitational field strength from (batch_size,) to (batch_size, 1,)
    r2_mu = keras.layers.Reshape((1,))(r2_mu)

    # The initial position and velocity of the two particles
    init_pos_shape = (space_dims,)
    init_pos_shape_layer = keras.layers.Reshape(target_shape=init_pos_shape, name='init_pos_shape')
    q0_p1 = init_pos_shape_layer(q0[:, 0, :])
    q0_p2 = init_pos_shape_layer(q0[:, 1, :])
    v0_p1 = init_pos_shape_layer(v0[:, 0, :])
    v0_p2 = init_pos_shape_layer(v0[:, 1, :])

    # The relative position and relative velocity in Jacobi coordinates
    r2_q0 = q0_p2 - q0_p1
    r2_v0 = v0_p2 - v0_p1

    # Tuple of inputs for the model converting from configuration to orbital elements
    r2_cfg = (r2_q0, r2_v0, r2_mu)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e = make_model_cfg_to_elt()

    # Extract the orbital elements of the initial conditions
    a0, e0, inc0, Omega0, omega0, f0, M0, N0 = model_c2e(r2_cfg)
    # Alias r2_mu for naming consistency
    mu0 = r2_mu

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)

    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a0_vec = keras.layers.RepeatVector(n=traj_size, name='a0_vec')(a0)
    e0_vec = keras.layers.RepeatVector(n=traj_size, name='e0_vec')(e0)
    inc0_vec = keras.layers.RepeatVector(n=traj_size, name='inc0_vec')(inc0)
    Omega0_vec = keras.layers.RepeatVector(n=traj_size, name='Omega0_vec')(Omega0)
    omega0_vec = keras.layers.RepeatVector(n=traj_size, name='omega0_vec')(omega0)
    # f0_vec = keras.layers.RepeatVector(n=traj_size, name='f0_vec')(f0)
    
    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M0_vec = keras.layers.RepeatVector(n=traj_size, name='M0_vec')(M0)
    N0_vec = keras.layers.RepeatVector(n=traj_size, name='N0_vec')(N0)
    # Compute the mean anomaly M(t) as a function of time
    N_t = keras.layers.multiply(inputs=[N0_vec, t_vec], name='N_t')
    M_vec = keras.layers.add(inputs=[M0_vec, N_t], name='M_vec')

    # Compute the true anomaly from the mean anomly and eccentricity
    f_vec = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M_vec, e0_vec])
    
    # Combine the trajectory-wide scalars into one feature of shape (batch_size, num_features)
    phi_traj = keras.layers.concatenate(
        inputs=[m, q0_p1, q0_p2,
                a0, e0, inc0, Omega0, omega0, N0], 
        name='phi_traj')
    
    # Repeat phi_traj traj_size times so it has a shape of (batch_size, traj_size, num_features)
    phi_traj_vec = keras.layers.RepeatVector(n=traj_size, name='phi_traj_vec')(phi_traj)

    # Combine the following into an initial feature vector, phi_0
    # 1) The time t
    # 2) The repeated orbital elements (which remain constant)
    # 3) The computed mean anomaly M and true anomaly f
    phi_0 = keras.layers.concatenate(
        inputs=[t_vec, phi_traj_vec, M_vec, f_vec], 
        name='phi_0')
    
    # Hidden layers as specified in hidden_sizes
    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # phi_n will update to the last available feature layer for the output portion
    phi_n = phi_0

    # First hidden layer if applicable
    if num_layers > 0:
        phi_1 = keras.layers.Dense(units=hidden_sizes[0], activation='tanh', name='phi_1')(phi_0)
        if skip_layers:
            phi_1 = keras.layers.concatenate(inputs=[phi_0, phi_1], name='phi_1_aug')
        phi_n = phi_1

    # Second hidden layer if applicable
    if num_layers > 1:
        phi_2 = keras.layers.Dense(units=hidden_sizes[1], activation='tanh', name='phi_2')(phi_1)
        if skip_layers:
            phi_2 = keras.layers.concatenate(inputs=[phi_1, phi_2], name='phi_2_aug')
        phi_n = phi_2

    # Compute the change in orbital elements from the final features
    
    # Semimajor axis
    delta_a = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='delta_a')(phi_n)

    # Eccentricity
    delta_e = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='delta_e')(phi_n)

    # Inclination
    delta_inc = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='delta_inc')(phi_n)
    
    # Longitude of ascending node
    delta_Omega = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='delta_Omega')(phi_n)
    
    # Argument of periapsis
    delta_omega = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='delta_omega')(phi_n)
    
    # True anomaly
    delta_f = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='delta_f')(phi_n)    

    # Compute the orbital elements as the sum of the original elemenets and their change
    # Limit a to be non-negative
    a = keras.layers.add(inputs=[a0_vec, delta_a], name='a')
    a = keras.layers.ReLU(name='a_relu')(a)
    
    # Limit e to be in interval [0.0, 1.0]
    e = keras.layers.add(inputs=[e0_vec, delta_e], name='e')
    # e = tf.clip_by_value(e, 0.0, 1.0)
    clip_func = lambda x : tf.clip_by_value(x, 0.0, 1.0)
    e = keras.layers.Activation(activation=clip_func, name='e_clip')(e)
    
    # The remaining elements can take any value; angles can wrap around past 2pi
    inc = keras.layers.add(inputs=[inc0_vec, delta_inc], name='inc')
    Omega = keras.layers.add(inputs=[Omega0_vec, delta_Omega], name='Omega')
    omega = keras.layers.add(inputs=[omega0_vec, delta_omega], name='omega')
    f = keras.layers.add(inputs=[f_vec, delta_f], name='f')
    
    # The gravitational field strength mu is a constant over each trajectory
    # Reshape it to match the other elements
    mu = keras.layers.RepeatVector(n=traj_size, name='mu')(mu0)

    # Wrap orbital elements into one tuple of inputs for model converting to cartesian coordinates
    r2_elt = (a, e, inc, Omega, omega, f, mu,)

    # Model mapping orbital elements to cartesian coordinates
    model_e2c = make_model_elt_to_cfg()

    # Convert from orbital elements to cartesian coordinates
    # This is the position and velocity of the Jacobi coordinate r2 = q2 - q1
    r2_q, r2_v = model_e2c(r2_elt)

    # Reshape coefficients for q1 and q2 from r2
    coeff_shape = (1,1,)
    coeff_shape_layer = keras.layers.Reshape(target_shape=coeff_shape, name='coeff_shape')
    coeff1 = coeff_shape_layer(-m2 / m_tot)
    coeff2 = coeff_shape_layer( m1 / m_tot)

    # Compute the position and velocity of the individual particles from the Jacobi coordinates
    q1 = coeff1 * r2_q
    q2 = coeff2 * r2_q
    v1 = coeff1 * r2_v
    v2 = coeff2 * r2_v

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
    suffix = '_'.join(str(sz) for sz in hidden_sizes)
    model_name = f'model_g2b_nn_{suffix}'
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


# ********************************************************************************************************************* 
def make_model_g2b_nn(hidden_sizes, skip_layers=True, traj_size = 731):
    """Create a NN model for the general two body problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_g2b_nn(hidden_sizes=hidden_sizes, 
                                                skip_layers=skip_layers, traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_g2b(position_model=position_model, traj_size=traj_size)

