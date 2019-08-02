"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Models for restricted two body problem - neural network

Michael S. Emanuel
Thu Aug  1 17:05:57 2019
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
# from tf_utils import EpochLoss, TimeHistory
# import numpy as np

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_r2b_nn(hidden_sizes, skip_layers=True, traj_size = 731):
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
    
    # Reshape the gravitational field strength from (batch_size,) to (batch_size, 1,)
    mu = keras.layers.Reshape((1,), name='mu_reshape')(mu)

    # Tuple of inputs for the model converting from configuration to orbital elements
    inputs_c2e = (q0, v0, mu)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e = make_model_cfg_to_elt()
    
    # Extract the orbital elements of the initial conditions
    a0, e0, inc0, Omega0, omega0, f0, M0, N0 = model_c2e(inputs_c2e)

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a0_vec = keras.layers.RepeatVector(n=traj_size, name='a0_vec')(a0)
    e0_vec = keras.layers.RepeatVector(n=traj_size, name='e0_vec')(e0)
    inc0_vec = keras.layers.RepeatVector(n=traj_size, name='inc0_vec')(inc0)
    Omega0_vec = keras.layers.RepeatVector(n=traj_size, name='Omega0_vec')(Omega0)
    omega0_vec = keras.layers.RepeatVector(n=traj_size, name='omega0_vec')(omega0)
    mu_vec = keras.layers.RepeatVector(n=traj_size, name='mu_vec')(mu)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M0_vec = keras.layers.RepeatVector(n=traj_size, name='M0_vec')(M0)
    N0_vec = keras.layers.RepeatVector(n=traj_size, name='N0_vec')(N0)
    # Compute the mean anomaly M(t) as a function of time
    N_t = keras.layers.multiply(inputs=[N0_vec, t_vec])
    M_vec = keras.layers.add(inputs=[M0_vec, N_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f_vec = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M_vec, e0_vec])
    
    # Combine the trajectory-wide scalars into one feature of shape (batch_size, 14)
    # One row has 3+3+6 = 12 elements
    phi_traj = keras.layers.concatenate(
        inputs=[q0, v0, a0, e0, inc0, Omega0, omega0, N0], 
        name='phi_traj')
    
    # Repeat phi_traj traj_size times so it has a shape of (batch_size, traj_size, 14)
    phi_traj_vec = keras.layers.RepeatVector(n=traj_size, name='phi_traj_vec')(phi_traj)

    # Combine the following into an initial feature vector, phi_0
    # 1) The time t
    # 2) The repeated orbital elements (which remain constant); not including f0
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
    a = keras.layers.add(inputs=[a0_vec, delta_a], name='a')
    a = tf.nn.relu(a)
    
    e = keras.layers.add(inputs=[e0_vec, delta_e], name='e')
    e = tf.clip_by_value(e, 0.0, 1.0)
    
    inc = keras.layers.add(inputs=[inc0_vec, delta_inc], name='inc')
    Omega = keras.layers.add(inputs=[Omega0_vec, delta_Omega], name='Omega')
    omega = keras.layers.add(inputs=[omega0_vec, delta_omega], name='omega')
    f = keras.layers.add(inputs=[f_vec, delta_f], name='f')
    
    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    inputs_e2c = (a, e, inc, Omega, omega, f, mu,)
    
    # Convert from orbital elements to cartesian
    qx, qy, qz, vx, vy, vz = OrbitalElementToConfig(name='orbital_element_to_config')(inputs_e2c)
    
    # Wrap up the outputs
    outputs = (qx, qy, qz, vx, vy, vz)
    
    # Wrap this into a model
    suffix = '_'.join(str(sz) for sz in hidden_sizes)
    model_name = f'model_r2b_nn_{suffix}'
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# ********************************************************************************************************************* 
def make_model_r2b_nn(hidden_sizes, skip_layers=True, traj_size = 731):
    """Create a math model for the restricted two body circular problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_r2b_nn(hidden_sizes=hidden_sizes, skip_layers=skip_layers, traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_r2b(position_model=position_model, traj_size=traj_size)

