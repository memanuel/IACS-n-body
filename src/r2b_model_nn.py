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
from tf_utils import EpochLoss, TimeHistory
import numpy as np

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
    mu0 = keras.layers.Reshape((1,))(mu)

    # Tuple of inputs for the model converting from configuration to orbital elements
    inputs_c2e = (q0, v0, mu0)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e = make_model_cfg_to_elt()
    
    # Extract the orbital elements of the initial conditions
    a0, e0, inc0, Omega0, omega0, f0, M0, N0 = model_c2e(inputs_c2e)

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    # a = keras.layers.RepeatVector(n=traj_size, name='a')(a0)
    e_ = keras.layers.RepeatVector(n=traj_size, name='e_')(e0)
    # inc = keras.layers.RepeatVector(n=traj_size, name='inc')(inc0)
    # Omega = keras.layers.RepeatVector(n=traj_size, name='Omega')(Omega0)
    # omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)
    mu = keras.layers.RepeatVector(n=traj_size, name='mu_vec')(mu0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M0_vec = keras.layers.RepeatVector(n=traj_size, name='M0_vec')(M0)
    N0_vec = keras.layers.RepeatVector(n=traj_size, name='N0_vec')(N0)
    # Compute the mean anomaly M(t) as a function of time
    N_t = keras.layers.multiply(inputs=[N0_vec, t_vec])
    M_ = keras.layers.add(inputs=[M0_vec, N_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f_ = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M_, e_])
    
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
        inputs=[t_vec, phi_traj_vec, M_, f_], 
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

    # Compute the orbital elements from the final features
    
    # Semimajor axis
    a = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='ones', name='a')(phi_n)

    # Eccentricity
    e = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='e')(phi_n)

    # Inclination
    inc = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='inc')(phi_n)
    
    # Longitude of ascending node
    Omega = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='Omega')(phi_n)
    
    # Argument of periapsis
    omega = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='omega')(phi_n)
    
    # True anomaly
    f = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='f')(phi_n)    

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

# ********************************************************************************************************************* 
def compile_and_fit(model, ds, epochs, loss, optimizer, metrics, save_freq, prev_history=None):
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_name = model.name
    filepath=f'../models/r2b/{model_name}.h5'

    # Parameters for callbacks
    interval = 1
    patience = max(epochs // 10, 1)
    
    # Create callbacks
    cb_log = EpochLoss(interval=interval, newline=True)
    
    cb_time = TimeHistory()

    cb_ckp = keras.callbacks.ModelCheckpoint(
            filepath=filepath, 
            save_freq=save_freq,
            save_best_only=True,
            save_weights_only=True,
            monitor='loss',
            verbose=0)    

    cb_early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.0,
            patience=patience,
            verbose=2,
            restore_best_weights=True)    

    # All selected callbacks
    # callbacks = [cb_log, cb_time, cb_ckp, cb_early_stop]
    callbacks = [cb_log, cb_time, cb_ckp]
    
    # Fit the model
    hist = model.fit(ds, epochs=epochs, callbacks=callbacks, verbose=1)

    # Add the times to the history
    history = hist.history
    
    # Convert from lists to numpy arrays 
    for key in history.keys():
        history[key] = np.array(history[key])

    # Merge the previous history if provided
    if prev_history is not None:
        for key in history.keys():
            history[key] = np.concatenate([prev_history[key], history[key]])
        # Merge wall times; new times need to add offset from previous wall time
        prev_times = prev_history['time']
        time_offset = prev_times[-1]
        new_times = np.array(cb_time.times) + time_offset
        history['time'] = np.concatenate([prev_times, new_times])
    else:
        history['time'] = np.array(cb_time.times)
    
    # Restore the model to the best weights
    model.load_weights(filepath)

    return history
