"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Models for circular motion with neural nets

Michael S. Emanuel
Thu Jun 27 21:43:17 2019
"""


# Library imports
import tensorflow as tf
import numpy as np

# Aliases
keras = tf.keras

# Local imports
from tf_utils import Identity, EpochLoss, TimeHistory
from r2b import KineticEnergy_R2B, PotentialEnergy_R2B, AngularMomentum_R2B
from r2b import ConfigToPolar2D
from r2b import Motion_R2B

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

def make_position_model_r2bc_nn(hidden_sizes, skip_layers=True, traj_size = 731):
    """
    Compute orbit positions for the restricted two body circular problem from 
    the initial polar coordinates (orbital elements) with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    # Create input layers
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(2,), name='q0')
    v0 = keras.Input(shape=(2,), name='v0')
    mu = keras.Input(shape=(1,), name='mu')
    # The combined input layers
    inputs = [t, q0, v0, mu]

    # Check sizes
    batch_size = t.shape[0]
    tf.debugging.assert_shapes(shapes={
        t: (batch_size, traj_size),
        q0: (batch_size, 2),
        v0: (batch_size, 2),
        mu: (batch_size, 1),
    }, message='make_position_model_r2bc_nn / inputs')
    
    # The polar coordinates of the initial conditions
    # r0, theta0, and omega0 each scalars in each batch
    r0, theta0, omega0 = ConfigToPolar2D(name='polar0')([q0, v0])
    
    # Name the outputs of the initial polar
    # These each have shape (batch_size, 1)
    r0 = Identity(name='r0')(r0)
    theta0 = Identity(name='theta0')(theta0)
    omega0 = Identity(name='omega0')(omega0)

    # Check sizes
    tf.debugging.assert_shapes(shapes={
        r0: (batch_size, 1),
        theta0: (batch_size, 1),
        omega0: (batch_size, 1),
    }, message='make_position_model_r2bc_nn / polar elements r0, theta0, omega0')
   
    # Combine the trajectory-wide scalars into one feature
    # Size of each row is 2+2+1+1+1=7; shape is (batch_size, 7)
    phi_traj = keras.layers.concatenate(inputs=[q0, v0, r0, theta0, omega0], name='phi_traj')
    
    # Repeat phi_traj traj_size times so it has a shape of (batch_size, traj_size, 7)
    phi_traj_vec = keras.layers.RepeatVector(n=traj_size, name='phi_traj_vec')(phi_traj)

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # Concatenate phi_traj with the time t to make the initial feature vector phi_0
    phi_0 = keras.layers.concatenate(inputs=[t_vec, phi_traj_vec], name='phi_0')
    
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
    
    # Check shapes
    tf.debugging.assert_shapes(shapes={
        phi_traj: (batch_size, 7),
        phi_traj_vec: (batch_size, traj_size, 7),        
        t_vec: (batch_size, traj_size, 1),
        phi_0: (batch_size, traj_size, 8),
        # phi_1: (batch_size, traj_size, 'hs1'),
        # phi_2: (batch_size, traj_size, 'hs2'),
    }, message='make_position_model_r2bc_nn / phi')

    # Compute the radius r from the features; initialize weights to 0, bias to 1
    r = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='ones', name='r')(phi_n)
    
    # Compute the mean angular frequency omega from the features
    omega = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='omega') (phi_n)
    
    # Add a feature for omega * t + theta0; a rough estimate of the angular offset
    omega_t = keras.layers.multiply(inputs=[omega, t_vec], name='omega_t')

    # Compute the offset to angle theta to its mean trend from the features
    theta_adj = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros', name='theta_adj')(phi_n)
    
    # Compute the angle theta as the sum of its original offset theta0, its trend rate omega_bar_t
    # and the adjustment above
    theta0_vec = keras.layers.RepeatVector(n=traj_size, name='theta0_vec')(theta0)
    theta = keras.layers.add(inputs=[omega_t, theta0_vec, theta_adj], name='theta')
    
    # Check shapes
    tf.debugging.assert_shapes(shapes={
        r: (batch_size, traj_size, 1),
        omega: (batch_size, traj_size, 1),
        omega_t: (batch_size, traj_size, 1),
        theta_adj: (batch_size, traj_size, 1),
        theta0_vec: (batch_size, traj_size, 1),
        theta: (batch_size, traj_size, 1)
    }, message='make_position_model_r2bc_nn / r, omega, theta')
    
    # Cosine and sine of theta
    cos_theta = keras.layers.Activation(activation=tf.cos, name='cos_theta')(theta)
    sin_theta = keras.layers.Activation(activation=tf.sin, name='sin_theta')(theta)

    # Compute qx and qy from r, theta
    qx = keras.layers.multiply(inputs=[r, cos_theta], name='qx')
    qy = keras.layers.multiply(inputs=[r, sin_theta], name='qy')
    
    # Check shapes
    batch_size = t.shape[0]
    tf.debugging.assert_shapes(shapes={
        cos_theta: (batch_size, traj_size, 1),
        sin_theta: (batch_size, traj_size, 1),
        qx: (batch_size, traj_size, 1),
        qy: (batch_size, traj_size, 1),
    }, message='make_position_model_r2bc_nn / outputs')
    
    # Wrap this into a model
    outputs = [qx, qy]
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_r2bc_position_nn')
    return model

# ********************************************************************************************************************* 
def make_physics_model_r2bc_nn(position_model: keras.Model, 
                               traj_size: int, 
                               model_name: str):
    """Create a physics model for the restricted two body problem from a position model"""
    # Create input layers
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(2,), name='q0')
    v0 = keras.Input(shape=(2,), name='v0')
    mu = keras.Input(shape=(1,), name='mu')
    # The combined input layers
    inputs = [t, q0, v0, mu]

    # Check sizes
    batch_size = t.shape[0]
    tf.debugging.assert_shapes(shapes={
        t: (batch_size, traj_size),
        q0: (batch_size, 2),
        v0: (batch_size, 2),
        mu: (batch_size, 1),
    }, message='make_physics_model_r2bc_nn / inputs')
        
    # Return row 0 of a position or velocity for q0_rec and v0_rec
    initial_row_func = lambda q : q[:, 0, :]
       
    # Compute the motion from the specified position layer
    q, v, a = Motion_R2B(position_model=position_model, name='motion')([t, q0, v0, mu])

    # Name the outputs of the circular motion
    # These each have shape (batch_size, traj_size, 2)
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)

    # Check sizes
    tf.debugging.assert_shapes(shapes={
        q: (batch_size, traj_size, 2),
        v: (batch_size, traj_size, 2),
        a: (batch_size, traj_size, 2),
    }, message='make_physics_model_r2bc_nn / outputs q, v, a')
        
    # Compute q0_rec and v0_rec
    # These each have shape (batch_size, 2)
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)

    # Check sizes
    tf.debugging.assert_shapes(shapes={
        q0_rec: (batch_size, 2),
        v0_rec: (batch_size, 2),
    }, message='make_physics_model_r2bc_nn / outputs q0_rec, v0_rec')

    # Compute kinetic energy T and potential energy U
    T = KineticEnergy_R2B(name='T')(v)
    U = PotentialEnergy_R2B(name='U')([q, mu])

    # Compute the total energy H
    H = keras.layers.add(inputs=[T,U], name='H')

    # Compute angular momentum L
    # This has shape (batch_size, traj_size)
    L = AngularMomentum_R2B(name='L')([q, v])
    
    # Check sizes
    tf.debugging.assert_shapes(shapes={
        T: (batch_size, traj_size),
        U: (batch_size, traj_size),
        H: (batch_size, traj_size),
        L: (batch_size, traj_size),
    }, message='make_physics_model_r2bc_nn / outputs H, L')

    # Wrap this up into a model
    outputs = [q, v, a, q0_rec, v0_rec, H, L]
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# ********************************************************************************************************************* 
def make_model_r2bc_nn(hidden_sizes, skip_layers=True, traj_size: int = 731):
    """Create a neural net model for the restricted two body circular problem"""
    # Build the position model
    position_model = make_position_model_r2bc_nn(hidden_sizes=hidden_sizes, skip_layers=skip_layers)
    
    # Build the model with this position model and the input trajectory size
    model_name = 'model_r2bc_nn_' + '_'.join([str(sz) for sz in hidden_sizes])
    return make_physics_model_r2bc_nn(position_model=position_model, traj_size=traj_size, model_name=model_name)

# ********************************************************************************************************************* 
def compile_and_fit(model, ds, epochs, loss, optimizer, metrics, save_freq, prev_history=None):
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_name = model.name
    filepath=f'../models/r2bc/{model_name}.h5'

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
    history['time'] = cb_time.times
    
    # Merge the previous history if provided
    if prev_history is not None:
        for key in prev_history.keys():
            history[key] = np.concatenate([prev_history[key], history[key]])
    
    # Restore the model to the best weights
    model.load_weights(filepath)

    return history
