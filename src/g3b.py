"""
Harvard IACS Masters Thesis
General Three Body Problem
Custom layers and reusable calculations

Michael S. Emanuel
Thu Aug 8 11:43:00 2019
"""

# Library imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tf_utils import Identity, EpochLoss, TimeHistory
from orbital_element import G_

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class KineticEnergy_G3B(keras.layers.Layer):
    """Compute the kinetic energy from masses m and velocities v"""

    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of v is (batch_size, traj_size, num_particles, 3,)
        m, v = inputs

        # Element-wise velocity squared
        v2 = tf.square(v)
        # Reshape mass to (batch_size, 1, num_particles)
        target_shape = (1, -1)
        m_vec = keras.layers.Reshape(target_shape)(m)
        
        # The KE is 1/2 m v^2 = 1/2 v^2 per unit mass in restricted problem
        # First compute the K.E. for each particle
        Ti = 0.5 * m_vec * tf.reduce_sum(v2, axis=-1, keepdims=False)
        # Sum the K.E. for the whole system
        T = tf.reduce_sum(Ti, axis=-1, keepdims=False)        

        return T

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class PotentialEnergy_G3B(keras.layers.Layer):
    """Compute the potential energy from masses m and positions q"""
    def __init__(self, **kwargs):
        super(PotentialEnergy_G3B, self).__init__(**kwargs)
        # Compute the norm of a 3D vector
        self.norm_func = lambda q : tf.norm(q, axis=-1, keepdims=False)
    
    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of q is (batch_size, traj_size, num_particles, 3,)
        m, q = inputs

        # The gravitational constant; numerical value close to 4 pi^2; see rebound documentation for exact value        
        G = tf.constant(G_)
        # Gravitational field strength
        m0 = m[:, 0]
        m1 = m[:, 1]
        m2 = m[:, 2]
        k_01 = G * m0 * m1
        k_02 = G * m0 * m2
        k_12 = G * m1 * m2

        # The displacement q_12
        q0 = q[:, :, 0, :]
        q1 = q[:, :, 1, :]
        q2 = q[:, :, 2, :]
        q_01 = q1 - q0
        q_02 = q2 - q0
        q_12 = q2 - q1

        # The distance r; shape (batch_size, num_particles, traj_size)
        r_01 = keras.layers.Activation(self.norm_func, name='r_01')(q_01)
        r_02 = keras.layers.Activation(self.norm_func, name='r_02')(q_02)
        r_12 = keras.layers.Activation(self.norm_func, name='r_12')(q_12)

        # Reshape gravitational field constant to match r
        target_shape = [1] * (len(r_01.shape)-1)
        k_01 = keras.layers.Reshape(target_shape, name='k_01_vec')(k_01)
        k_02 = keras.layers.Reshape(target_shape, name='k_02_vec')(k_02)
        k_12 = keras.layers.Reshape(target_shape, name='k_12_vec')(k_12)
        
        # The gravitational potential is -G mi mj / r_1i = - k_ij / r_ij
        U = tf.negative(tf.reduce_sum([tf.divide(k_01, r_01),
                                       tf.divide(k_02, r_02),
                                       tf.divide(k_12, r_12),],
                                      axis=0, keepdims=False))

        return U

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class Momentum_G3B(keras.layers.Layer):
    """Compute the momentum from masses m and velocities v"""

    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of v is (batch_size, traj_size, num_particles, 3,)
        m, v = inputs

        # Reshape mass to (batch_size, 1, num_particles, 1)
        num_particles=3
        target_shape = (1, num_particles, 1,)
        m_vec = keras.layers.Reshape(target_shape)(m)
        
        # The momentum is m * v
        # First compute the momentum for each particle
        Pi = m_vec * v
        # Sum the momentum for the whole system
        P = tf.reduce_sum(Pi, axis=-2, keepdims=False)        

        return P

    def get_config(self):
        return dict()
    
# ********************************************************************************************************************* 
class AngularMomentum_G3B(keras.layers.Layer):
    """Compute the angular momentum from position and velocity"""
    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of q, v are (batch_size, traj_size, num_particles, 3,)
        m, q, v = inputs

        # Reshape mass to (batch_size, 1, num_particles, 1)
        num_particles=3
        target_shape = (1, num_particles, 1,)
        m_vec = keras.layers.Reshape(target_shape)(m)
        
        # Compute the momentum of each particle, m * v
        Pi = m_vec * v
        
        # Compute the angular momentum of each particle, q x (m*v)
        Li = tf.linalg.cross(a=q, b=Pi, name='Li')

        # Sum the angular momentum for the whole system
        L = tf.reduce_sum(Li, axis=-2, keepdims=False)        

        return L

# ********************************************************************************************************************* 
# Custom Losses
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class VectorError(keras.losses.Loss):
    """Specialized loss for error in a vector like velocity or acceleration."""
    def __init__(self, regularizer=0.0, **kwargs):
        super(VectorError, self).__init__(**kwargs)
        self.regularizer = tf.constant(regularizer)

    def call(self, y_true, y_pred):
        err2_num = math_ops.reduce_sum(math_ops.square(y_true - y_pred), axis=-1)
        err2_den = math_ops.reduce_sum(math_ops.square(y_true), axis=-1) + self.regularizer
        return K.mean(err2_num / err2_den)

# ********************************************************************************************************************* 
class EnergyError(keras.losses.Loss):
    """Specialized loss for error in energy.  Mean of log (1 + relative error^2)."""
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        rel_error = (y_pred - y_true) / y_true
        # rel_abs_error = math_ops.abs(rel_error)
        rel_sq_error =  math_ops.square(rel_error)
        log_rel_error = math_ops.log1p(rel_sq_error)
        return K.mean(log_rel_error, axis=-1)

# ********************************************************************************************************************* 
# Custom Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class Motion_G3B(keras.Model):
    """Motion for general three body problem generated from a position calculation model."""

    def __init__(self, position_model, **kwargs):
        super(Motion_G3B, self).__init__(**kwargs)
        self.position_model = position_model

    def call(self, inputs):
        """
        Compute full orbits for the general two body problem.
        Computes positions using the passed position_layer, 
        then uses automatic differentiation for velocity v and acceleration a.
        INPUTS:
            t: the times to report the orbit; shape (batch_size, traj_size)
            q0: the initial position; shape (batch_size, 3, 3)
            v0: the initial velocity; shape (batch_size, 3, 3)
            m: the object masses; shape (batch_size, 3)
        OUTPUTS:
            q: the position at time t; shape (batch_size, traj_size, 3, 3)
            v: the velocity at time t; shape (batch_size, traj_size, 3, 3)
            a: the acceleration at time t; shape (batch_size, traj_size, 3, 3)
        """
        # Unpack the inputs
        t, q0, v0, m = inputs

        # Get the trajectory size and target shape of t
        traj_size = t.shape[1]
        time_shape = (traj_size, 1)

        # Reshape t to have shape (batch_size, traj_size, 1)
        t = keras.layers.Reshape(target_shape=time_shape, name='t')(t)

        # Shape of trajectory (position or velocity) of one particle
        space_dims = 3
        particle_traj_shape = (-1, 1, space_dims)
        particle_traj_shape_layer = keras.layers.Reshape(target_shape=particle_traj_shape, name='particle_traj_shape')

        # Evaluation of the position is under the scope of two gradient tapes
        # These are for velocity and acceleration
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
            gt2.watch(t)

            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
                gt1.watch(t)       
        
                # Get the position using the input position layer
                position_inputs = (t, q0, v0, m)
                # The velocity from the position model assumes the orbital elements are not changing
                # Here we only want to take the position output and do a full automatic differentiation
                q, v_, a_ = self.position_model(position_inputs)
                # Unpack separate components of position; unable to get gt.jacobian or output_gradients to work
                q0x, q0y, q0z = q[:, :, 0, 0], q[:, :, 0, 1], q[:, :, 0, 2]
                q1x, q1y, q1z = q[:, :, 1, 0], q[:, :, 1, 1], q[:, :, 1, 2]
                q2x, q2y, q2z = q[:, :, 2, 0], q[:, :, 2, 1], q[:, :, 2, 2]

            # Compute the velocity v = dq/dt with gt1
            v0x = gt1.gradient(q0x, t)
            v0y = gt1.gradient(q0y, t)
            v0z = gt1.gradient(q0z, t)
            v1x = gt1.gradient(q1x, t)
            v1y = gt1.gradient(q1y, t)
            v1z = gt1.gradient(q1z, t)
            v2x = gt1.gradient(q2x, t)
            v2y = gt1.gradient(q2y, t)
            v2z = gt1.gradient(q2z, t)

            # Assemble the velocity components for each particle
            v0 = keras.layers.concatenate(inputs=[v0x, v0y, v0z], axis=-1, name='v0')
            v1 = keras.layers.concatenate(inputs=[v1x, v1y, v1z], axis=-1, name='v1')
            v2 = keras.layers.concatenate(inputs=[v2x, v2y, v2z], axis=-1, name='v2')
            # Reshape the single particle trajectories
            v0 = particle_traj_shape_layer(v0)
            v1 = particle_traj_shape_layer(v1)
            v2 = particle_traj_shape_layer(v2)
            # Combine the particle velocity vectors
            v = keras.layers.concatenate(inputs=[v0, v1, v2], axis=-2, name='v')
            del gt1
            
        # Compute the acceleration a = d2q/dt2 = dv/dt with gt2
        a0x = gt2.gradient(v0x, t)
        a0y = gt2.gradient(v0y, t)
        a0z = gt2.gradient(v0z, t)
        a1x = gt2.gradient(v1x, t)
        a1y = gt2.gradient(v1y, t)
        a1z = gt2.gradient(v1z, t)
        a2x = gt2.gradient(v2x, t)
        a2y = gt2.gradient(v2y, t)
        a2z = gt2.gradient(v2z, t)

        # Assemble the velocity components for each particle
        a0 = keras.layers.concatenate(inputs=[a0x, a0y, a0z], axis=-1, name='a0')
        a1 = keras.layers.concatenate(inputs=[a1x, a1y, a1z], axis=-1, name='a1')
        a2 = keras.layers.concatenate(inputs=[a2x, a2y, a2z], axis=-1, name='a2')
        # Reshape the single particle trajectories
        a0 = particle_traj_shape_layer(a0)
        a1 = particle_traj_shape_layer(a1)
        a2 = particle_traj_shape_layer(a2)
        # Combine the particle acceleration vectors
        a = keras.layers.concatenate(inputs=[a0, a1, a2], axis=-2, name='a')
        del gt2

        return q, v, a
    
# ********************************************************************************************************************* 
# Model Factory
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_physics_model_g3b(position_model: keras.Model, 
                           use_autodiff: bool,
                           traj_size: int, 
                           batch_size: int=64):
    """Create a physics model for the general two body problem from a position model"""
    # Create input layers
    num_particles = 3
    space_dims = 3
    t = keras.Input(shape=(traj_size,), batch_size=batch_size, name='t')
    q0 = keras.Input(shape=(num_particles, space_dims,), batch_size=batch_size, name='q0')
    v0 = keras.Input(shape=(num_particles, space_dims,), batch_size=batch_size, name='v0')
    m = keras.Input(shape=(num_particles,), batch_size=batch_size, name='m')
    
    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, m)

    # Return row 0 of a position or velocity for q0_rec and v0_rec
    initial_row_func = lambda q : q[:, 0, :]

    # Compute the motion from the specified position layer; inputs are the same for position and physics model
    if use_autodiff:
        q, v, a = Motion_G3B(position_model=position_model, name='motion')(inputs)
    else:
        q, v, a = position_model(inputs)
    
    # Name the outputs of the motion
    # These each have shape (batch_size, num_particles, traj_size, 3)
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)

    # Compute q0_rec and v0_rec
    # These each have shape (batch_size, num_particles, 3)
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)

    # Compute kinetic energy T and potential energy U
    # These have shape (batch_size, traj_size)
    T = KineticEnergy_G3B(name='T')([m, v])
    U = PotentialEnergy_G3B(name='U')([m, q])

    # Compute the total energy H
    H = keras.layers.add(inputs=[T,U], name='H')

    # Compute momentum P and angular momentum L
    # These have shape (batch_size, traj_size, 3)
    P = Momentum_G3B(name='P')([m, v])
    L = AngularMomentum_G3B(name='L')([m, q, v])

    # Wrap this up into a model
    outputs = [q, v, a, q0_rec, v0_rec, H, P, L]
    model_name = position_model.name.replace('model_g3b_position_', 'model_g3b_')
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# ********************************************************************************************************************* 
# Utility Functions
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def fit_model(model, ds, epochs, loss, optimizer, metrics, 
                    save_freq, prev_history=None, batch_num=0):
    # Name for model data
    model_name = model.name
    filepath=f'../models/g3b/{model_name}_batch_{batch_num:03}.h5'

    # Parameters for callbacks
    interval = 1
    # patience = max(epochs // 10, 1)
    
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

    # All selected callbacks
    # callbacks = [cb_log, cb_time, cb_ckp, cb_early_stop]
    callbacks = [cb_log, cb_time, cb_ckp]
    
    # Fit the model
    hist = model.fit(ds, epochs=epochs, callbacks=callbacks, verbose=1)

    # Add the times to the history
    history = hist.history
    
    # Convert from lists to numpy arrays 
    for key in history.keys():
        history[key] = np.array(history[key], dtype=np.float32)
        
    # Add the batch num to history
    history['batch_num'] = np.array([batch_num], dtype=np.float32)

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
        history['time'] = np.array(cb_time.times, dtype=np.float32)
    
    # Restore the model to the best weights
    model.load_weights(filepath)
    
    # Return the history
    return history