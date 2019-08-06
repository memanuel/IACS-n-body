"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Custom layers and reusable calculations

Michael S. Emanuel
Mon Aug 5 14:30:00 2019
"""

# Library imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
from tf_utils import Identity, EpochLoss, TimeHistory

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class KineticEnergy_G2B(keras.layers.Layer):
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
class PotentialEnergy_G2B(keras.layers.Layer):
    """Compute the potential energy from masses m and positions q"""
    def __init__(self, **kwargs):
        super(PotentialEnergy_G2B, self).__init__(**kwargs)
        # Compute the norm of a 3D vector
        self.norm_func = lambda q : tf.norm(q, axis=-1, keepdims=False)
    
    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of q is (batch_size, traj_size, num_particles, 3,)
        m, q = inputs

        # The gravitational constant; numerical value close to 4 pi^2; see rebound documentation for exact value        
        G = tf.constant(39.476926421373)
        # Gravitational field strength
        m1 = m[:, 0]
        m2 = m[:, 1]
        mu = G * m1 * m2

        # The displacement q_12
        q1 = q[:, :, 0, :]
        q2 = q[:, :, 1, :]
        q_12 = q2 - q1

        # The distance r; shape (batch_size, num_particles, traj_size)
        r = keras.layers.Activation(self.norm_func, name='r')(q_12)

        # Reshape gravitational field constant to match r
        target_shape = [1] * (len(r.shape)-1)
        mu = keras.layers.Reshape(target_shape, name='mu_vec')(mu)
        
        # The gravitational potential is -G m0 m1 / r = - mu / r per unit mass m1 in restricted problem
        U = tf.negative(tf.divide(mu, r))

        return U

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class Momentum_G2B(keras.layers.Layer):
    """Compute the momentum from masses m and velocities v"""

    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of v is (batch_size, traj_size, num_particles, 3,)
        m, v = inputs

        # Reshape mass to (batch_size, 1, num_particles, 1)
        target_shape = (1, 2, 1,)
        m_vec = keras.layers.Reshape(target_shape)(m)
        
        # The momentum is m * v
        # First compute the momentum for each particle
        Pi = 0.5 * m_vec * v
        # Sum the momentum for the whole system
        P = tf.reduce_sum(Pi, axis=-2, keepdims=False)        

        return P

    def get_config(self):
        return dict()
    
# ********************************************************************************************************************* 
class AngularMomentum_G2B(keras.layers.Layer):
    """Compute the angular momentum from position and velocity"""
    def call(self, inputs):
        # Unpack inputs
        # Shape of m is (batch_size, num_particles,)
        # Shape of q, v are (batch_size, traj_size, num_particles, 3,)
        m, q, v = inputs

        # Reshape mass to (batch_size, 1, num_particles, 1)
        target_shape = (1, 2, 1,)
        m_vec = keras.layers.Reshape(target_shape)(m)
        
        # Compute the momentum of each particle, m * v
        P = 0.5 * m_vec * v
        
        # Compute the angular momentum of each particle, q x (m*v)
        Li = tf.linalg.cross(a=q, b=P, name='Li')

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
class Motion_G2B(keras.Model):
    """Motion for general two body problem generated from a position calculation model."""

    def __init__(self, position_model, **kwargs):
        super(Motion_G2B, self).__init__(**kwargs)
        self.position_model = position_model

    def call(self, inputs):
        """
        Compute full orbits for the general two body problem.
        Computes positions using the passed position_layer, 
        then uses automatic differentiation for velocity v and acceleration a.
        INPUTS:
            t: the times to report the orbit; shape (batch_size, traj_size)
            q0: the initial position; shape (batch_size, 2, 3)
            v0: the initial velocity; shape (batch_size, 2, 3)
            m: the object masses; shape (batch_size, 2)
        OUTPUTS:
            q: the position at time t; shape (batch_size, traj_size, 2, 3)
            v: the velocity at time t; shape (batch_size, traj_size, 2, 3)
            a: the acceleration at time t; shape (batch_size, traj_size, 2, 3)
        """
        # Unpack the inputs
        t, q0, v0, m = inputs

        # Get the trajectory size and target shape of t
        traj_size = t.shape[1]
        time_shape = (traj_size, 1)

        # Reshape t to have shape (batch_size, traj_size, 1)
        t = keras.layers.Reshape(target_shape=time_shape, name='t')(t)

        # Evaluation of the position is under the scope of two gradient tapes
        # These are for velocity and acceleration
        with tf.GradientTape(persistent=True) as gt2:
            gt2.watch(t)

            with tf.GradientTape(persistent=True) as gt1:
                gt1.watch(t)       
        
                # Get the position using the input position layer
                position_inputs = (t, q0, v0, m)
                # The velocity from the position model assumes the orbital elements are not changing
                # Here we only want to take the position output and do a full automatic differentiation
                qx, qy, qz, vx_, vy_, vz_ = self.position_model(position_inputs)
                q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=2, name='q')

            # Compute the velocity v = dq/dt with gt1
            vx = gt1.gradient(qx, t)
            vy = gt1.gradient(qy, t)
            vz = gt1.gradient(qz, t)
            v = keras.layers.concatenate(inputs=[vx, vy, vz], axis=2, name='v')
            del gt1
            
        # Compute the acceleration a = d2q/dt2 = dv/dt with gt2
        ax = gt2.gradient(vx, t)
        ay = gt2.gradient(vy, t)
        az = gt2.gradient(vz, t)
        a = keras.layers.concatenate(inputs=[ax, ay, az], name='a')
        del gt2
        
        return q, v, a
    
# ********************************************************************************************************************* 
# Model Factory
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_physics_model_g2b(position_model: keras.Model, traj_size: int):
    """Create a physics model for the general two body problem from a position model"""
    # Create input layers 
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(2, 3,), name='q0')
    v0 = keras.Input(shape=(2, 3,), name='v0')
    m = keras.Input(shape=(1, 2,), name='mu')
    
    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, m)
    
    # Return row 0 of a position or velocity for q0_rec and v0_rec
    initial_row_func = lambda q : q[:, 0, :]

    # Compute the motion from the specified position layer; inputs are the same for position and physics model
    q, v, a = Motion_G2B(position_model=position_model, name='motion')(inputs)
    
    # Name the outputs of the motion
    # These each have shape (batch_size, traj_size, 3)
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)

    # Compute q0_rec and v0_rec
    # These each have shape (batch_size, 2)
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)

    # Compute kinetic energy T and potential energy U
    T = KineticEnergy_G2B(name='T')([m, v])
    U = PotentialEnergy_G2B(name='U')([m, q])

    # Compute the total energy H
    H = keras.layers.add(inputs=[T,U], name='H')

    # Compute momentum P and angular momentum L
    # These have shape (batch_size, traj_size, 3)
    P = Momentum_G2B(name='P')([m, v])
    L = AngularMomentum_G2B(name='L')([m, q, v])

    # Wrap this up into a model
    outputs = [q, v, a, q0_rec, v0_rec, H, P, L]
    model_name = position_model.name.replace('model_g2b_position_', 'model_g2b_physics_')
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# ********************************************************************************************************************* 
# Utility Functions
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def compile_and_fit(model, ds, epochs, loss, optimizer, metrics, save_freq, prev_history=None):
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_name = model.name
    filepath=f'../models/g2b/{model_name}.h5'

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
