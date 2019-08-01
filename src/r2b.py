"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Custom layers and reusable calculations

Michael S. Emanuel
Tue Jul 30 11:44:00 2019
"""

# Library imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
# import numpy as np

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class KineticEnergy_R2B(keras.layers.Layer):
    """Compute the kinetic energy from velocity"""

    def call(self, inputs):
        # Alias inputs to v
        # Shape of v is (batch_size, traj_size, 3,)
        v = inputs

        # Element-wise velocity squared
        v2 = tf.square(v)
        
        # The KE is 1/2 m v^2 = 1/2 v^2 per unit mass in restricted problem
        T = 0.5 * tf.reduce_sum(v2, axis=-1, keepdims=False)

        # Check shapes
        batch_size, traj_size = v.shape[0:2]
        tf.debugging.assert_shapes(shapes={
            v: (batch_size, traj_size, 3),
            v2: (batch_size, traj_size, 3),
            T: (batch_size, traj_size)
        }, message='KineticEnergy_R2B')
        
        return T

    def get_config(self):
        return dict()
   
# ********************************************************************************************************************* 
class PotentialEnergy_R2B(keras.layers.Layer):
    """Compute the potential energy from position q and gravitational constant mu"""

    def call(self, inputs):
        # Unpack inputs
        q = inputs

        # Shape of q is (batch_size, traj_size, 3,)
        batch_size, traj_size = q.shape[0:2]
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 3),
        }, message='PotentialEnergy_R2B / inputs')

        # The gravitational constant
        # The numerical value mu0 is close to 4 pi^2; see rebound documentation for exact value
        
        mu = tf.constant(-39.476924896240234)

        # Compute the norm of a 2D vector
        norm_func = lambda q : tf.norm(q, axis=-1, keepdims=False)

        # The distance r; shape (batch_size, traj_size)
        r = keras.layers.Activation(norm_func, name='r')(q)
        
        # The gravitational potential is -G m0 m1 / r = - mu / r per unit mass m1 in restricted problem
        U = tf.divide(mu, r)

        # Check shapes
        tf.debugging.assert_shapes(shapes={
            r: (batch_size, traj_size),
            U: (batch_size, traj_size)
        }, message='PotentialEnergy_R2B / outputs')

        return U

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class AngularMomentum_R2B(keras.layers.Layer):
    """Compute the angular momentum from position and velocity in 2D"""
    def call(self, inputs):
        # Unpack inputs
        q, v = inputs
        
        # Shape of q and v is (batch_size, traj_size, 3,)
        batch_size, traj_size = q.shape[0:2]
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 3),
            v: (batch_size, traj_size, 3)
        }, message='AngularMomentum_R2B / inputs')

        # Compute the angular momentum
        L = tf.linalg.cross(a=q, b=v, name='L')
        
        # Check shape
        tf.debugging.assert_shapes(shapes={
            L: (batch_size, traj_size, 3)
        }, message='AngularMomentum_R2B / outputs')
        
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
class AngularMomentumError(keras.losses.Loss):
    """Specialized loss for error in angular momentum.  Mean squared relative error."""
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        rel_error = (y_pred - y_true) / y_true
        return K.mean(math_ops.square(rel_error), axis=-1)

# ********************************************************************************************************************* 
# Custom Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class Motion_R2B(keras.Model):
    """Motion for restricted two body problem generated from a position calculation model."""

    def __init__(self, position_model, **kwargs):
        super(Motion_R2B, self).__init__(**kwargs)
        self.position_model = position_model

    def call(self, inputs):
        """
        Compute full orbits for the restricted two body problem.
        Computes positions using the passed position_layer, 
        then uses automatic differentiation for velocity v and acceleration a.
        INPUTS:
            t: the times to report the orbit; shape (batch_size, traj_size)
            q0: the initial position; shape (batch_size, 3)
            v0: the initial velocity; shape (batch_size, 3)
            mu: the gratitational field strength; shape (batch_size, 1)
        OUTPUTS:
            q: the position at time t; shape (batch_size, traj_size, 3)
            v: the velocity at time t; shape (batch_size, traj_size, 3)
            a: the acceleration at time t; shape (batch_size, traj_size, 3)
        """
        # Unpack the inputs
        t, q0, v0, mu = inputs

        # Get the trajectory size and target shape of t
        traj_size = t.shape[1]
        target_shape = (traj_size, 1)

        # Reshape t to have shape (batch_size, traj_size, 1)
        t = keras.layers.Reshape(target_shape=target_shape, name='t')(t)

        # Check shapes after resizing operation; can accept t of shape EITHER 
        # (batch_size, traj_size) or (batch_size, traj_size, 1)
        batch_size = t.shape[0]
        tf.debugging.assert_shapes(shapes={
            t: (batch_size, traj_size, 1),
        }, message='Motion_R2B.call / inputs')
    
        # Evaluation of the position is under the scope of two gradient tapes
        # These are for velocity and acceleration
        with tf.GradientTape(persistent=True) as gt2:
            gt2.watch(t)

            with tf.GradientTape(persistent=True) as gt1:
                gt1.watch(t)       
        
                # Get the position using the input position layer
                position_inputs = (t, q0, v0, mu)
                # The velocity from the position model assumes the orbital elements are not changing
                # Here we only want to take the position output and do a full automatic differentiation
                qx, qy, qz, vx_, vy_, vz_ = self.position_model(position_inputs)
                q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=2, name='q')

                tf.debugging.assert_shapes(shapes={
                    qx: (batch_size, traj_size, 1),
                    qy: (batch_size, traj_size, 1),
                    qz: (batch_size, traj_size, 1),
                    vx_: (batch_size, traj_size, 1),
                    vy_: (batch_size, traj_size, 1),
                    vz_: (batch_size, traj_size, 1),
                }, message='Motion_R2B / outputs of position model')
                
            # Compute the velocity v = dq/dt with gt1
            vx = gt1.gradient(qx, t)
            vy = gt1.gradient(qy, t)
            vz = gt1.gradient(qz, t)
            v = keras.layers.concatenate(inputs=[vx, vy, vz], axis=2, name='v')
            del gt1
            
            tf.debugging.assert_shapes(shapes={
                vx: (batch_size, traj_size, 1),
                vy: (batch_size, traj_size, 1),
                vz: (batch_size, traj_size, 1),
            }, message='Motion_R2B / velocity by automatic differentation')

        # Compute the acceleration a = d2q/dt2 = dv/dt with gt2
        ax = gt2.gradient(vx, t)
        ay = gt2.gradient(vy, t)
        az = gt2.gradient(vz, t)
        a = keras.layers.concatenate(inputs=[ax, ay, az], name='a')
        del gt2
        
        # Check shapes
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 3),
            v: (batch_size, traj_size, 3),
            a: (batch_size, traj_size, 3),
        }, message='Motion_R2B.call / outputs')

        return q, v, a