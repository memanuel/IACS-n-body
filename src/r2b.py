"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Custom layers and reusable calculations

Michael S. Emanuel
Thu Jun 27 10:27:20 2019
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
        # Shape of v is (batch_size, traj_size, 2,)
        v = inputs

        # Element-wise velocity squared
        v2 = tf.square(v)
        
        # The KE is 1/2 m v^2 = 1/2 v^2 per unit mass in restricted problem
        T = 0.5 * tf.reduce_sum(v2, axis=-1, keepdims=False)

        # Check shapes
        batch_size, traj_size = v.shape[0:2]
        tf.debugging.assert_shapes(shapes={
            v: (batch_size, traj_size, 2),
            v2: (batch_size, traj_size, 2),
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
        q, mu = inputs

        # Shape of q is (batch_size, traj_size, 2,)
        # Shape of mu is (batch_size, 1)
        batch_size, traj_size = q.shape[0:2]
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 2),
            mu: (batch_size, 1)
        }, message='PotentialEnergy_R2B / inputs')

        # Compute the norm of a 2D vector
        norm_func = lambda q : tf.norm(q, axis=-1, keepdims=False)

        # The distance r; shape (batch_size, traj_size)
        r = keras.layers.Activation(norm_func, name='r')(q)
        
        # The gravitational potential is -G m0 m1 / r = - mu / r per unit mass m1 in restricted problem
        U = tf.negative(tf.divide(mu, r))

        # Check shapes
        tf.debugging.assert_shapes(shapes={
            r: (batch_size, traj_size),
            U: (batch_size, traj_size)
        }, message='PotentialEnergy_R2B / outputs')

        return U

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class AngularMomentum0_R2B(keras.layers.Layer):
    """Compute the initial angular momentum from initial position and velocity in 2D"""
    def call(self, inputs):
        # Unpack inputs
        q0, v0 = inputs

        # Shape of q0 and v0 is (batch_size, 2,)
        batch_size = q0.shape[0]
        tf.debugging.assert_shapes(shapes={
            q0: (batch_size, 2),
            v0: (batch_size, 2)
        }, message='AngularMomentum0_R2B / inputs')

        # Compute the angular momentum
        L0 = (q0[:,0]*v0[:,1]) - (q0[:,1]*v0[:,0])

        # Reshape from (batch_size,) to (batch_size, 1)
        L0 = keras.layers.Reshape(target_shape=(1,))(L0)
        
        # Check shape
        tf.debugging.assert_shapes(shapes={
            L0: (batch_size, 1)
        }, message='AngularMomentum0_R2B / outputs')
        
        return L0
    
# ********************************************************************************************************************* 
class AngularMomentum_R2B(keras.layers.Layer):
    """Compute the angular momentum from position and velocity in 2D"""
    def call(self, inputs):
        # Unpack inputs
        q, v = inputs
        
        # Shape of q and v is (batch_size, traj_size, 2,)
        batch_size, traj_size = q.shape[0:2]
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 2),
            v: (batch_size, traj_size, 2)
        }, message='AngularMomentum_R2B / inputs')

        # Compute the angular momentum
        L = (q[:,:,0]*v[:,:,1]) - (q[:,:,1]*v[:,:,0])
        
        # Check shape
        tf.debugging.assert_shapes(shapes={
            L: (batch_size, traj_size)
        }, message='AngularMomentum_R2B / outputs')
        
        return L
    
# ********************************************************************************************************************* 
class ConfigToPolar2D(keras.layers.Layer):
    def call(self, inputs):
        """Compute r0, thetat0 and omega0 from initial configuration (q0, v0)"""
        # Unpack inputs
        # q0 and v0 have shape (batch_size, 2,)
        q0, v0 = inputs
        
        # Wrap division operator
        div_func = lambda xx: tf.math.divide(xx[0], xx[1])
        
        # Compute the norm of a 2D vector
        norm_func = lambda q : tf.norm(q, axis=1, keepdims=True)
    
        # Compute the argument theta of a point q = (x, y)
        # theta reshaped to (batch_size, 1,)
        arg_func = lambda q : tf.reshape(tensor=tf.atan2(y=q[:,1], x=q[:,0]), shape=(-1, 1,))

        # The radius r0 at time t=0
        # r0 will have shape (batch_size, 1,)
        r0 = keras.layers.Activation(norm_func, name='r0')(q0)

        # The initial angle theta0
        # theta0 will have shape (batch_size, 1)
        theta0 = keras.layers.Lambda(arg_func, name='theta0')(q0)

        # Compute the angular momentum and square of r0
        L0 = AngularMomentum0_R2B(name='ang_mom')([q0, v0])
        r0_2 = keras.layers.Activation(tf.square, name='r0_2')(r0)

        # Compute the angular velocity omega from angular momentum and r0
        # omega will have shape (batch_size, 1,)
        omega0 = keras.layers.Lambda(div_func, name='omega0')([L0, r0_2])    
        
        # Return the three inital polar quantities
        return r0, theta0, omega0
    
    def get_config(self):
        return dict()

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
            r0: the initial distance; shape (batch_size, 1)
            theta0: the initial angle; shape (batch_size, 1)
            omega0: the angular velocity; shape (batch_size, 1)
        OUTPUTS:
            q: the position at time t; shape (batch_size, traj_size, 2)
            v: the velocity at time t; shape (batch_size, traj_size, 2)
            a: the acceleration at time t; shape (batch_size, traj_size, 2)
        """
        # Unpack time from first input; the rest are passed as-is to position_layer
        t = inputs[0]

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
                # qx, qy = self.position_layer([t, r0, theta0, omega0])
                position_inputs = [t] + inputs[1:]
                qx, qy = self.position_model(position_inputs)
                q = keras.layers.concatenate(inputs=[qx, qy], axis=2, name='q')

            # Compute the velocity v = dq/dt with gt1
            vx = gt1.gradient(qx, t)
            vy = gt1.gradient(qy, t)
            v = keras.layers.concatenate(inputs=[vx, vy], name='v')
            del gt1
            
        # Compute the acceleration a = d2q/dt2 = dv/dt with gt2
        ax = gt2.gradient(vx, t)
        ay = gt2.gradient(vy, t)
        a = keras.layers.concatenate(inputs=[ax, ay], name='a')
        del gt2
        
        # Check shapes
        batch_size = t.shape[0]
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 2),
            v: (batch_size, traj_size, 2),
            a: (batch_size, traj_size, 2),
        }, message='Motion_R2B.call / outputs')

        return q, v, a

