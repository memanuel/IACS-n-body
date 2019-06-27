"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Custom layers and reusable calculations

Michael S. Emanuel
Thu Jun 27 10:27:20 2019
"""

# Library imports
import tensorflow as tf
# import numpy as np

# Aliases
keras = tf.keras

# Local imports

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class AngularMomentum0_R2B(keras.layers.Layer):
    """Compute the initial angular momentum from initial position and velocity in 2D"""
    def call(self, inputs):
        # Unpack inputs
        # Shape of q0 and v0 is (batch_size, 2,)
        q0, v0 = inputs
        
        # Compute the angular momentum
        L0 = (q0[:,0]*v0[:,1]) - (q0[:,1]*v0[:,0])

        # Reshape from (batch_size,) to (batch_size, 1)
        return keras.layers.Reshape(target_shape=(1,))(L0)
    
# ********************************************************************************************************************* 
class AngularMomentum_R2B(keras.layers.Layer):
    """Compute the angular momentum from position and velocity in 2D"""
    def call(self, inputs):
        # Unpack inputs
        # Shape of q and v is (batch_size, traj_size, 2,)
        q, v = inputs
        
        # Compute the angular momentum
        L = (q[:,:,0]*v[:,:,1]) - (q[:,:,1]*v[:,:,0])
        return L
    
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
        return T
    
# ********************************************************************************************************************* 
class PotentialEnergy_R2B(keras.layers.Layer):
    """Compute the potential energy from position q and gravitational constant mu"""
    def call(self, inputs):
        # Unpack inputs
        # Shape of q is (batch_size, traj_size, 2,)
        # Shape of mu is (batch_size, 1)
        q, mu = inputs
        
        # Compute the norm of a 2D vector
        norm_func = lambda q : tf.norm(q, axis=-1, keepdims=False)

        # The distance r
        r = keras.layers.Activation(norm_func, name='r')(q)
        
        # The gravitational potential is -G m0 m1 / r = - mu / r per unit mass m1 in restricted problem
        U = tf.negative(tf.divide(mu, r))
        return U
    
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
    
# ********************************************************************************************************************* 
class Motion_R2B(keras.layers.Layer):
    """Motion for restricted two body problem generated from a position calculation layer."""

    def __init__(self, position_layer, **kwargs):
        super(Motion_R2B, self).__init__(**kwargs)
        self.position_layer = position_layer

    def call(self, inputs):
        """
        Compute full orbits for the restricted two body problem.
        Computes positions using the passed position_layer, 
        then uses automatic differentiation for velocity v and acceleration a.
        INPUTS:
            t: the times to report the orbit; shape (batch_size, 1,)
            r0: the initial distance; shape (batch_size, 1,)
            theta0: the initial angle; shape (batch_size, 1,)
            omega0: the angular velocity; shape (batch_size, 1,)
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

        # Evaluation of the position is under the scope of two gradient tapes
        # These are for velocity and acceleration
        with tf.GradientTape(persistent=True) as gt2:
            gt2.watch(t)       
        
            with tf.GradientTape(persistent=True) as gt1:
                gt1.watch(t)       
        
                # Get the position using the input position layer
                # qx, qy = self.position_layer([t, r0, theta0, omega0])
                position_inputs = [t] + inputs[1:]
                qx, qy = self.position_layer(position_inputs)
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
            
        return q, v, a
    
# ********************************************************************************************************************* 
class Position_R2BC_Math(keras.layers.Layer):
    def call(self, inputs):
        """
        Compute orbit positions for the restricted two body circular problem from 
        the initial polar coordinates (orbital elements) with a deterministic mathematical model.
        INPUTS:
            t: the times to report the orbit; shape (batch_size, traj_size, 1,)
            r0: the initial distance; shape (batch_size, 1,)
            theta0: the initial angle; shape (batch_size, 1,)
            omega0: the angular velocity; shape (batch_size, 1,)
        OUTPUTS:
            qx: the x position at time t; shape (batch_size, traj_size, 1)
            qy: the y position at time t; shape (batch_size, traj_size, 1)
        """
        # Unpack inputs
        t, r0, theta0, omega0 = inputs

        # Get the trajectory size
        traj_size = t.shape[1]
        
        # Repeat r, theta0 and omega to be vectors of shape matching t
        r = keras.layers.RepeatVector(n=traj_size, name='r')(r0)
        theta0 = keras.layers.RepeatVector(n=traj_size, name='theta0')(theta0)
        omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)

        # The angle theta at time t
        # theta = omega * t + theta0
        omega_t = keras.layers.multiply(inputs=[omega, t], name='omega_t')
        theta = keras.layers.add(inputs=[omega_t, theta0], name='theta')

        # Cosine and sine of theta
        cos_theta = keras.layers.Activation(activation=tf.cos, name='cos_theta')(theta)
        sin_theta = keras.layers.Activation(activation=tf.sin, name='sin_theta')(theta)

        # Compute qx and qy from r, theta
        qx = keras.layers.multiply(inputs=[r, cos_theta], name='qx')
        qy = keras.layers.multiply(inputs=[r, sin_theta], name='qy')
           
        return qx, qy
    
