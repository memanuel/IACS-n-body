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
        
        # Check sizes
        batch_size = t.shape[0]
        tf.debugging.assert_shapes(shapes={
            q: (batch_size, traj_size, 2),
            v: (batch_size, traj_size, 2),
            a: (batch_size, traj_size, 2),
        }, message='Motion_R2B.call / outputs')

        return q, v, a
    
# ********************************************************************************************************************* 
def make_position_model_r2bc_math(traj_size = 731):
    """
    Compute orbit positions for the restricted two body circular problem from 
    the initial polar coordinates (orbital elements) with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    # Create input layers
    t = keras.Input(shape=(traj_size,1), name='t')
    r0 = keras.Input(shape=(1,), name='r0')
    theta0 = keras.Input(shape=(1,), name='theta0')
    omega0 = keras.Input(shape=(1,), name='omega0')
    # The combined input layers
    inputs = [t, r0, theta0, omega0]
    
    # Repeat r, theta0 and omega to be vectors of shape matching t
    r = keras.layers.RepeatVector(n=traj_size, name='r')(r0)
    theta0 = keras.layers.RepeatVector(n=traj_size, name='theta0_vec')(theta0)
    omega = keras.layers.RepeatVector(n=traj_size, name='omega_vec')(omega0)

    # Check shapes
    batch_size = t.shape[0]
    tf.debugging.assert_shapes(shapes={
        r: (batch_size, traj_size, 1),
        theta0: (batch_size, traj_size, 1),
        omega: (batch_size, traj_size, 1)
    }, message='make_position_model_r2bc_math / inputs')
    
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
    
    # Check shapes
    batch_size = t.shape[0]
    tf.debugging.assert_shapes(shapes={
        omega_t: (batch_size, traj_size, 1),
        theta: (batch_size, traj_size, 1),
        cos_theta: (batch_size, traj_size, 1),
        sin_theta: (batch_size, traj_size, 1),
        qx: (batch_size, traj_size, 1),
        qy: (batch_size, traj_size, 1),
    }, message='make_position_model_r2bc_math / outputs')
    
    # Wrap this into a model
    outputs = [qx, qy]
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_r2bc_math')
    return model

# ********************************************************************************************************************* 
# OLD
class Position_Layer_R2BC_Math(keras.layers.Layer):
    """
    Compute orbit positions for the restricted two body circular problem from 
    the initial polar coordinates (orbital elements) with a deterministic mathematical model.
    """

    def call(self, inputs):
        """
        INPUTS:
            inputs: a list of the tensors requred.  inputs = [t, r0, theta0, omega0], where
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

# ********************************************************************************************************************* 
# OLD
class Position_Model_R2BC_Math(keras.Model):
    """
    Compute orbit positions for the restricted two body circular problem from 
    the initial polar coordinates (orbital elements) with a deterministic mathematical model.
    """

    def call(self, inputs):
        """
        INPUTS:
            inputs: a list of the tensors requred.  inputs = [t, r0, theta0, omega0], where
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