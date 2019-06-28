"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Models for circular motion with neural nets

Michael S. Emanuel
Thu Jun 27 21:43:17 2019
"""


# Library imports
import tensorflow as tf
# import numpy as np

# Aliases
keras = tf.keras

# Local imports
from tf_utils import Identity
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
   
    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # Repeat r, theta0 and omega to be vectors of shape matching t
    r = keras.layers.RepeatVector(n=traj_size, name='r')(r0)
    theta0_vec = keras.layers.RepeatVector(n=traj_size, name='theta0_vec')(theta0)
    omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)

    # Check shapes
    tf.debugging.assert_shapes(shapes={
        t_vec: (batch_size, traj_size, 1),
        r: (batch_size, traj_size, 1),
        theta0_vec: (batch_size, traj_size, 1),
        omega: (batch_size, traj_size, 1)
    }, message='make_position_model_r2bc_nn / r_vec, theta0_vec, omega_vec')
    
    # The angle theta at time t
    # theta = omega * t + theta0
    omega_t = keras.layers.multiply(inputs=[omega, t_vec], name='omega_t')
    theta = keras.layers.add(inputs=[omega_t, theta0_vec], name='theta')

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
    }, message='make_position_model_r2bc_nn / outputs')
    
    # Wrap this into a model
    outputs = [qx, qy]
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_r2bc_nn')
    return model