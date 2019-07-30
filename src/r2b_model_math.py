"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Models for restricted two body problem using math (closed form)

Michael S. Emanuel
Tue Jul 30 14:52:00 2019
"""

# Library imports
import tensorflow as tf
# import numpy as np

# Aliases
keras = tf.keras

# Local imports
from tf_utils import Identity
from orbital_element import OrbitalElementToConfig
from r2b import KineticEnergy_R2B, PotentialEnergy_R2B, AngularMomentum_R2B
from r2b import Motion_R2B

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_r2b_math(traj_size = 731):
    """
    Compute orbit positions for the restricted two body problem from 
    the initial orbital elements with a deterministic mathematical model.
    Factory function that returns a functional model.
    """
    # Create input layers
    t = keras.Input(shape=(traj_size), name='t')
    a0 = keras.Input(shape=(1,), name='a0')
    e0 = keras.Input(shape=(1,), name='e0')
    inc0 = keras.Input(shape=(1,), name='inc0')
    Omega0 = keras.Input(shape=(1,), name='Omega0')
    omega0 = keras.Input(shape=(1,), name='omega0')
    f0 = keras.Input(shape=(1,), name='f0')
    mu0 = keras.Input(shape=(1,), name='mu0')
    
    # The combined input layers
    inputs = [t, a0, e0, inc0, Omega0, omega0, f0, mu0]

    # Get batch size explicitly
    batch_size = t.shape[0]

    # Reshape inputs if necessary
    a0 = tf.reshape(a0, (-1, 1,))
    e0 = tf.reshape(e0, (-1, 1,))
    inc0 = tf.reshape(inc0, (-1, 1,))
    Omega0 = tf.reshape(Omega0, (-1, 1,))
    omega0 = tf.reshape(omega0, (-1, 1,))
    f0 = tf.reshape(f0, (-1, 1,))
    mu0 = tf.reshape(mu0, (-1, 1,))
    
    # Check inputs shapes
    tf.debugging.assert_shapes(shapes={
        a0: (batch_size, 1),
        e0: (batch_size, 1),
        inc0: (batch_size, 1),
        Omega0: (batch_size, 1),
        omega0: (batch_size, 1),
        mu0: (batch_size, 1),
    }, message='make_position_model_r2b_math / inputs')
    
    # Reshape t to (batch_size, traj_size, 1)
    # t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)
    
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a = keras.layers.RepeatVector(n=traj_size, name='a')(a0)
    e = keras.layers.RepeatVector(n=traj_size, name='e')(e0)
    inc = keras.layers.RepeatVector(n=traj_size, name='inc')(inc0)
    Omega = keras.layers.RepeatVector(n=traj_size, name='Omega')(Omega0)
    omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)
    mu = keras.layers.RepeatVector(n=traj_size, name='mu')(mu0)

    # Check shapes
    tf.debugging.assert_shapes(shapes={
        a: (batch_size, traj_size, 1),
        e: (batch_size, traj_size, 1),
        inc: (batch_size, traj_size, 1),
        Omega: (batch_size, traj_size, 1),
        omega: (batch_size, traj_size, 1),
        mu: (batch_size, traj_size, 1),
    }, message='make_position_model_r2b_math / calcs')

    # Throwaway - duplicate the initial true anomaly
    # This is wrong, but the code should run
    f = keras.layers.RepeatVector(n=traj_size, name='f')(f0)
    
    tf.debugging.assert_shapes(shapes={
        f: (batch_size, traj_size, 1),
    }, message='make_position_model_r2b_math / f')
    
    # Wrap these up into one tuple of inputs
    inputs_elt = (a, e, inc, Omega, omega, f, mu,)
    
    # Convert from orbital elements to cartesian
    qx, qy, qz, vx, vy, vz = OrbitalElementToConfig(name='orbital_element_to_config')(inputs_elt)
    
    # Wrap up the outputs
    outputs = (qx, qy, qz)
    
    # Check shapes
    tf.debugging.assert_shapes(shapes={
        qx: (batch_size, traj_size, 1),
        qy: (batch_size, traj_size, 1),
        qz: (batch_size, traj_size, 1),
    }, message='make_position_model_r2b_math / outputs')
    
    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_r2b_math')
    return model

# ********************************************************************************************************************* 
def make_physics_model_r2bc_math(position_model: keras.Model, traj_size: int):
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
    }, message='make_physics_model_r2bc_math / inputs')
        
    # Return row 0 of a position or velocity for q0_rec and v0_rec
    initial_row_func = lambda q : q[:, 0, :]

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
    }, message='make_physics_model_r2bc_math / polar elements r0, theta0, omega0')
        
    # Compute the motion from the specified position layer
    q, v, a = Motion_R2BC(position_model=position_model, name='motion')([t, r0, theta0, omega0])
    
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
    }, message='make_physics_model_r2bc_math / outputs q, v, a')
        
    # Compute q0_rec and v0_rec
    # These each have shape (batch_size, 2)
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)

    # Check sizes
    tf.debugging.assert_shapes(shapes={
        q0_rec: (batch_size, 2),
        v0_rec: (batch_size, 2),
    }, message='make_physics_model_r2bc_math / outputs q0_rec, v0_rec')

    # Compute kinetic energy T and potential energy U
    T = KineticEnergy_R2BC(name='T')(v)
    U = PotentialEnergy_R2BC(name='U')([q, mu])

    # Compute the total energy H
    H = keras.layers.add(inputs=[T,U], name='H')

    # Compute angular momentum L
    # This has shape (batch_size, traj_size)
    L = AngularMomentum_R2BC(name='L')([q, v])
    
    # Check sizes
    tf.debugging.assert_shapes(shapes={
        T: (batch_size, traj_size),
        U: (batch_size, traj_size),
        H: (batch_size, traj_size),
        L: (batch_size, traj_size),
    }, message='make_physics_model_r2bc_math / outputs H, L')

    # Wrap this up into a model
    outputs = [q, v, a, q0_rec, v0_rec, H, L]
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_math')
    return model

# ********************************************************************************************************************* 
def make_model_r2bc_math(traj_size: int = 731):
    """Create a math model for the restricted two body circular problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_r2bc_math(traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_r2bc_math(position_model=position_model, traj_size=traj_size)

