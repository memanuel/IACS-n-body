"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Generate and plot training data (trajectories)

Michael S. Emanuel
Thu Jun 20 10:16:45 2019
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
from r2b import Motion_R2B, make_position_model_r2bc_math

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_r2b(position_layer: keras.layers.Layer, traj_size: int):
    """Create a model for the restricted two body problem"""
    # Create input layers
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(2,), name='q0')
    v0 = keras.Input(shape=(2,), name='v0')
    mu = keras.Input(shape=(1,), name='mu')
    # The combined input layers
    inputs = [t, q0, v0, mu]

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

    # Compute the motion from the specified position layer
    q, v, a = Motion_R2B(position_layer=position_layer, name='motion')([t, r0, theta0, omega0])
    
    # Name the outputs of the circular motion
    # These each have shape (batch_size, traj_size, 2)
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)

    # Compute q0_rec and v0_rec
    # These each have shape (batch_size, 2)
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)

    # Compute kinetic energy T and potential energy U
    T = KineticEnergy_R2B(name='T')(v)
    U = PotentialEnergy_R2B(name='U')([q, mu])
    # Compute the total energy H
    H = keras.layers.add(inputs=[T,U], name='H')

    # Compute angular momentum L
    # This has shape (batch_size, 1)
    L = AngularMomentum_R2B(name='L')([q, v])
    
    # Wrap this up into a model
    outputs = [q, v, a, q0_rec, v0_rec, H, L]
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_math')
    return model

# ********************************************************************************************************************* 
def make_model_r2bc_math(traj_size: int = 731):
    """Create an anlytical model for the restricted two body circular problem"""
    # Build the position model
    position_model = make_position_model_r2bc_math(traj_size=traj_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_model_r2b(position_model=position_model, traj_size=traj_size)

# ********************************************************************************************************************* 
def make_model_r2bc():
    """Create a model for the restricted two body circular problem"""
    # Create input layers
    t = keras.Input(shape=(1,), name='t')
    q0 = keras.Input(shape=(2,), name='q0')
    v0 = keras.Input(shape=(2,), name='v0')
    # The combined input layers
    inputs = [t, q0, v0]
    
    # Combine the input features for the initial configuration
    config0 = keras.layers.concatenate(inputs=[q0, v0], name='config0') 

    # 2 Dense feature layers depending ONLY on the configuration (not the time)
    phi_1 = keras.layers.Dense(units=16, activation='tanh', name='phi_1')(config0)
    phi_2 = keras.layers.Dense(units=16, activation='tanh', name='phi_2')(phi_1)
    
    # The radius r; this is the same at time 0 and t because phi_2 does not depend on t
    r = keras.layers.Dense(1, name='r')(phi_2)
  
    # The angular velocity omega
    omega = keras.layers.Dense(1, name='omega')(phi_2)
    
    # Negative of omega and omega2; used below for computing the velocity and acceleration components
    neg_omega = keras.layers.Activation(activation=tf.negative, name='neg_omega')(omega)
    neg_omega2 = keras.layers.multiply(inputs=[neg_omega, omega], name='neg_omega2')
    
    # The initial angle theta_0
    theta0 = keras.layers.Dense(1, name='theta0')(phi_2)
    
    # The angle theta at time t
    omega_t = keras.layers.multiply(inputs=[omega, t], name='omega_t')
    theta = keras.layers.add(inputs=[omega_t, theta0], name='theta')
    
    # Cosine and sine of theta
    cos_theta = keras.layers.Activation(activation=tf.cos, name='cos_theta')(theta)
    sin_theta = keras.layers.Activation(activation=tf.sin, name='sin_theta')(theta)

    # Compute qx and qy from r, theta
    qx = keras.layers.multiply(inputs=[r, cos_theta], name='qx')
    qy = keras.layers.multiply(inputs=[r, sin_theta], name='qy')
    q = keras.layers.concatenate(inputs=[qx, qy], name='q')
    
    # Compute vx and vy from r, theta
    vx = keras.layers.multiply(inputs=[neg_omega, qy], name='vx')
    vy = keras.layers.multiply(inputs=[omega, qx], name='vy')
    v = keras.layers.concatenate(inputs=[vx, vy], name='v')

    # Compute ax and ay from r, theta
    ax = keras.layers.multiply(inputs=[neg_omega2, qx], name='ax')
    ay = keras.layers.multiply(inputs=[neg_omega2, qy], name='ay')
    a = keras.layers.concatenate(inputs=[ax, ay], name='a')

    # The sine and cosine of theta0 are used for the recovered initial configuration
    cos_theta0 = keras.layers.Activation(activation=tf.cos, name='cos_theta0')(theta0)
    sin_theta0 = keras.layers.Activation(activation=tf.sin, name='sin_theta0')(theta0)

    # The recovered initial position q0_rec
    qx0_rec = keras.layers.multiply(inputs=[r, cos_theta0], name='qx0_rec')
    qy0_rec = keras.layers.multiply(inputs=[r, sin_theta0], name='qy0_rec')
    q0_rec = keras.layers.concatenate(inputs=[qx0_rec, qy0_rec], name='q0_rec')

    # The recovered initial velocity v0_rec
    vx0_rec = keras.layers.multiply(inputs=[neg_omega, qy0_rec], name='vx0_rec')
    vy0_rec = keras.layers.multiply(inputs=[omega, qx0_rec], name='vy0_rec')
    v0_rec = keras.layers.concatenate(inputs=[vx0_rec, vy0_rec], name='v0_rec')       

    # The combined output layers
    outputs = [q, v, a, q0_rec, v0_rec]
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='r2bc')
    return model

