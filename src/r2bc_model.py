"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Generate and plot training data (trajectories)

Michael S. Emanuel
Thu Jun 20 10:16:45 2019
"""

# Library imports
import tensorflow as tf
import numpy as np

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class AngularMomentum2D(keras.layers.Layer):
    """Compute the angular momentum from initial position and velocity in 2D"""
    def __init__(self, **kwargs):
        super(AngularMomentum2D, self).__init__(**kwargs)
        
    def call(self, inputs):
        # Unpack inputs
        # Expected shape of q0 and v0 is (batch_size,)
        q0, v0 = inputs[0], inputs[1]
        
        # Compute the angular momentum
        L0z = (q0[:,0]*v0[:,1]) - (q0[:,1]*v0[:,0])

        # Reshape from (batch_size,) to (batch_size, 1)
        return keras.layers.Reshape(target_shape=(1,))(L0z)
    
# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_r2bc_math(traj_size=731):
    """Create an anlytical model for the restricted two body circular problem"""
    # Create input layers
    # t = keras.Input(shape=(None,), name='t')
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(2,), name='q0')
    v0 = keras.Input(shape=(2,), name='v0')
    # The combined input layers
    inputs = [t, q0, v0]

    # Get the trajectory size; default to 731 (2 years) so TF doesn't get upset at compile time
    traj_size = t.shape[1] or traj_size

    # One-liners to add and multiply two vectors
    # These are convenient because they use broadcoasting
    # Wrapping these lambda functions in a layers.Lambda layer gives them nice names and makes them easier to save
    div_func = lambda xx: tf.math.divide(xx[0], xx[1])

    # Compute the norm of a 2D vector
    norm_func = lambda q : tf.norm(q, axis=1, keepdims=True)
    
    # Return row 0 of a position or velocity for q0_rec and v0_rec
    initial_row_func = lambda q : q[:, 0, :]

    # Shape of outputs is (batch_size, traj_size, 2); each component has 1 in last place
    target_shape = (traj_size, 1)

    # Reshape t to have shape (traj_size, 1)
    t = keras.layers.Reshape(target_shape=target_shape, name='t_3d')(t)
    
    # The radius r0 at time t=0
    r0 = keras.layers.Activation(norm_func, name='r0')(q0)

    # Repeat r to be a vector of shape matching t
    r = keras.layers.RepeatVector(n=traj_size, name='r')(r0)

    # mu = tf.constant((2.0*np.pi)**2, name='mu')

    # Compute the angular momentum and square of r0
    L0z = AngularMomentum2D(name='ang_mom')([q0, v0])
    r0_2 = keras.layers.Activation(tf.square, name='r0_2')(r0)

    # Compute the angular velocity omega from angular momentum and r0
    omega0 = keras.layers.Lambda(div_func, name='omega0')([L0z, r0_2])    
    # Repeat omega to be a vector of shape matching t
    omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)

    # Negative of omega and omega2; used below for computing the velocity and acceleration components
    neg_omega = keras.layers.Activation(activation=tf.negative, name='neg_omega')(omega)
    neg_omega2 = keras.layers.multiply(inputs=[neg_omega, omega], name='neg_omega2')

    # The initial angle theta0
    atan_func = lambda q : tf.reshape(tensor=tf.atan2(y=q[:,1], x=q[:,0]), shape=(-1,1,1))
    # theta0_scalar = keras.layers.Lambda(atan_func, name='theta0_scalar')(q0)
    theta0 = keras.layers.Lambda(atan_func, name='theta0')(q0)

    # The angle theta at time t
    # theta = omega * t + theta0
    omega_t = keras.layers.multiply(inputs=[omega, t], name='omega_t')
    theta = keras.layers.add(inputs=[omega_t, theta0], name='theta')
    # theta = keras.layers.Lambda(add_func, name='theta')([omega_t, theta0])

    # Cosine and sine of theta
    cos_theta = keras.layers.Activation(activation=tf.cos, name='cos_theta')(theta)
    sin_theta = keras.layers.Activation(activation=tf.sin, name='sin_theta')(theta)

    # Compute qx and qy from r, theta
    qx = keras.layers.multiply(inputs=[r, cos_theta], name='qx')
    qy = keras.layers.multiply(inputs=[r, sin_theta], name='qy')
    q = keras.layers.concatenate(inputs=[qx, qy], axis=2, name='q')
   
    # Compute vx and vy from r, theta
    vx = keras.layers.multiply(inputs=[neg_omega, qy], name='vx')
    vy = keras.layers.multiply(inputs=[omega, qx], name='vy')
    v = keras.layers.concatenate(inputs=[vx, vy], name='v')

    # Compute ax and ay from r, theta
    ax = keras.layers.multiply(inputs=[neg_omega2, qx], name='ax')
    ay = keras.layers.multiply(inputs=[neg_omega2, qy], name='ay')
    a = keras.layers.concatenate(inputs=[ax, ay], name='a')
    
    # Compute q0_rec and v0_rec
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)
    
    model = keras.Model(inputs=inputs, outputs=[q, v, a, q0_rec, v0_rec], name='model_math')
    return model

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

