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

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class KineticEnergy(keras.layers.Layer):
    """Compute the kinetic energy from velocity"""
    def call(self, inputs):
        # Alias inputs to v
        # Shape of v is (batch_size, traj_size, 2,)
        v = inputs
        
        # Compute the kinetic energy
        T = 0.5 * tf.norm(v, axis=-1, keepdims=True)

        # Reshape from (batch_size,) to (batch_size, 1)
        return T

# ********************************************************************************************************************* 
class AngularMomentum2D(keras.layers.Layer):
    """Compute the angular momentum from initial position and velocity in 2D"""
    def call(self, inputs):
        # Unpack inputs
        # Shape of q0 and v0 is (batch_size, 2,)
        # q0, v0 = inputs[0], inputs[1]
        q0, v0 = inputs
        
        # Compute the angular momentum
        L0 = (q0[:,0]*v0[:,1]) - (q0[:,1]*v0[:,0])

        # Reshape from (batch_size,) to (batch_size, 1)
        return keras.layers.Reshape(target_shape=(1,))(L0)

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
        L0 = AngularMomentum2D(name='ang_mom')([q0, v0])
        r0_2 = keras.layers.Activation(tf.square, name='r0_2')(r0)

        # Compute the angular velocity omega from angular momentum and r0
        # omega will have shape (batch_size, 1,)
        omega0 = keras.layers.Lambda(div_func, name='omega0')([L0, r0_2])    
        
        # Return the three inital polar quantities
        return r0, theta0, omega0
    
# ********************************************************************************************************************* 
class MotionR2BC(keras.layers.Layer):
    def call(self, inputs):
        """
        Compute orbits for the restricted two body circular problem from 
        the initial polar coordinates (orbital elements)
        INPUTS:
            t: the times to report the orbit; shape (batch_size, traj_size, 1,)
            r0: the initial distance; shape (batch_size, 1,)
            theta0: the initial angle; shape (batch_size, 1,)
            omega0: the angular velocity; shape (batch_size, 1,)
        OUTPUTS:
            q: the position at time t; shape (batch_size, traj_size, 2)
            v: the velocity at time t; shape (batch_size, traj_size, 2)
            a: the acceleration at time t; shape (batch_size, traj_size, 2)
        """
        # Unpack inputs
        t, r0, theta0, omega0 = inputs

        # Evaluation of the position is under the scope of two gradient tapes
        # These are for velocity and acceleration
        with tf.GradientTape(persistent=True) as gt2:
            gt2.watch(t)       
        
            with tf.GradientTape(persistent=True) as gt1:
                gt1.watch(t)       
        
                # Get the trajectory size; default to 731 (2 years) so TF doesn't get upset at compile time
                traj_size = t.shape[1]

                # Shape of outputs is (batch_size, traj_size, 2); each component has 1 in last place
                target_shape = (traj_size, 1)

                # Reshape t to have shape (traj_size, 1)
                t_3d = keras.layers.Reshape(target_shape=target_shape, name='t_3d')(t)

                # Repeat r, theta0 and omega to be vectors of shape matching t
                r = keras.layers.RepeatVector(n=traj_size, name='r')(r0)
                theta0 = keras.layers.RepeatVector(n=traj_size, name='theta0')(theta0)
                omega = keras.layers.RepeatVector(n=traj_size, name='omega')(omega0)

                # The angle theta at time t
                # theta = omega * t + theta0
                omega_t = keras.layers.multiply(inputs=[omega, t_3d], name='omega_t')
                theta = keras.layers.add(inputs=[omega_t, theta0], name='theta')

                # Cosine and sine of theta
                cos_theta = keras.layers.Activation(activation=tf.cos, name='cos_theta')(theta)
                sin_theta = keras.layers.Activation(activation=tf.sin, name='sin_theta')(theta)

                # Compute qx and qy from r, theta
                qx = keras.layers.multiply(inputs=[r, cos_theta], name='qx')
                qy = keras.layers.multiply(inputs=[r, sin_theta], name='qy')
                q = keras.layers.concatenate(inputs=[qx, qy], axis=2, name='q')

            # Compute the velocity v = dq/dt with gt1
            vx = gt1.gradient(qx, t_3d)
            vy = gt1.gradient(qy, t_3d)
            v = keras.layers.concatenate(inputs=[vx, vy], name='v')
            del gt1
            
        # Compute the acceleration a = d2q/dt2 = dv/dt with gt2
        ax = gt2.gradient(vx, t_3d)
        ay = gt2.gradient(vy, t_3d)
        a = keras.layers.concatenate(inputs=[ax, ay], name='a')
        del gt2
            
        return q, v, a
    
# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_r2bc_math(traj_size=731):
    """Create an anlytical model for the restricted two body circular problem"""
    # Create input layers
    t = keras.Input(shape=(traj_size,), name='t')
    q0 = keras.Input(shape=(2,), name='q0')
    v0 = keras.Input(shape=(2,), name='v0')
    # The combined input layers
    inputs = [t, q0, v0]

    # Return row 0 of a position or velocity for q0_rec and v0_rec
    initial_row_func = lambda q : q[:, 0, :]

    # The polar coordinates of the initial conditions
    # r0, theta0, and omega0 each scalars in each batch
    r0, theta0, omega0 = ConfigToPolar2D(name='initial_polar')([q0, v0])
    
    # Name the outputs of the initial polar
    r0 = Identity(name='r0')(r0)
    theta0 = Identity(name='theta0')(theta0)
    omega0 = Identity(name='omega0')(omega0)

    # Compute the circular motion
    q, v, a = MotionR2BC(name='motion')([t, r0, theta0, omega0])
    
    # Name the outputs of the circular motion
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)

    # Compute q0_rec and v0_rec
    q0_rec = keras.layers.Lambda(initial_row_func, name='q0_rec')(q)
    v0_rec = keras.layers.Lambda(initial_row_func, name='v0_rec')(v)

    # Wrap this up into a model
    outputs = [q, v, a, q0_rec, v0_rec]
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_math')
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

