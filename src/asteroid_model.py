"""
Harvard IACS Masters Thesis
Solar Asteroid Model: Predict the movement of a test particle (e.g. asteroid) in the solar system
using the Kepler approximation with the sun as a fixed central attractor.

Michael S. Emanuel
Sun Oct 13 11:56:50 2019
"""

# Library imports
import tensorflow as tf
import numpy as np
# import rebound

# Local imports
# from tf_utils import Identity
from orbital_element import MeanToTrueAnomaly, TrueToMeanAnomaly
from asteroid_data import make_dataset_ast_pos, make_dataset_ast_dir, get_earth_pos, orbital_element_batch

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
# Set autograph logging verbosity
tf.autograph.set_verbosity(0)

# ********************************************************************************************************************* 
# Constants

# The gravitational constant in ('day', 'AU', 'Msun') coordinates
# sim = rebound.Simulation()
# sim.units = ('day', 'AU', 'Msun')
# G_ = sim.G
G_ = 0.00029591220828559104
mu = tf.constant(G_)
space_dims = 3

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class ElementToPosition(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ElementToPosition, self).__init__(**kwargs)

    # @tf.function
    def call(self, inputs):
        """Compute position from orbital elements (a, e, inc, Omega, omega, f)"""
        # Unpack inputs
        # a: semimajor axis
        # e: eccentricity
        # inc: inclination
        # Omega: longitude of ascending node
        # omega: argument of pericenter
        # f: true anomaly
        a, e, inc, Omega, omega, f = inputs

        # See module OrbitalElements for original version that includes velocity as well
        # This is pared down for speed
        
        # Shape of input
        shape = a.shape
        
        # sine and cosine of the angles inc, Omega, omega, and f
        ci = keras.layers.Activation(activation=tf.cos, name='cos_inc')(inc)
        si = keras.layers.Activation(activation=tf.sin, name='sin_inc')(inc)
        cO = keras.layers.Activation(activation=tf.cos, name='cos_Omega')(Omega)
        sO = keras.layers.Activation(activation=tf.sin, name='sin_Omega')(Omega)
        co = keras.layers.Activation(activation=tf.cos, name='cos_omega')(omega)
        so = keras.layers.Activation(activation=tf.sin, name='sin_omega')(omega)
        cf = keras.layers.Activation(activation=tf.cos, name='cos_f')(f)
        sf = keras.layers.Activation(activation=tf.sin, name='sin_f')(f)

        # Distance from center
        # one_minus_e2 = tf.constant(1.0) - tf.square(e)
        # one_plus_e_cos_f = tf.constant(1.0) + e * tf.cos(f)
        # r = a * one_minus_e2 / one_plus_e_cos_f
        e2 = keras.layers.Activation(activation=tf.square, name='e2')(e)
        # one_minus_e2 = tf.subtract(tf.constant(1.0), e2, name='one_minus_e2')
        one = tf.broadcast_to(1.0, shape)
        # one_minus_e2 = keras.layers.subtract([one, e2], name='one_minus_e2')
        one_minus_e2 = tf.subtract(one, e2, name='one_minus_e2')
        # e_cos_f = keras.layers.multiply([e, cf], name='e_cos_f')
        e_cos_f = tf.multiply(e, cf, name='e_cos_f')
        # one_plus_e_cos_f = keras.layers.add([one, e_cos_f], name='one_plus_e_cos_f')
        one_plus_e_cos_f = tf.add(one, e_cos_f, name='one_plus_e_cos_f')
        # a_x_one_minus_e2 = keras.layers.multiply([a, one_minus_e2], name='a_x_one_minus_e2')
        a_x_one_minus_e2 = tf.multiply(a, one_minus_e2, name='a_x_one_minus_e2')
        r = tf.divide(a_x_one_minus_e2, one_plus_e_cos_f, name='r')
        
        # Position
        # cocf = keras.layers.multiply([co,cf], name='cocf')
        cocf = tf.multiply(co ,cf, name='cocf')
        # sosf = keras.layers.multiply([so,sf], name='sosf')
        sosf = tf.multiply(so, sf, name='sosf')
        # cocf_sosf = keras.layers.subtract([cocf, sosf], name='cocf_sosf')
        cocf_sosf = tf.subtract(cocf, sosf, name='cocf_sosf')

        # socf = keras.layers.multiply([so,cf], name='socf')
        socf = tf.multiply(so, cf, name='socf')
        # cosf = keras.layers.multiply([co,sf], name='cosf')
        cosf = tf.multiply(co, sf, name='cosf')
        # socf_cosf = keras.layers.add([socf, cosf], name='socf_cosf')
        socf_cosf = tf.add(socf, cosf, name='socf_cosf')

        # cO_x_cocf_sosf = keras.layers.multiply([cO, cocf_sosf], name='cO_x_cocf_sosf')
        # sO_x_socf_cosf = keras.layers.multiply([sO, socf_cosf], name = 'sO_x_socf_cosf')
        # sO_x_socf_cosf_x_ci = keras.layers.multiply([sO_x_socf_cosf, ci], name='sO_x_socf_cosf_x_ci')       
        # sO_x_cocf_sosf = keras.layers.multiply([sO, cocf_sosf], name='sO_x_cocf_sosf')
        # cO_x_socf_cosf = keras.layers.multiply([cO, socf_cosf], name='cO_x_socf_cosf')
        # cO_x_socf_cosf_x_ci = keras.layers.multiply([cO_x_socf_cosf, ci], name='cO_x_socf_cosf_x_ci')

        cO_x_cocf_sosf = tf.multiply(cO, cocf_sosf, name='cO_x_cocf_sosf')
        sO_x_socf_cosf = tf.multiply(sO, socf_cosf, name = 'sO_x_socf_cosf')
        sO_x_socf_cosf_x_ci = tf.multiply(sO_x_socf_cosf, ci, name='sO_x_socf_cosf_x_ci')       
        sO_x_cocf_sosf = tf.multiply(sO, cocf_sosf, name='sO_x_cocf_sosf')
        cO_x_socf_cosf = tf.multiply(cO, socf_cosf, name='cO_x_socf_cosf')
        cO_x_socf_cosf_x_ci = tf.multiply(cO_x_socf_cosf, ci, name='cO_x_socf_cosf_x_ci')

        # Direction components
        # ux = keras.layers.subtract([cO_x_cocf_sosf, sO_x_socf_cosf_x_ci], name='ux')
        ux = tf.subtract(cO_x_cocf_sosf, sO_x_socf_cosf_x_ci, name='ux')
        # uy = keras.layers.add([sO_x_cocf_sosf, cO_x_socf_cosf_x_ci], name='uy')
        uy = tf.add(sO_x_cocf_sosf, cO_x_socf_cosf_x_ci, name='uy')
        # uz = keras.layers.multiply([socf_cosf, si], name='socf_cosf_x_si')
        uz = tf.multiply(socf_cosf, si, name='socf_cosf_x_si')

        # Position components
        # qx = keras.layers.multiply([r, ux], name='qx')
        # qy = keras.layers.multiply([r, uy], name='qy')
        # qz = keras.layers.multiply([r, uz], name='qz')
        qx = tf.multiply(r, ux, name='qx')
        qy = tf.multiply(r, uy, name='qy')
        qz = tf.multiply(r, uz, name='qz')

        # Assemble the position vector
        q = keras.layers.concatenate(inputs=[qx, qy, qz], axis=-1, name='q')
        return q

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class AsteroidPosition(keras.layers.Layer):
    """
    Compute orbit positions for asteroids in the solar system from the initial orbital elements with the Kepler model.
    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
    Outputs of the model are the position of the asteroid relative to the sun.    
    """
    def __init__(self, ts, batch_size: int, **kwargs):
        """
        INPUTS:
            ts: fixed tensor of time snapshots at which to simulate the position
            batch_size: the number of elements to simulate at a time, e.g. 64; not to be confused with traj_size!
        """
        super(AsteroidPosition, self).__init__(**kwargs)
        # Get trajectory size from ts
        self.traj_size = ts.shape[0]
        self.batch_size = batch_size
        # self.ts = ts

        # Reshape ts to (batch_size, traj_size, 1)
        target_shape = (-1, 1)

        # print(f'ts.shape = {ts.shape}')
        # First repeat ts batch_size times; now size is (traj_size, batch_size, 1)
        t_rep= keras.layers.RepeatVector(n=batch_size, name='ts_rep')(keras.backend.reshape(ts, target_shape))
        # print(f't_rep.shape = {t_rep.shape}')
        # Transpose axes to make shape (batch_size, traj_size, 1)
        self.t_vec = tf.transpose(t_rep, perm=(1,0,2))
        # print(f't_vec.shape = {t_vec.shape}')

    def call(self, a, e, inc, Omega, omega, f, epoch):
        """
        Simulate the orbital trajectories.  
        Snapshot times t shared by all the input elements.  
        The inputs orbital elements and reference epoch should all have size (batch_size,).
        """
        # Alias traj_size, batch_size for legibility
        traj_size = self.traj_size

        # Reshape epoch to (batch_size, traj_size, 1)
        target_shape = (-1, 1)
        epoch_vec = keras.layers.RepeatVector(n=traj_size, name='epoch_vec')(keras.backend.reshape(epoch, target_shape))
        
        # Subtract epoch from t_vec; now it is relative to the epoch
        t = keras.layers.subtract([self.t_vec, epoch_vec], name='t')        

        # Compute eccentric anomaly E from f and e
        M = TrueToMeanAnomaly(name='TrueToMeanAnomaly')([f, e])
        
        # Compute mean motion N from mu and a
        a3 = tf.math.pow(a, 3, name='a3')
        mu_over_a3 = tf.divide(mu, a3, name='mu_over_a3')
        N = tf.sqrt(mu_over_a3, name='N')

        # Reshape t to (batch_size, traj_size, 1)
        target_shape = (-1, 1)
        # ******************************************************************
        # Predict orbital elements over time
        
        # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size, 1)
        target_shape = (-1, 1)
        a_t = keras.layers.RepeatVector(n=traj_size, name='a_t')(keras.backend.reshape(a, target_shape))
        e_t = keras.layers.RepeatVector(n=traj_size, name='e_t')(keras.backend.reshape(e, target_shape))
        inc_t = keras.layers.RepeatVector(n=traj_size, name='inc_t')(keras.backend.reshape(inc, target_shape))
        Omega_t = keras.layers.RepeatVector(n=traj_size, name='Omega_t')(keras.backend.reshape(Omega, target_shape))
        omega_t = keras.layers.RepeatVector(n=traj_size, name='omega_t')(keras.backend.reshape(omega, target_shape))
        
        # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
        M0_t = keras.layers.RepeatVector(n=traj_size, name='M0_t')(keras.backend.reshape(M, target_shape))
        N0_t = keras.layers.RepeatVector(n=traj_size, name='N0_t')(keras.backend.reshape(N, target_shape))
        # Compute the mean anomaly M(t) as a function of time
        N_mult_t = keras.layers.multiply(inputs=[N0_t, t])
        M_t = keras.layers.add(inputs=[M0_t, N_mult_t])
    
        # Compute the true anomaly from the mean anomly and eccentricity
        f_t = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M_t, e_t])
    
        # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
        elt_t = (a_t, e_t, inc_t, Omega_t, omega_t, f_t,)
        
        # Convert orbital elements to cartesian coordinates 
        q = ElementToPosition(name='q')(elt_t)
    
        return q

# ********************************************************************************************************************* 
class DirectionUnitVector(keras.layers.Layer):
    """
    Layer to compute the direction from object 1 (e.g. earth) to object 2 (e.g. asteroid)
    """
    
    def __init__(self, **kwargs):
        super(DirectionUnitVector, self).__init__(**kwargs)

    # don't declare this tf.function because it breaks when using it with q_earth
    # still not entirely sure how tf.function works ...
    # @tf.function
    def call(self, q1, q2):
        # Relative displacement from earth to asteroid
        q_rel = tf.subtract(q2, q1, name='q_rel')
        # Distance between objects
        r = tf.norm(q_rel, axis=-1, keepdims=True, name='r')
        # Unit vector pointing from object 1 to object 2
        u = tf.divide(q_rel, r, name='q_rel_over_r')
        return u
    
    def get_config(self):
        return dict()       


## ********************************************************************************************************************* 
class AsteroidDirection(keras.layers.Layer):
    """
    Layer to compute the direction from earth to asteroid.
    """
    def __init__(self, ts, batch_size: int, **kwargs):
        """
        INPUTS:
            ts: fixed tensor of time snapshots at which to simulate the position
            batch_size: the number of elements to simulate at a time, e.g. 64; not to be confused with traj_size!
        """
        super(AsteroidDirection, self).__init__(**kwargs)
        
        # Build layer to compute positions
        self.q_layer = AsteroidPosition(ts=ts, batch_size=batch_size, name='q_ast')
        
        # Take a one time snapshot of the earth's position at these times
        q_earth_np = get_earth_pos(ts)
        traj_size = ts.shape[0]
        q_earth_np = q_earth_np.reshape(1, traj_size, space_dims)
        self.q_earth = keras.backend.constant(q_earth_np, dtype=tf.float32, shape=q_earth_np.shape, name='q_earth')
        # print(f'q_earth.shape = {self.q_earth.shape}')

    def call(self, a, e, inc, Omega, omega, f, epoch):
        # Calculate position
        q_ast = self.q_layer(a, e, inc, Omega, omega, f, epoch)

        # Unit displacement vector (direction) from earth to asteroid
        u = DirectionUnitVector(name='u')(self.q_earth, q_ast)
        
        return u

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_ast_pos(ts: tf.Tensor, batch_size:int =64) -> keras.Model:
    """
    Compute orbit positions for asteroids in the solar system from
    the initial orbital elements with the Kepler model.
    Factory function that returns a functional model.
    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
    Outputs of the model are the position of the asteroid relative to the sun.
    INPUTS;
        ts: times to evaluate asteroid position in heliocentric coordinates
        batch_size: defaults to None for variable batch size
    """
    # Inputs: 6 orbital elements; epoch;
    a = keras.Input(shape=(), batch_size=batch_size, name='a')
    e = keras.Input(shape=(), batch_size=batch_size, name='e')
    inc = keras.Input(shape=(), batch_size=batch_size, name='inc')
    Omega = keras.Input(shape=(), batch_size=batch_size, name='Omega')
    omega = keras.Input(shape=(), batch_size=batch_size, name='omega')
    f = keras.Input(shape=(), batch_size=batch_size, name='f')
    epoch = keras.Input(shape=(), batch_size=batch_size, name='epoch')

    # Wrap these up into one tuple of inputs for the model
    inputs = (a, e, inc, Omega, omega, f, epoch)
    
    # Output times are a constant
    ts = keras.backend.constant(ts, name='ts')

    # Call asteroid position layer
    q = AsteroidPosition(ts, batch_size, name='q')(a, e, inc, Omega, omega, f, epoch)
    
    # Wrap up the outputs
    outputs = (q,)

    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_asteroid_pos')
    return model


## ********************************************************************************************************************* 
#def make_model_ast_dir(ts: tf.Tensor, batch_size:int =64) -> keras.Model:
#    """
#    Compute direction from earth to asteroids in the solar system from
#    the initial orbital elements with the Kepler model.
#    Factory function that returns a functional model.
#    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
#    Outputs of the model are the unit vector (direction) pointing from earth to the asteroid
#    INPUTS;
#        ts: times to evaluate asteroid direction from earth
#        batch_size: defaults to None for variable batch size
#    """
#    # Get trajectory size from ts
#    traj_size: int = ts.shape[0]
#    
#    # Inputs: 6 orbital elements; epoch; ts (output times as MJD)
#    a = keras.Input(shape=(), batch_size=batch_size, name='a')
#    e = keras.Input(shape=(), batch_size=batch_size, name='e')
#    inc = keras.Input(shape=(), batch_size=batch_size, name='inc')
#    Omega = keras.Input(shape=(), batch_size=batch_size, name='Omega')
#    omega = keras.Input(shape=(), batch_size=batch_size, name='omega')
#    f = keras.Input(shape=(), batch_size=batch_size, name='f')
#    epoch = keras.Input(shape=(), batch_size=batch_size, name='epoch')
#
#    # Wrap these up into one tuple of inputs for the model
#    inputs = (a, e, inc, Omega, omega, f, epoch)
#    
#    # Output times are a constant
#    ts = keras.backend.constant(ts, name='ts')
#
#    # Call asteroid position layer
#    q = AsteroidPosition(ts, batch_size, name='q')(a, e, inc, Omega, omega, f, epoch)
#
#    # Take a one time snapshot of the earth's position at these times
#    q_earth_np = get_earth_pos(ts)
#    # print(f'q_earth_np loaded, shape = {q_earth_np.shape}')
#    q_earth_np = q_earth_np.reshape(1, traj_size, space_dims)
#    q_earth = keras.backend.constant(q_earth_np, 
#                                     dtype=tf.float32,
#                                     shape=q_earth_np.shape,
#                                     name='q_earth')
#    # print(f'q_earth keras.constant created, shape = {q_earth.shape}')
#
#    # Unit displacement vector (direction) from earth to asteroid
#    u = DirectionUnitVector(name='dir_earth_ast')(q_earth, q)
#
#    # Name the outputs
#    u = Identity(name='u')(u)
#
#    # Wrap the outputs
#    outputs = (u,)
#    
#    # Wrap this into a model
#    model = keras.Model(inputs=inputs, outputs=outputs, name='model_asteroid_dir')
#    return model
#
# ********************************************************************************************************************* 
def make_model_ast_dir(ts: tf.Tensor, batch_size:int =64) -> keras.Model:
    """
    Compute direction from earth to asteroids in the solar system from
    the initial orbital elements with the Kepler model.
    Factory function that returns a functional model.
    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
    Outputs of the model are the unit vector (direction) pointing from earth to the asteroid
    INPUTS;
        ts: times to evaluate asteroid direction from earth
        batch_size: defaults to None for variable batch size
    """
    # Inputs: 6 orbital elements; epoch; ts (output times as MJD)
    a = keras.Input(shape=(), batch_size=batch_size, name='a')
    e = keras.Input(shape=(), batch_size=batch_size, name='e')
    inc = keras.Input(shape=(), batch_size=batch_size, name='inc')
    Omega = keras.Input(shape=(), batch_size=batch_size, name='Omega')
    omega = keras.Input(shape=(), batch_size=batch_size, name='omega')
    f = keras.Input(shape=(), batch_size=batch_size, name='f')
    epoch = keras.Input(shape=(), batch_size=batch_size, name='epoch')

    # Wrap these up into one tuple of inputs for the model
    inputs = (a, e, inc, Omega, omega, f, epoch)
    
    # Output times are a constant
    ts = keras.backend.constant(ts, name='ts')

    # All the work done in a single layer
    u = AsteroidDirection(ts, batch_size, name='u')(a, e, inc, Omega, omega, f, epoch)
    
    # Wrap the outputs
    outputs = (u,)
    
    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_asteroid_dir')
    return model

# ********************************************************************************************************************* 
def test_ast_pos_layer():
    """Test custom layer for asteroid positions"""
    ast_pos_layer = AsteroidPosition(batch_size=64)
    ts = np.arange(51544, 58744, dtype=np.float32)
    a,e,inc,Omega,omega,f, epoch = orbital_element_batch(1).values()
    q_ast = ast_pos_layer(ts,a,e,inc,Omega,omega,f,epoch)
    return q_ast

# ********************************************************************************************************************* 
def test_ast_pos() -> bool:
    """Test asteroid position model"""
    # Load data for the first 1000 asteroids
    ds: tf.data.Dataset = make_dataset_ast_pos(0, 1)
    # Get reference times
    batch_in, batch_out = list(ds.take(1))[0]
    ts = batch_in['ts'][0]
    # Create the model to predict asteroid trajectories
    model: keras.Model = make_model_ast_pos(ts=ts)
    # Compile with MSE (mean squared error) loss
    model.compile(loss='MSE')
    # Evaluate this model
    mse: float = model.evaluate(ds)
    rmse: float = np.sqrt(mse)
    # Threshold for passing
    thresh: float = 0.125
    isOK: bool = (rmse < thresh)
    # Report results
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'Root MSE for asteroid model on first 1000 asteroids = {rmse:8.6f}')
    print(f'***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
def test_ast_dir() -> bool:
    """Test the asteroid direction model"""
    # Load data for the first 1000 asteroids
    ds: tf.data.Dataset = make_dataset_ast_dir(0, 1)
    # Get reference times
    batch_in, batch_out = list(ds.take(1))[0]
    ts = batch_in['ts'][0]
    # Create the model to predict asteroid trajectories
    model: keras.Model = make_model_ast_dir(ts=ts)
    # Compile with MSE (mean squared error) loss
    model.compile(loss='MSE')
    # Evaluate this model
    mse: float = model.evaluate(ds)
    rmse: float = np.sqrt(mse)
    # Convert error from unit vector to angle
    rmse_rad = 2.0 * np.arcsin(rmse / 2.0)
    rmse_deg = np.rad2deg(rmse_rad)
    rmse_sec = rmse_deg * 3600
    # Threshold for passing
    thresh: float = 2.5
    isOK: bool = (rmse_deg < thresh)
    
    # Report results
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'MSE for asteroid model on first 1000 asteroids = {mse:8.6f}')
    print(f'Angle error = {rmse_rad:5.3e} rad / {rmse_deg:8.6f} degrees / {rmse_sec:6.2f} arc seconds')
    print(f'***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
def main():
    test_ast_pos()
    test_ast_dir()
    
# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
