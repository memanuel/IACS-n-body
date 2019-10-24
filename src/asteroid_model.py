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
from tf_utils import Identity
from orbital_element import make_model_elt_to_pos, MeanToTrueAnomaly, TrueToMeanAnomaly
from asteroid_data import make_dataset_ast_pos, make_dataset_ast_dir, get_earth_pos, orbital_element_batch

# Aliases
keras = tf.keras

# The gravitational constant in ('day', 'AU', 'Msun') coordinates
# sim = rebound.Simulation()
# sim.units = ('day', 'AU', 'Msun')
# G_ = sim.G
G_ = 0.00029591220828559104
mu = tf.constant(G_)

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

class AsteroidPosition(keras.layers.Layer):
    """
    Compute orbit positions for asteroids in the solar system from the initial orbital elements with the Kepler model.
    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
    Outputs of the model are the position of the asteroid relative to the sun.    
    """
    def __init__(self, batch_size: int, **kwargs):
        """Constructor; batch_size is the number of elements to simulate at a time, e.g. 64."""
        super(AsteroidPosition, self).__init__(**kwargs)
        # Get trajectory size from ts
        self.batch_size = batch_size

        # Layer mapping orbital elements to cartesian coordinates
        self.model_e2c = make_model_elt_to_pos(batch_size=batch_size)

    def call(self, t, a, e, inc, Omega, omega, f, epoch):
        """
        Simulate the orbital trajectories.  
        Snapshot times t shared by all the input elements.  
        The inputs orbital elements and reference epoch should all have size (batch_size,).
        """
        # Alias traj_size, batch_size
        traj_size = t.shape[0]
        batch_size = self.batch_size

        # Compute eccentric anomaly E from f and e
        M = TrueToMeanAnomaly(name='TrueToMeanAnomaly')([f, e])
        
        # Compute mean motion N from mu and a
        a3 = tf.math.pow(a, 3, name='a3')
        mu_over_a3 = tf.divide(mu, a3, name='mu_over_a3')
        N = tf.sqrt(mu_over_a3, name='N')

        # Reshape t to (batch_size, traj_size, 1)
        target_shape = (-1, 1)
        # print(f'ts.shape = {ts.shape}')
        # First repeat ts batch_size times; now size is (traj_size, batch_size, 1)
        t_rep= keras.layers.RepeatVector(n=batch_size, name='ts_rep')(keras.backend.reshape(t, target_shape))
        # print(f't_rep.shape = {t_rep.shape}')
        # Transpose axes to make shape (batch_size, traj_size, 1)
        t_vec = tf.transpose(t_rep, perm=(1,0,2))
        # print(f't_vec.shape = {t_vec.shape}')
        # Reshape epoch to (batch_size, traj_size, 1)
        epoch_vec = keras.layers.RepeatVector(n=traj_size, name='epoch_vec')(keras.backend.reshape(epoch, target_shape))
        
        # Subtract epoch from t_vec; now it is relative to the epoch
        t_vec = t_vec - epoch_vec
        
        # ******************************************************************
        # Predict orbital elements over time
        
        # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size, 1)
        target_shape = (-1, 1)
        a_t = keras.layers.RepeatVector(n=traj_size, name='a_t')(keras.backend.reshape(a, target_shape))
        e_t = keras.layers.RepeatVector(n=traj_size, name='e_t')(keras.backend.reshape(e, target_shape))
        inc_t = keras.layers.RepeatVector(n=traj_size, name='inc_t')(keras.backend.reshape(inc, target_shape))
        Omega_t = keras.layers.RepeatVector(n=traj_size, name='Omega_t')(keras.backend.reshape(Omega, target_shape))
        omega_t = keras.layers.RepeatVector(n=traj_size, name='omega_t')(keras.backend.reshape(omega, target_shape))
        # Repeat the gravitational field strength to vector of shape (batch_size, traj_size)
        mu_t = keras.layers.RepeatVector(n=traj_size, name='mu_t')(keras.backend.reshape(mu, (1, 1)))
        
        # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
        M0_t = keras.layers.RepeatVector(n=traj_size, name='M0_t')(keras.backend.reshape(M, target_shape))
        N0_t = keras.layers.RepeatVector(n=traj_size, name='N0_t')(keras.backend.reshape(N, target_shape))
        # Compute the mean anomaly M(t) as a function of time
        N_mult_t = keras.layers.multiply(inputs=[N0_t, t_vec])
        M_t = keras.layers.add(inputs=[M0_t, N_mult_t])
    
        # Compute the true anomaly from the mean anomly and eccentricity
        f_t = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M_t, e_t])
    
        # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
        elt_t = (a_t, e_t, inc_t, Omega_t, omega_t, f_t, mu_t,)
        
        # ******************************************************************
        # Convert orbital elements to cartesian coordinates 
        
        # Convert from orbital elements to cartesian coordinates (position only)
        q = self.model_e2c(elt_t)
        
        # Name the outputs
        q = Identity(name='q')(q)
        # v = Identity(name='v')(v)
    
        return q

# ********************************************************************************************************************* 
class DirectionUnitVector(keras.layers.Layer):
    """
    Layer to compute the direction from object 1 (e.g. earth) to object 2 (e.g. asteroid)
    """
    
    # don't declare this tf.function because it breaks when using it with q_earth
    # still not entirely sure how tf.function works ...
    # tf.function
    def call(self, inputs):
        # Unpack inputs
        q1, q2 = inputs
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
#class AsteroidDirection(keras.layers.Layer):
#    """
#    Layer to compute the direction from earth to asteroid.
#    """
#    def __init__(self, batch_size: int, **kwargs):
#        """Constructor; batch_size is the number of elements to simulate at a time, e.g. 64."""
#        super(AsteroidPosition, self).__init__(**kwargs)
#        # Get trajectory size from ts
#        self.batch_size = batch_size
#
#        # Layer mapping orbital elements to cartesian coordinates
#        self.model_e2c = make_model_elt_to_pos(batch_size=batch_size)
#

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
    t = keras.backend.constant(ts, name='t')

    # Call asteroid position layer
    q = AsteroidPosition(batch_size)(t, a, e, inc, Omega, omega, f, epoch)
    
    # Name the outputs
    q = Identity(name='q')(q)
    # v = Identity(name='v')(v)
    
    # Wrap up the outputs
    outputs = (q,)

    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=outputs, name='model_asteroid_pos')
    return model


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
    # Get trajectory size from ts
    traj_size: int = ts.shape[0]
    
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
    
    # Model with asteroid position
    model_ast_pos = make_model_ast_pos(ts=ts, batch_size=batch_size)
    # Get the asteroid position with this model
    q = model_ast_pos(inputs)
    # print(f'q.shape = {q.shape}')

    # Take a one time snapshot of the earth's position at these times
    q_earth_np = get_earth_pos(ts)
    # print(f'q_earth_np loaded, shape = {q_earth_np.shape}')
    space_dims=3
    q_earth_np = q_earth_np.reshape(1, traj_size, space_dims)
    q_earth = keras.backend.constant(q_earth_np, 
                                     dtype=tf.float32,
                                     shape=q_earth_np.shape,
                                     name='q_earth')
    # print(f'q_earth keras.constant created, shape = {q_earth.shape}')

    # Unit displacement vector (direction) from earth to asteroid
    u = DirectionUnitVector(name='dir_earth_ast')([q_earth, q])

    # Name the outputs
    u = Identity(name='u')(u)
    
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

