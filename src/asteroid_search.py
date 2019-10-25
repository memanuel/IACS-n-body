"""
Harvard IACS Masters Thesis
asteroid_search.py: Search for orbital elements of asteroids given observational data.

Michael S. Emanuel
Thu Oct 17 15:24:10 2019
"""

# Library imports
import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import numpy as np

# Local imports
from asteroid_integrate import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, random_direction
# from observation_data import make_synthetic_obs_tensors
from asteroid_data import get_earth_pos, orbital_element_batch
from asteroid_model import AsteroidDirection
from asteroid_model import AsteroidPosition, DirectionUnitVector
# from asteroid_model make_model_ast_dir, make_model_ast_pos
from tf_utils import Identity

# Aliases
keras = tf.keras

# Constants
space_dims = 3

# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# ********************************************************************************************************************* 
# Functions for scoring predicted asteroid directions vs. observations
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def score_mean(A: np.array):
    """
    Expected value of the score function 
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    If we use cylindrical coordinates (z, phi), with r^2 +z^2 = 1, x^2 + y^2 = r^2, and
    x = r cos(phi), y = r sin(phi), z = z
    and set the true direction at (0, 0, 1), we can get the EV by integrating over the sphere.
    epsilon is a right triangle with sides (1-z) and r, regardless of phi, so
    epsilon^2 = (1-z)^2 + r^2 = 1 -2z + z^2 + r^2 = 1 + (r^2+z^2) - 2z = 2 - 2z = 2(1-z)
    This can be integrated symbolically.
    """
    # The expected value is (1 - e^-2A) / 2A
    # return (1.0 - np.exp(-2.0*A)) / (2*A)
    minus_two_A: np.array = -2.0*A
    return np.expm1(minus_two_A) / minus_two_A

# ********************************************************************************************************************* 
def score_var(A: np.array):
    """
    Variance of the score function 
    f(epsilon) = exp(-1/2 A epsilon^2)
    This can be integrated symbolically.
    """
    # The variance is (A*Coth(A)) * E[f]^2
    # Calculate the leading factor
    B: np.array = (A / np.tanh(A)) - 1.0
    # Calculate the mean
    mean: np.array = score_mean(A)
    return B * mean * mean

# ********************************************************************************************************************* 
def score_std(A: np.array):
    """
    Standard deviation of the score function 
    f(epsilon) = exp(-1/2 A epsilon^2)
    This can be integrated symbolically.
    """
    # The variance is (A*Coth(A)) * E[f]^2
    # The std deviation is sqrt(A*Coth(A)) * E[f]
    # Calculate the leading factor
    B: np.array = np.sqrt(A / np.tanh(A) - 1.0)
    # Calculate the mean
    mean: np.array = score_mean(A)
    return B * mean

# ********************************************************************************************************************* 
def test_score_moments_one(A: float, mean_exp: float, var_exp: float) -> bool:
    """Test the mean and variance for the score function with three known values"""
    # Test the mean
    mean_calc = score_mean(A)
    isOK: bool = np.isclose(mean_exp, mean_calc, atol=1.0E-7)     
    if not isOK:
        print(f'Failed on A = 1.0, expected mean {mean_exp}, got mean {mean_calc}.')

    # Test the variance
    var_calc = score_var(A)
    isOK = isOK and np.isclose(var_exp, var_calc, atol=1.0E-7)
    if not isOK:
        print(f'Failed on A = 1.0, expected var {var_exp}, got var {var_calc}.')
        
    return isOK

# ********************************************************************************************************************* 
def test_score_moments():
    """Test the mean and variance for the score function with three known values"""
    # When A = 1.0, mean = 0.432332, var = 0.0585098
    A = 1.0
    mean_exp = 0.432332
    var_exp = 0.0585098
    isOK: bool = test_score_moments_one(A, mean_exp, var_exp)
    
    # When A = 32.828063500117445 (10 degrees), mean = 0.0152309, var = 0.00738346
    A = 1.0 / np.deg2rad(10.0)**2
    mean_exp = 0.0152309
    var_exp = 0.00738346
    isOK = isOK and test_score_moments_one(A, mean_exp, var_exp)
    
    # Report results
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'Test score moments: ***** {msg} *****')    
    return isOK

# ********************************************************************************************************************* 
def test_score_mean_num(N: int, R_deg: float):
    """
    Numerical test of score_mean
    INPUTS:
        N: number of points to test
        R_deg: resolution factor in degrees
    """
    # Convert resolution factor to A
    R: float = np.deg2rad(R_deg)
    A: float = 1.0 / R**2
    # Calculated mean and variance
    mean_calc: float = score_mean(A)
    std_calc: float = score_std(A)

    # Numerical samples
    directions: np.array = random_direction(N)
    # Difference with north pole    
    north_pole = np.array([0.0, 0.0, 1.0])
    epsilon = directions- north_pole
    epsilon2 = np.sum(epsilon*epsilon, axis=-1)

    # Simulated mean and stdev
    scores = np.exp(-0.5 * A * epsilon2)
    mean_num: float = np.mean(scores)
    mean_error_rel = np.abs(mean_num-mean_calc) / mean_calc
    std_num: float = np.std(scores)
    std_error_rel = np.abs(std_num-std_calc) / std_calc

    # Test the mean
    rtol: float = 2.0 / N**0.4
    isOK: bool = np.isclose(mean_calc, mean_num, rtol=rtol)
    print(f'Test score function with N={N}, R={R_deg} degrees:')
    print(f'Mean:')
    print(f'calc      = {mean_calc:10.8f}')
    print(f'num       = {mean_num:10.8f}')
    print(f'error_rel = {mean_error_rel:10.8f}')
    
    # Test the standard deviation
    rtol = 2.0 / N**0.4
    isOK = isOK and np.isclose(mean_calc, mean_num, rtol=rtol)
    print(f'Standard Deviation:')
    print(f'calc      = {std_calc:10.8f}')
    print(f'num       = {std_num:10.8f}')
    print(f'error_rel = {std_error_rel:10.8f}')
    
    # Summary results    
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class DirectionDifference(keras.layers.Layer):
    """Compute the difference in direction between observed and predicted directions"""
    def __init__(self, batch_size: int, traj_size: int, max_obs: int, **kwargs):
        super(DirectionDifference, self).__init__(**kwargs)
        # Save sizes
        self.batch_size = batch_size
        self.traj_size = traj_size
        self.max_obs = max_obs
    
    def call(self, u_obs, u_pred):
        """
        INPUTS:
            u_obs: observed directions, PADDED to a regular tensor; shape (traj_size, max_obs, 3,)
            u_pred: predicted directions; shape (batch_size, traj_size, 3,)
        """
        # Get sizes
        # batch_size: int
        # traj_size: int
        # batch_size, traj_size = u_pred.shape[0:2]
        # max_obs: int = u_obs.shape[-1]
        batch_size = self.batch_size
        traj_size = self.traj_size
        max_obs = self.max_obs

        # Debug
        # print(f'u_obs.shape = {u_obs.shape}')
        # print(f'u_pred.shape = {u_pred.shape}')
        # print(f'batch_size={batch_size}, traj_size={traj_size}, max_obs={max_obs}.')

        # The observations; broadcast to shape (1, traj_size, max_obs, 3)
        y = tf.broadcast_to(u_obs, (1, traj_size, max_obs, space_dims))
        # print(f'y.shape = {y.shape}')
        # The predicted directions; reshape to (batch_size, traj_size, 1, 3)
        x = tf.reshape(u_pred, (batch_size, traj_size, 1, space_dims))
        # print(f'x.shape = {x.shape}')
        
        # The difference in directions; size (batch_size, traj_size, max_obs, 3)
        z = tf.subtract(y, x, name='z')
        # print(f'z.shape = {z.shape}')

        return z

# ********************************************************************************************************************* 
class TrajectoryScore(keras.layers.Layer):
    """Score candidate trajectories"""
    def __init__(self, batch_size: int, **kwargs):
        super(TrajectoryScore, self).__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, z: tf.Tensor, R: tf.Tensor, num_obs: float):
        """
        Score candidate trajectories in current batch based on how well they match observations
        INPUTS:
            z: difference in direction between u_pred and u_obs
            R: resolution factor in radians for score function
            num_obs: total number of observations (real, not padded!)
        """
        # The scaling coefficient for scores; score = exp(-1/2 A epsilon^2)
        A = 1.0 / R**2
        
        # The coefficient that multiplies epsilon^2
        B = tf.reshape(-0.5 * A, (self.batch_size, 1, 1,))
        # print(f'B.shape = {B.shape}')
        
        # Argument to the exponential
        arg = tf.multiply(B, tf.linalg.norm(z, axis=-1))
        # print(f'arg.shape = {arg.shape}')
        
        # The score function
        score = K.sum(tf.exp(arg), axis=(1,2))
        # print(f'score.shape = {score.shape}')
        
        # The expected score
        mu = num_obs * score_mean(A)
        # print(f'mu.shape = {mu.shape}')
        
        # The expected variance
        sigma2 = num_obs * score_var(A)
        sigma = tf.sqrt(sigma2)
        # print(f'sigma.shape = {sigma.shape}')
        
        # The t-score for each sample
        # This is a modified t-score with an extra penalty for overly large R
        mu_factor: float = 1.10
        t_score = (score - mu_factor * mu) / sigma
        # print(f't_score.shape = {t_score.shape}')

        # Return both the raw and t scores
        return score, t_score

# ********************************************************************************************************************* 
class TrajectoryLoss(keras.losses.Loss):    
    """Specialized loss for predicted asteroid directions."""
    def __init__(self, batch_size: int, traj_size: int, max_obs: int, num_obs: float, **kwargs):
        super(TrajectoryLoss, self).__init__(**kwargs)
        self.num_obs = num_obs
        
        # The direction difference layer
        self.dir_diff_layer = DirectionDifference(batch_size=batch_size, traj_size=traj_size, 
                                                  max_obs=max_obs, name='z')
        # The trajectory score layer
        self.traj_score_layer = TrajectoryScore(batch_size=batch_size, name='traj_score')
        

    def call(self, y_true, y_pred, sample_weight=None):
        """
        Loss is the negative of the total t-score summed over all pointss
        INPUTS:
            t_score: (score - mu) / sigma; mu, sigma are theoretical mean and variance for num_obs
                     observations uniformly distributed on the unit sphere.
        """
        # Alias inputs
        u_obs = y_true
        u_pred = y_pred
        
        # Compute the difference in direction
        z = self.dir_diff_layer(u_obs, u_pred)
        
        # TODO: Make R flow through
        # Placeholder: ignore R
        R = tf.ones(u_pred.shape[0])*np.deg2rad(10.0)
        
        # Compute the scores from z and R
        score, t_score = self.traj_score_layer(z, R, self.num_obs)
        score = Identity(name='score')(score)
        t_score = Identity(name='t_score')(t_score)

        # Sum the scores with a negative sign
        return -K.sum(t_score)

# ********************************************************************************************************************* 
class PredictedDirection(keras.layers.Layer):
    """Custom layer to predict directions from orbital elements."""
    def __init__(self, ts, elts_np: dict, batch_size: int, R_deg: float, **kwargs):
        super(PredictedDirection, self).__init__(**kwargs)
        self.a = tf.Variable(initial_value=elts_np['a'], trainable=True, name='a')
        self.e = tf.Variable(initial_value=elts_np['e'], trainable=True, name='e')
        self.inc = tf.Variable(initial_value=elts_np['inc'], trainable=True, name='inc')
        self.Omega = tf.Variable(initial_value=elts_np['Omega'], trainable=True, name='Omega')
        self.omega = tf.Variable(initial_value=elts_np['omega'], trainable=True, name='omega')
        self.f = tf.Variable(initial_value=elts_np['f'], trainable=True, name='f')

        # The epoch is not trainable
        self.epoch = tf.Variable(initial_value=elts_np['epoch'], trainable=False, name='epoch')

        # The resolution factor
        R_np  = np.deg2rad(R_deg) * np.ones_like(elts_np['a'])
        self.R = tf.Variable(initial_value=R_np, trainable=True, name='R')

        # Output times are a constant
        ts = keras.backend.constant(ts, name='ts')
    
        # Layer to compute asteroid direction from elements
        self.ast_dir_layer = AsteroidDirection(ts=ts, batch_size=batch_size, name='u')
        
    def call(self, inputs):
        """Predict directions with the current orbital elements; also current resolution"""
        u = self.ast_dir_layer(self.a, self.e, self.inc, self.Omega, self.omega, self.f, self.epoch)

        return u, self.R
        
# ********************************************************************************************************************* 
class SearchCandidates(keras.layers.Layer):
    """Custom layer to maintain state of orbital elements."""

    def __init__(self, elts_np: dict, batch_size: int, R_deg: float, **kwargs):
        super(SearchCandidates, self).__init__(**kwargs)
        self.a = tf.Variable(initial_value=elts_np['a'], trainable=True, name='a')
        self.e = tf.Variable(initial_value=elts_np['e'], trainable=True, name='e')
        self.inc = tf.Variable(initial_value=elts_np['inc'], trainable=True, name='inc')
        self.Omega = tf.Variable(initial_value=elts_np['Omega'], trainable=True, name='Omega')
        self.omega = tf.Variable(initial_value=elts_np['omega'], trainable=True, name='omega')
        self.f = tf.Variable(initial_value=elts_np['f'], trainable=True, name='f')

        # The epoch is not trainable
        self.epoch = tf.Variable(initial_value=elts_np['epoch'], trainable=False, name='epoch')
        
        # The resolution factor
        R_np  = np.deg2rad(R_deg) * np.ones_like(elts_np['a'])
        self.R = tf.Variable(initial_value=R_np, trainable=True, name='R')

    def call(self, inputs):
        """Return the current settings"""
        return self.a, self.e, self.inc, self.Omega, self.omega, self.f, self.epoch, self.R

# ********************************************************************************************************************* 
# Functional API models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_asteroid_search(ts: tf.Tensor, max_obs: int, 
                               elt_batch_size: int=64, time_batch_size: int=None):
    """Make functional API model for scoring elements"""
    traj_size: int = ts.shape[0]
    if time_batch_size is None:
        time_batch_size = traj_size

    # Inputs
    t = keras.Input(shape=(), name='t', batch_size=time_batch_size, dtype=tf.float32)
    idx = keras.Input(shape=(), name='idx', batch_size=time_batch_size, dtype=tf.int32)
    row_len = keras.Input(shape=(), name='idx', batch_size=time_batch_size)
    
    # Output times are a constant
    ts = keras.backend.constant(ts, name='ts')

    # Orbital elements to try
    elts_np = orbital_element_batch(1)
    R_deg = 10.0
    
    # The predicted direction and resolution
    # u, R = PredictedDirection(ts=ts, elts_np=elts_np, batch_size=elt_batch_size, R_deg=R_deg, name='pred_dir')(idx)

    # Name the direction and resolution
    # u = Identity(name='u')(u)
    # R = Identity(name='R')(R)

    a, e, inc, Omega, omega, f, epoch, R = \
        SearchCandidates(elts_np=elts_np, batch_size=elt_batch_size, R_deg=R_deg, name='candidates')(idx)
    R = Identity(name='R')(R)

    # The predicted direction
    u = AsteroidDirection(ts=ts, batch_size=elt_batch_size, name='u')(a, e, inc, Omega, omega, f, epoch)

    # Wrap inputs and outputs
    inputs = (t, idx,)
    outputs = (u, R)

    # Create model with functional API    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Instantiate custom loss layer
    # traj_loss = TrajectoryLoss(batch_size=elt_batch_size, traj_size=traj_size, max_obs=max_obs, num_obs=num_obs)
    
    
    # model.add_loss()
    return model

# ********************************************************************************************************************* 
# Dataset of observations: synthetic data on first 1000 asteroids
n0: int = 1
n1: int = 10
time_batch_size = None
# Build the dataset
ds, ts, row_len = make_synthetic_obs_dataset(n0=n0, n1=n1, batch_size=time_batch_size)
# Get example batch
batch_in, batch_out = list(ds.take(1))[0]
# Contents of this batch
t = batch_in['t']
idx = batch_in['idx']
row_len = batch_in['row_len']
u_obs = batch_out['u']

# Get trajectory size and max_obs
traj_size: int = ts.shape[0]
max_obs: int = u_obs.shape[1]

# Build functional model for asteroid score
R_deg: float = 10.0
elts_np = orbital_element_batch(1)
elt_batch_size = 64
model = make_model_asteroid_search(ts=ts, max_obs=max_obs, 
                                   elt_batch_size=elt_batch_size, time_batch_size=time_batch_size)


# pred_dir = PredictedDirection(ts=ts, elts_np=elts_np, batch_size=elt_batch_size, R_deg=R_deg, name='pred_dir')

# Total number of observations
num_obs: float = float(np.sum(row_len))
# pred = model.predict_on_batch(batch_in)
u_pred, R = model.predict_on_batch(batch_in)

z = DirectionDifference(batch_size=elt_batch_size, traj_size=traj_size, max_obs=max_obs, name='z')(u_obs, u_pred)
score, t_score = TrajectoryScore(batch_size=elt_batch_size)(z, R, num_obs)

traj_loss = TrajectoryLoss(batch_size=elt_batch_size, traj_size=traj_size, max_obs=max_obs, num_obs=num_obs)
batch_loss = traj_loss(u_obs, u_pred)

loss_dict = {
        'R': tf.losses.MeanSquaredError(),
        }
model.compile(loss=loss_dict)
# model.compile(loss={'u':traj_loss})
model.evaluate(ds)
