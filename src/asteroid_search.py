"""
Harvard IACS Masters Thesis
asteroid_search.py: Search for orbital elements of asteroids given observational data.

Michael S. Emanuel
Thu Oct 17 15:24:10 2019
"""

# Library imports
import tensorflow as tf
# from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import numpy as np

# Local imports
from asteroids import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset
# from observation_data import make_synthetic_obs_tensors
from asteroid_model import make_model_ast_dir

# Aliases
keras = tf.keras

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
class ObservationScore(keras.losses.Loss):
    """Specialized loss for predicted asteroid directions."""
    def call(self, u_obs, u_pred_in):
        """
        Loss is the negative of the total t-score summed over all points with non-negative scores
        INPUTS:
            u_obs: observed directions, PADDED to a regular tensor; shape (traj_size, max_obs, 3,)
            u_pred_in: tuple (u_pred, R)
                u_pred: predicted directions; shape (batch_size, traj_size, 3,)
                R: resolution factor in radians for score function
        """
        # Unpack inputs; y_pred includes both predicted direction AND resolution factor R in radians
        u_pred, R = u_pred_in

        # Get sizes
        batch_size: int
        traj_size: int
        max_obs: int
        batch_size, traj_size = u_pred.shape[0:2]
        max_obs = u_obs.shape[1]

        # Debug
        # print(f'u_obs.shape = {u_obs.shape}')
        # print(f'u_pred.shape = {u_pred.shape}')
        # print(f'R.shape = {R.shape}')
        # print(f'batch_size={batch_size}, traj_size={traj_size}, max_obs={max_obs}.')

        # The observations; broadcast to shape (1, traj_size, max_obs, 3)
        y = tf.broadcast_to(u_obs, (1, traj_size, max_obs, space_dims))
        # The predicted directions; reshape to (batch_size, traj_size, 1, 3)
        x = tf.reshape(u_pred, (batch_size, traj_size, 1, space_dims))
        
        # Debug
        # print(f'y.shape = {y.shape}')
        # print(f'x.shape = {x.shape}')
        # print(f'R.shape = {R.shape}')

        # The difference in directions; size (batch_size, traj_size, max_obs, 3)
        z = y-x
        # print(f'z.shape = {z.shape}')
        
        # The scaling coefficient for scores; score = exp(-1/2 A epsilon^2)
        A = 1.0 / R**2
        
        # The coefficient that multiplies epsilon^2
        B = tf.reshape(-0.5 * A, (batch_size, 1, 1,))
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
        t_score = (score - mu) / sigma
        # print(f't_score.shape = {t_score.shape}')
        
        # Sum the non-negative scores; put a negative sign because TF minimizes losses
        # return -K.sum(tf.nn.relu(t_score))
        
        # Sum the scores
        return -K.sum(t_score)

# ********************************************************************************************************************* 
def make_search_model():
    """Make a model to search for asteroids."""
    pass

# ********************************************************************************************************************* 
# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# Dataset of observations: synthetic data on first 1000 asteroids
n0: int = 1
# n1: int = 1000
n1: int = 100
ds = make_synthetic_obs_dataset(n0=n0, n1=n1)
# Get reference times
batch_in, batch_out = list(ds.take(1))[0]
ts = batch_in['t']
# Get observed directions
u_obs_ragged = batch_out['u']
ast_num = batch_out['ast_num']

# Get trajectory size
traj_size: int = ts.shape[0]
space_dims: int = 3

# Model predicting asteroid direction with batch size 1
model_1 = make_model_ast_dir(ts=ts, batch_size=1)
model_64 = make_model_ast_dir(ts=ts, batch_size=64)
model = model_64
batch_size: int = 64

# Values to try: first 64 asteroids
dtype = np.float32
n0: int = 1
n1: int = 64
mask = (n0 <= ast_elt.Num) & (ast_elt.Num <= n1)

# Make input batch
a = ast_elt.a[mask].astype(dtype).to_numpy()
e = ast_elt.e[mask].astype(dtype).to_numpy()
inc = ast_elt.inc[mask].astype(dtype).to_numpy()
Omega = ast_elt.Omega[mask].astype(dtype).to_numpy()
omega = ast_elt.omega[mask].astype(dtype).to_numpy()
f = ast_elt.f[mask].astype(dtype).to_numpy()
epoch = ast_elt.epoch_mjd[mask].astype(dtype).to_numpy()

# The resolution factor in degrees and radians
R_deg: float = 10.0
R_rad: float = np.deg2rad(R_deg)
# Wrap resolution R into a numpy array
R_np = R_rad * np.ones(shape=batch_size, dtype=dtype)
R = tf.convert_to_tensor(R_np)

# Wrap inputs
inputs = {
    'a': a, 
    'e': e, 
    'inc': inc, 
    'Omega': Omega, 
    'omega': omega, 
    'f': f, 
    'epoch': epoch,
    'R': R
}

# Predicted asteroid positions
u_pred = model.predict(inputs)

# Total number of observations
num_obs: float = np.sum(u_obs_ragged.row_lengths())

# Pad u_obs into a regular tensor
pad_default = np.array([0.0, 0.0, 65536.0])
u_obs = u_obs_ragged.to_tensor(default_value=pad_default)
max_obs: int  = u_obs.shape[1]

# The observations; broadcast to shape (1, traj_size, max_obs, 3)
y = tf.broadcast_to(u_obs, (1, traj_size, max_obs, space_dims))
# The predicted directions; reshape to (batch_size, traj_size, 1, 3)
x = tf.reshape(u_pred, (batch_size, traj_size, 1, space_dims))
# The difference in directions; size (batch_size, traj_size, max_obs, 3)
z = y-x
# The scaling coefficient for scores; score = exp(-1/2 A epsilon^2)
A_np = 1.0 / R_np**2
# The coefficient that multiplies epsilon^2
B_np = -0.5 * A_np
B = K.constant(B_np.reshape((batch_size, 1, 1,)))
# Argument to the exponential
arg = tf.multiply(B, tf.linalg.norm(z, axis=-1))
# The score function
score = K.sum(tf.exp(arg), axis=(1,2))
# The expected score
mu = num_obs * score_mean(A_np)
# The expected variance
sigma2 = num_obs * score_var(A_np)
sigma = np.sqrt(sigma2)
# The t-score
t_score = (score - mu) / sigma

# Test custom loss layer
score_layer = ObservationScore()
u_pred_in = u_pred, R
loss = score_layer(u_obs, u_pred_in)

