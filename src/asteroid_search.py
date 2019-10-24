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
from asteroids import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, random_direction
# from observation_data import make_synthetic_obs_tensors
from asteroid_model import make_model_ast_dir
from tf_utils import Identity

# Aliases
keras = tf.keras

# Constants
space_dims = 3

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
class DirectionDifference(keras.layers.Layer):
    """Compute the difference in direction between observed and predicted directions"""
    def call(self, u_obs, u_pred):
        """
        Loss is the negative of the total t-score summed over all points with non-negative scores
        INPUTS:
            u_obs: observed directions, PADDED to a regular tensor; shape (traj_size, max_obs, 3,)
            u_pred: predicted directions; shape (batch_size, traj_size, 3,)
        """
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
    def __init__(self, **kwargs):
        super(TrajectoryLoss, self).__init__(**kwargs)
        
    def call(self, t_score):
        """
        Loss is the negative of the total t-score summed over all points with non-negative scores
        INPUTS:
            t_score: (score - mu) / sigma; mu, sigma are theoretical mean and variance for num_obs
                     observations uniformly distributed on the unit sphere.
        """
        # Sum the scores with a negative sign
        return -K.sum(t_score)

# ********************************************************************************************************************* 
class AsteroidSearchModel(keras.Model):
    def __init__(self, ts: np.array, row_lens, 
                 batch_size: int = 64, R_deg: float=10.0, **kwargs):
        super(AsteroidSearchModel, self).__init__(**kwargs)

        # Save input arguments
        # self.ts = ts
        self.batch_size = batch_size
        self.R_deg = R_deg
        self.row_lens = row_lens
        
        # The trajectory size
        self.traj_size = ts.shape[0]
        # Total number of observations; cast to tf.float32 type for compatibility with score
        self.num_obs = tf.cast(tf.reduce_sum(row_lens), tf.float32)
        
        # Load asteroid names and orbital elements
        self.ast_elt = load_data_asteroids()
        # Placeholder for orbital elements
        elt_placeholder = np.zeros(shape=(batch_size,), dtype=np.float32)

        # Create trainable variables for the 6 orbital elements
        self.a = tf.Variable(initial_value=elt_placeholder, trainable=True, name='a')
        self.e = tf.Variable(initial_value=elt_placeholder, trainable=True, name='e')
        self.inc = tf.Variable(initial_value=elt_placeholder, trainable=True, name='inc')
        self.Omega = tf.Variable(initial_value=elt_placeholder, trainable=True, name='Omega')
        self.omega = tf.Variable(initial_value=elt_placeholder, trainable=True, name='omega')
        self.f = tf.Variable(initial_value=elt_placeholder, trainable=True, name='f')
        
        # Create a non-trainable variable for the epoch
        self.epoch = tf.Variable(initial_value=elt_placeholder, trainable=False, name='epoch')
        
        # Initialize the orbital elements to the first 64 asteroids
        self.set_orbital_elements_asteroids(1)
        
        # Dictionary wrapping inputs for the asteroid direction model
        self.inputs_ast_dir = {
            'a': self.a,
            'e': self.e,
            'inc': self.inc,
            'Omega': self.Omega,
            'omega': self.omega,
            'f': self.f,
            'epoch': self.epoch,        
        }                

        # Create trainable variable for resolution factor
        R_np = np.deg2rad(self.R_deg) * np.ones(shape=batch_size, dtype=np.float32)
        self.R = tf.Variable(initial_value=R_np, trainable=True, name='R')
        
        # Create a layer to compute directions from orbital elements at these times
        self.model_ast_dir = make_model_ast_dir(ts=ts, batch_size=batch_size)
        
        # Create variable to store the predicted directions
        # u_pred_shape = shape=(batch_size, traj_size, space_dims)
        self.u_pred = tf.Variable(initial_value=self.model_ast_dir.predict(self.inputs_ast_dir),
                                  trainable=False, name='u_pred')
        
        # Create layer to score trajectories
        self.traj_score = TrajectoryScore(batch_size=self.batch_size, name='traj_score')
        
        # Variable to save the raw and t scores
        self.score = tf.Variable(initial_value=elt_placeholder, trainable=False, name='score')
        self.t_score = tf.Variable(initial_value=elt_placeholder, trainable=False, name='t_score')
        
        # Create layer to accumulate losses
        self.traj_loss= TrajectoryLoss()

    def set_orbital_elements_asteroids(self, n0: int):
        """Set a batch of orbital elements with data for a block of asteroids starting at n0"""
        # Get start and end index location of this asteroid number
        i0: int = ast_elt.index.get_loc(n0)
        i1: int = i0 + self.batch_size
        # Assign the orbital elements and epoch
        self.a.assign(ast_elt.a[i0:i1])
        self.e.assign(ast_elt.e[i0:i1])
        self.inc.assign(ast_elt.inc[i0:i1])
        self.Omega.assign(ast_elt.Omega[i0:i1])
        self.omega.assign(ast_elt.omega[i0:i1])
        self.f.assign(ast_elt.f[i0:i1])
        self.epoch.assign(ast_elt.epoch_mjd[i0:i1])

    def set_R(self, R_deg: float):
        """Update the resolution factor"""
        R_np = np.deg2rad(R_deg) * np.ones(shape=self.batch_size, dtype=np.float32)
        self.R.assign(R_np)

    def predict_directions(self):
        """
        Compute orbits and implied directions for one batch of orbital element parameters.
        """
        # Predict asteroid positions and directions from earth;
        # update u_pred in place with the results
        self.u_pred.assign(self.model_ast_dir.predict(self.inputs_ast_dir))

    def __call__(self, inputs):
        """
        Compute orbits and implied directions for one batch of orbital element parameters.
        Score the predicted directions with the current resolution settings.
        """
        # Unpack the inputs
        u_obs = inputs
        
        # Predicted asteroid positions
        self.predict_directions()

        # Difference in between observed and predicted directions
        z = DirectionDifference(name='direction_difference')(u_obs, self.u_pred)

        # raw and t score
        score, t_score = self.traj_score(z, self.R, self.num_obs)
        self.score.assign(score)
        self.t_score.assign(t_score)        
        
        # add negative t_score to the loss
        # self.add_loss(self.traj_loss, inputs=self.t_score)
    
        # Return
        return t_score

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
u_obs_ragged = batch_in['u']
# Get asteroid numbers for these observations
ast_num = batch_out['ast_num']

# Get trajectory size
traj_size: int = ts.shape[0]
space_dims: int = 3

# Row lengths
row_lens = u_obs_ragged.row_lengths()
# Total number of observations
num_obs: float = float(np.sum(row_lens))

# Pad u_obs into a regular tensor
pad_default = np.array([0.0, 0.0, 65536.0])
u_obs = u_obs_ragged.to_tensor(default_value=pad_default)
max_obs: float  = float(u_obs.shape[1])

# Build model
model = AsteroidSearchModel(ts=ts, row_lens=row_lens)


