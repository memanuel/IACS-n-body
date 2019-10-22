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
def score_elements(a, e, inc, Omega, omega, f, t, u) -> float:
    """
    Score a set of orbital elements against a dataset of observations
    INPUTS:
        a: semi-major axis
        e: eccentricity
        inc: inclination
        Omega: longitude of the ascending node
        omega: argument of pericenter
        f: true anomaly
        ts: observation times (MJDs)
        u: observation directions (3 vectors)
    """
    pass

# ********************************************************************************************************************* 
class ObservationScore(keras.losses.Loss):
    """Specialized loss for predicted asteroid directions."""
    def call(self, preds, obs):
        # Unpack inputs
        x, R = preds
        y = obs
        # The difference in directions; size (batch_size, traj_size, max_obs, 3)
        z = y-x
        # Get the 
        A_np = -0.5 / R**2
        A = K.constant(A_np.reshape((batch_size, 1, 1,)))
        # Argument to the exponential
        arg = tf.multiply(A, tf.linalg.norm(z, axis=-1))
        # The score function
        score = K.sum(tf.exp(arg), axis=(1,2))
        #The loss is negative of the score
        return tf.multiply(-1.0, score)
    
# ********************************************************************************************************************* 
# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# Dataset of observations: synthetic data on first 1000 asteroids
n0: int = 1
# n1: int = 1000
n1: int = 10
ds = make_synthetic_obs_dataset(n0=n0, n1=n1)
# Get reference times
batch_in, batch_out = list(ds.take(1))[0]
ts = batch_in['t']
# Get observed directions
u_obs = batch_out['u']
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
R = R_rad * np.ones(shape=batch_size, dtype=dtype)

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
num_obs: float = np.sum(u_obs.row_lengths())

# Pad u_obs into a regular tensor
pad_default = np.array([0.0, 0.0, 65536.0])
u_ = u_obs.to_tensor(default_value=pad_default)
max_obs: int  = u_.shape[1]

# The observations; broadcast to shape (1, traj_size, max_obs, 3)
y = tf.broadcast_to(u_, (1, traj_size, max_obs, space_dims))
# The predicted directions; reshape to (batch_size, traj_size, 1, 3)
x = tf.reshape(u_pred, (batch_size, traj_size, 1, space_dims))
# The difference in directions; size (batch_size, traj_size, max_obs, 3)
z = y-x
A_np = -0.5 / R**2
A = K.constant(A_np.reshape((batch_size, 1, 1,)))
# Argument to the exponential
arg = tf.multiply(A, tf.linalg.norm(z, axis=-1))
# The score function
score = K.sum(tf.exp(arg), axis=(1,2))