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
    def __init__(self, A=1.0, **kwargs):
        super(ObservationScore, self).__init__(**kwargs)
        self.A_inv = tf.constant(1.0 / A)

    def call(self, u_obs, u_pred):
        # Error between predicted and observed directions
        u_diff = (u_obs - u_pred)
        # Scaled by resolution parameter
        u_diff_scaled = self.A_inv * u_diff
        # Argument is -1/2 square of scaled difference
        arg = -0.5 * math_ops.square(u_diff_scaled)
        # Score contribution is e^(0.5 * scaled_diff)
        score_contrib = math_ops.exp(arg)
        return K.sum(score_contrib, axis=-1)

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

# Wrap inputs
inputs = {
    'a': a, 
    'e': e, 
    'inc': inc, 
    'Omega': Omega, 
    'omega': omega, 
    'f': f, 
    'epoch': epoch
}

# Predicted asteroid positions
u_pred = model.predict(inputs)

# Get difference between observed and predicted
y = u_obs
x = u_pred[0]
x2 = tf.reshape(x, (traj_size, 1, space_dims))
u_diff = y - x2

# ragged tensor compoments of observations
u_obs_tensor = u_obs.to_tensor()
rowids = u_obs.value_rowids()
row_starts = u_obs.row_starts()
