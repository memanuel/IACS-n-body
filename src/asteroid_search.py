"""
Harvard IACS Masters Thesis
asteroid_search.py: Search for orbital elements of asteroids given observational data.

Michael S. Emanuel
Thu Oct 17 15:24:10 2019
"""

# Library imports
import tensorflow as tf
import numpy as np

# Local imports
from asteroids import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, make_synthetic_obs_tensors
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

# Model predicting asteroid direction with batch size 1
model_1 = make_model_ast_dir(ts=ts, batch_size=1)
model_64 = make_model_ast_dir(ts=ts, batch_size=64)
model = model_64

# Values to try: first 64 asteroids
ast_num = 1
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
# u_pred = model()

from asteroid_model import make_model_ast_pos
model_ast_pos = make_model_ast_pos(ts=ts, batch_size=64)

# Get the asteroid position with this model
q = model_ast_pos(inputs)
