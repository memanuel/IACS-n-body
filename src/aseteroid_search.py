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
        t: observation times (MJDs)
        u: observation directions (3 vectors)
    """
    pass

# ********************************************************************************************************************* 
# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# Dataset of observations: synthetic data on first 1000 asteroids
n0: int = 1
n1: int = 1000
ds = make_synthetic_obs_dataset(n0=n0, n1=n1)
t, u, ast_num = make_synthetic_obs_tensors(n0=n0, n1=n1)
traj_size = t.shape[0]

# Model predicting asteroid direction
model = make_model_ast_dir(traj_size=traj_size)

# Values to try
ast_num = 1
a = ast_elt.a[ast_num]
e = ast_elt.e[ast_num]
inc = ast_elt.inc[ast_num]
Omega = ast_elt.Omega[ast_num]
omega = ast_elt.omega[ast_num]
f = ast_elt.f[ast_num]

