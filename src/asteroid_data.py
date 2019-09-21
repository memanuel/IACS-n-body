"""
Harvard IACS Masters Thesis
Generate TensorFlow datasets for asteroid trajectories.

Michael S. Emanuel
Sat Sep 21 10:38:38 2019
"""

# Library imports
import rebound
import numpy as np
from datetime import datetime

# Local imports
from astro_utils import mjd_to_datetime
from planets import make_sim_planets
from asteroids import load_data, load_sim_np

# ********************************************************************************************************************* 
# DataFrame of asteroid snapshots
ast_elt = load_data()

# Start time of simulation
dt0: datetime = datetime(2000, 1, 1)

# Load the simulation archive for the first 1000 asteroids
n0: int = 0
n1: int = 1000

# Name of the numpy archive
fname_np: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.npz'

# The full array of positions and velocities
q, v, elts, catalog = load_sim_np(fname_np=fname_np)
# The object names
object_names = catalog['object_names']

ts = catalog['ts']
epochs = catalog['epochs']

# mask for selected asteroids
mask = (n0 <= ast_elt.Num) & (ast_elt.Num < n1)

# names of selected asteroids
asteroid_names = list(ast_elt.Name.to_numpy())

# epoch as MJD of selected asteroids
epoch_mjd = ast_elt.epoch_mjd[mask].to_numpy()
# epoch as a datetime
epoch_dt = [mjd_to_datetime(mjd) for mjd in epoch_mjd]

# mean anomaly of selected asteroids
M = ast_elt.M[mask].to_numpy()

# dict with inputs
inputs = {
    'a': ast_elt.a[mask].to_numpy(),
    'e': ast_elt.e[mask].to_numpy(),
    'inc': ast_elt.inc[mask].to_numpy(),
    'Omega': ast_elt.Omega[mask].to_numpy(),
    'omega': ast_elt.omega[mask].to_numpy(),
    'f': ast_elt.f[mask].to_numpy(),
    'epoch': epoch
}
