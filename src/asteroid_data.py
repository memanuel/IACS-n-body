"""
Harvard IACS Masters Thesis
Generate TensorFlow datasets for asteroid trajectories.

Michael S. Emanuel
Sat Sep 21 10:38:38 2019
"""

# Library imports
import tensorflow as tf
# import rebound
import numpy as np
from datetime import datetime

# Local imports
# from astro_utils import datetime_to_mjd, mjd_to_datetime
# from planets import make_sim_planets
from asteroids import load_data, load_sim_np
# from utils import arange_inc

# ********************************************************************************************************************* 
# DataFrame of asteroid snapshots
ast_elt = load_data()

# Start and end time of simulation
dt0: datetime = datetime(2000, 1, 1)
dt1: datetime = datetime(2040, 12, 31)

# Load the simulation archive for the first 1000 asteroids
n0: int = 0
n1: int = 1000

# Name of the numpy archive
fname_np: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.npz'

# The full array of positions and velocities
q, v, elts, catalog = load_sim_np(fname_np=fname_np)
# The object names
object_names = catalog['object_names']
# The snapshot times; offset to start time t0=0
ts = catalog['ts']
epochs = catalog['epochs']

# mask for selected asteroids
mask = (n0 <= ast_elt.Num) & (ast_elt.Num < n1)

# names of selected asteroids
asteroid_names = list(ast_elt.Name[mask].to_numpy())
N_ast: int = np.sum(mask)

# epoch as MJD of selected asteroids
# epoch_mjd = ast_elt.epoch_mjd[mask].to_numpy()
# epoch as a datetime
# epoch_dt = [mjd_to_datetime(mjd) for mjd in epoch_mjd]

# Start and end dates as mjd
# t0: float = datetime_to_mjd(dt0)
# t1: float = datetime_to_mjd(dt1)

# Range of desired times for position output
# ts = arange_inc(t0, t1)

# dict with inputs
orb_a = ast_elt.a[mask].to_numpy()
orb_e = ast_elt.e[mask].to_numpy()
orb_inc = ast_elt.inc[mask].to_numpy()
orb_Omega = ast_elt.Omega[mask].to_numpy()
orb_omega = ast_elt.omega[mask].to_numpy()
orb_f = ast_elt.f[mask].to_numpy()


inputs = {
    'a': ast_elt.a[mask].to_numpy(),
    'e': ast_elt.e[mask].to_numpy(),
    'inc': ast_elt.inc[mask].to_numpy(),
    'Omega': ast_elt.Omega[mask].to_numpy(),
    'omega': ast_elt.omega[mask].to_numpy(),
    'f': ast_elt.f[mask].to_numpy(),
    'epoch': ast_elt.epoch_mjd[mask].to_numpy(),
    'ts': np.tile(ts, reps=(N_ast,1,))
}

# dict with outputs
outputs = {
    'q': np.swapaxes(q, 0, 1),
    'v': np.swapaxes(v, 0, 1),
}

ds = tf.data.Dataset.from_tensor_slices((inputs, inputs))
