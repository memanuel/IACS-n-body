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

# ********************************************************************************************************************* 
def make_dataset_one_file(n0: int, n1: int) -> tf.data.Dataset:
    """
    Wrap the data in one file of asteroid trajectory data into a TF Dataset
    INPUTS:
        n0: the first asteroid in the file, e.g. 0
        n1: the last asteroid in the file (exclusive), e.g. 1000
    OUTPUTS:
        ds: a tf.data.Dataset object for this 
    """
    # selected data type for TF tensors
    dtype = np.float32
    
    # Name of the numpy archive
    fname_np: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.npz'
    
    # The full array of positions and velocities
    q, v, elts, catalog = load_sim_np(fname_np=fname_np)

    # The object names
    object_names = catalog['object_names']

    # The snapshot times; offset to start time t0=0
    ts = catalog['ts'].astype(dtype)
    
    # mask for selected asteroids
    mask = (n0 <= ast_elt.Num) & (ast_elt.Num < n1)
    
    # count of selected asteroids
    # asteroid_names = list(ast_elt.Name[mask].to_numpy())
    N_ast: int = np.sum(mask)
    # offset for indexing into asteroids; the first [10] objects are sun and planets
    ast_offset: int = len(object_names) - N_ast

    # shrink down q and v to slice with asteroid data only; convert to selected data type
    q = q[:, ast_offset:, :].astype(dtype)
    v = v[:, ast_offset:, :].astype(dtype)

    # swap axes for time step and body number; TF needs inputs and outputs to have same number of samples
    # this means that inputs and outputs must first be indexed by asteroid number, then time time step
    q = np.swapaxes(q, 0, 1)
    v = np.swapaxes(v, 0, 1)

    # dict with inputs   
    inputs = {
        'a': ast_elt.a[mask].to_numpy().astype(dtype),
        'e': ast_elt.e[mask].to_numpy().astype(dtype),
        'inc': ast_elt.inc[mask].to_numpy().astype(dtype),
        'Omega': ast_elt.Omega[mask].to_numpy().astype(dtype),
        'omega': ast_elt.omega[mask].to_numpy().astype(dtype),
        'f': ast_elt.f[mask].to_numpy().astype(dtype),
        'epoch': ast_elt.epoch_mjd[mask].to_numpy().astype(dtype),
        'ts': np.tile(ts, reps=(N_ast,1,))
    }
    
    # dict with outputs
    outputs = {
        'q': q,
        'v': v,
    }

    # Build and return the dataset
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))    
    return ds

# ********************************************************************************************************************* 
# Start and end time of simulation
# dt0: datetime = datetime(2000, 1, 1)
# dt1: datetime = datetime(2040, 12, 31)

# Start and end dates as mjd
# t0: float = datetime_to_mjd(dt0)
# t1: float = datetime_to_mjd(dt1)


# Load the simulation archive for the first 1000 asteroids
n0: int = 0
n1: int = 1000

ds = make_dataset_one_file(n0=n0, n1=n1)

# one item from batch
batch_in, batch_out = list(ds.take(1))[0]
