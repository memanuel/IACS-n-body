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
# https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_vec_feature(values):
  """Returns a float_list from a numpy array of float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

# ********************************************************************************************************************* 
def serialize_ast_traj(it):
    """Serialize one asteroid trajectory to a proto message"""
    # Unpack inputs and outputs from iterator
    inputs, outputs = next(it)
    
    # Unpack inputs (orbital elements)
    a = inputs['a']
    e = inputs['e']
    inc = inputs['inc']
    Omega = inputs['Omega']
    omega = inputs['omega']
    f = inputs['f']
    ts = inputs['ts']
    
    # Unpack outputs (position and velocity)
    q = outputs['q']
    v = outputs['v']
    
    # Dictionary mapping feature names to date types compatible with tf.Example
    feature = {
        'a': _float_feature(a),
        'e': _float_feature(e),
        'inc': _float_feature(inc),
        'Omega': _float_feature(Omega),
        'omega': _float_feature(omega),
        'f': _float_feature(f),
        'ts': _float_vec_feature(ts),
        # q and v must be flattened
        'q': _float_vec_feature(tf.reshape(q, [-1])),
        'v': _float_vec_feature(tf.reshape(v, [-1])),
    }
    
    # create a message using tf.train.Example
    proto_msg = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto_msg.SerializeToString()

def deserialize_ast_traj(proto_msg):
    """Deserialize an asteroid trajectory stored as a proto message"""
    pass

# ********************************************************************************************************************* 
def map_func(n: int):
    batch_size: int = 1000
    n0 = batch_size*n
    n1 = n0 + batch_size
    return make_dataset_one_file(n0, n1)

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

if 'ds' not in globals():
    ds = make_dataset_one_file(n0=n0, n1=n1)

# one item from batch
batch_in, batch_out = list(ds.take(1))[0]

if 'ds1' not in globals():
    ds1 = make_dataset_one_file(n0=0, n1=1000)
    ds2 = make_dataset_one_file(n0=1000, n1=2000)
    # combined
    ds_concat = ds1.concatenate(ds2)

# ds1 = map_func(0)
# ds2 = map_func(0)
# idxs = tf.data.Dataset.from_tensor_slices(np.arange(542))
idxs = tf.data.Dataset.from_tensor_slices(np.arange(8))

# iterator for the dataset
it = iter(ds)

# serialize one example
msg = serialize_ast_traj(it)

# recover example
example = tf.train.Example.FromString(msg)
