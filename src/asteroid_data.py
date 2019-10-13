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
from tqdm.auto import tqdm

# Local imports
from astro_utils import datetime_to_mjd
from asteroids import load_data, load_sim_np

# ********************************************************************************************************************* 
# DataFrame of asteroid snapshots
ast_elt = load_data()

# ********************************************************************************************************************* 
def get_earth_pos():
    """Get the position of earth consistent with the asteroid data"""
    # selected data type for TF tensors
    dtype = np.float32
    
    # Name of the numpy archive
    n0, n1 = 0, 1000
    fname_np: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.npz'

    # The full array of positions and velocities
    q, v, elts, catalog = load_sim_np(fname_np=fname_np)

    # The object names
    object_names = catalog['object_names']
   
    # Extract position of earth
    earth_idx = object_names.index('Earth')
    q_earth = q[:, earth_idx, :].astype(dtype)
    
    return q_earth

# ********************************************************************************************************************* 
def make_data_one_file(n0: int, n1: int) -> tf.data.Dataset:
    """
    Wrap the data in one file of asteroid trajectory data into a TF Dataset
    INPUTS:
        n0: the first asteroid in the file, e.g. 0
        n1: the last asteroid in the file (exclusive), e.g. 1000
    OUTPUTS:
        inputs: a dict of numpy arrays
        outputs: a dict of numpy arrays
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
    # Convert ts from relative time vs. t0 to MJD
    dt0 = datetime(2000, 1, 1)
    t_offset = datetime_to_mjd(dt0)
    ts += t_offset
    
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
    
    # Compute relative displacement to earth
    q_earth = get_earth_pos()
    traj_size = q_earth.shape[0]
    space_dims = 3
    q_rel = q - q_earth.reshape(1, traj_size, space_dims,)
    # Distance to earth
    r_earth = np.linalg.norm(q_rel, axis=2, keepdims=True)
    # Direction from earth to asteroid as unit vectors u =(ux, uy, uz)
    u = q_rel / r_earth

    # dict with inputs   
    inputs = {
        'a': ast_elt.a[mask].to_numpy().astype(dtype),
        'e': ast_elt.e[mask].to_numpy().astype(dtype),
        'inc': ast_elt.inc[mask].to_numpy().astype(dtype),
        'Omega': ast_elt.Omega[mask].to_numpy().astype(dtype),
        'omega': ast_elt.omega[mask].to_numpy().astype(dtype),
        'f': ast_elt.f[mask].to_numpy().astype(dtype),
        'epoch': ast_elt.epoch_mjd[mask].to_numpy().astype(dtype),
        'ts': np.tile(ts, reps=(N_ast,1,)),
    }
    
    # dict with outputs
    outputs = {
        'q': q,
        'v': v,
        'u': u
    }
    
    return inputs, outputs

# ********************************************************************************************************************* 
def make_dataset_pos_file(n0: int, n1: int, include_vel: bool) -> tf.data.Dataset:
    """
    Wrap the data in one file of asteroid trajectory data into a TF Dataset
    INPUTS:
        n0: the first asteroid in the file, e.g. 0
        n1: the last asteroid in the file (exclusive), e.g. 1000
    OUTPUTS:
        ds: a tf.data.Dataset object for this 
    """
    # Load data
    inputs_all, outputs_all = make_data_one_file(n0, n1)
    
    # Use all inputs
    inputs = inputs_all
    
    # Wrap up selected outputs
    outputs ={'q': outputs_all['q']}
    if include_vel:
        outputs['v'] = outputs_all['v']
    
    # Wrap into a dataset
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))    
    return ds

# ********************************************************************************************************************* 
def make_dataset_pos_n(n: int, include_vel: bool):
    """Convert the nth file, with asteroids (n-1)*1000 to n*1000, to a tf Dataset"""
    batch_size: int = 1000
    n0 = batch_size*n
    n1 = n0 + batch_size
    return make_dataset_pos_file(n0, n1, include_vel)

# ********************************************************************************************************************* 
def combine_datasets_pos(n0: int, n1: int, include_vel: bool = False, 
                     batch_size: int = 64, progbar: bool = False):
    """Combine datasets in [n0, n1) into one large dataset"""
    # Iteration range for adding to the initial dataset
    ns = range(n0+1, n1)
    ns = tqdm(ns) if progbar else ns
    # Initial dataset
    ds = make_dataset_pos_n(n0, include_vel)
    # Extend initial dataset
    for n in ns:
        try:
            ds_new = make_dataset_pos_n(n, include_vel)
            ds = ds.concatenate(ds_new)
        except:
            pass
    # Batch the dataset
    drop_remainder: bool = True    
    return ds.batch(batch_size, drop_remainder=drop_remainder)

# ********************************************************************************************************************* 
def make_dataset_ast_pos(n0: int, num_files: int, include_vel: bool = False):
    """Create a dataset spanning files [n0, n1); user friendly API for combine_datasets"""
    n1: int = n0 + num_files
    progbar: bool = False
    return combine_datasets_pos(n0=n0, n1=n1, include_vel=include_vel, progbar=progbar)

# ********************************************************************************************************************* 
# Datasets where the output is the asteroid direction from earth (ux, uy, uz)
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_dataset_dir_file(n0: int, n1: int) -> tf.data.Dataset:
    """
    Wrap the data in one file of asteroid trajectory data into a TF Dataset
    INPUTS:
        n0: the first asteroid in the file, e.g. 0
        n1: the last asteroid in the file (exclusive), e.g. 1000
    OUTPUTS:
        ds: a tf.data.Dataset object for this 
    """
    # Load data
    inputs_all, outputs_all = make_data_one_file(n0, n1)
    
    # Use all inputs
    inputs = inputs_all
    
    # Wrap up selected outputs
    outputs ={'u': outputs_all['u']}

    # Wrap into a dataset
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))    
    return ds

# ********************************************************************************************************************* 
def make_dataset_dir_n(n: int):
    """Convert the nth file, with asteroids (n-1)*1000 to n*1000, to a tf Dataset with direction from earth"""
    batch_size: int = 1000
    n0 = batch_size*n
    n1 = n0 + batch_size
    return make_dataset_dir_file(n0, n1)

# ********************************************************************************************************************* 
def combine_datasets_dir(n0: int, n1: int, batch_size: int = 64, progbar: bool = False):
    """Combine datasets in [n0, n1) into one large dataset"""
    # Iteration range for adding to the initial dataset
    ns = range(n0+1, n1)
    ns = tqdm(ns) if progbar else ns
    # Initial dataset
    ds = make_dataset_dir_n(n0)
    # Extend initial dataset
    for n in ns:
        try:
            ds_new = make_dataset_dir_n(n)
            ds = ds.concatenate(ds_new)
        except:
            pass
    # Batch the dataset
    drop_remainder: bool = True    
    return ds.batch(batch_size, drop_remainder=drop_remainder)

# ********************************************************************************************************************* 
def make_dataset_ast_dir(n0: int, num_files: int):
    """Create a dataset spanning files [n0, n1); user friendly API for combine_datasets"""
    n1: int = n0 + num_files
    progbar: bool = False
    return combine_datasets_dir(n0=n0, n1=n1, progbar=progbar)

# ********************************************************************************************************************* 
# Attempts to get the entire dataset into one object
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def gen_batches(n_max: int = 541, include_vel: bool = False, batch_size: int = 64):
    """Python generator for all the asteroid trajectories"""
    # round up n_max to the next multiple of 8
    n_max = ((n_max+7) // 8) * 8
    # iterate over the combined datasets
    for n in range(0, n_max, 8):
        ds = combine_datasets_pos(n0=n, n1=n+8, include_vel=include_vel, batch_size=batch_size)
        # iterate over the batches in the combined dataset
        for batch in ds.batch(batch_size):
            yield batch

# ********************************************************************************************************************* 
def make_dataset_ast_gen():
    """Wrap the asteroid data into a Dataset using Python generator"""
    # see ds.element_spec()
    output_types = (
    {'a': tf.float32,
     'e': tf.float32,
     'inc': tf.float32,
     'Omega': tf.float32,
     'omega': tf.float32,
     'f': tf.float32,
     'epoch': tf.float32,
     'ts': tf.float32,
    },
    {'q': tf.float32,
     'v': tf.float32,
    }
    )
    
    # output shapes: 6 orbital elements and epoch are scalars; ts is vector of length N; qs, vs, Nx3
    N: int = 14976
    batch_size: int = 64
    output_shapes = (
    {'a': (batch_size,),
     'e': (batch_size,),
     'inc': (batch_size,),
     'Omega': (batch_size,),
     'omega': (batch_size,),
     'f': (batch_size,),
     'epoch': (batch_size,),
     'ts':(batch_size,N,),
    },
    {'q': (batch_size,N,3),
     'v': (batch_size,N,3),
    }
    )

    # Set arguments for gen_batches
    n_max: int = 8
    include_vel: bool = False
    batch_args = (n_max, include_vel, batch_size)

    # Build dataset from generator
    ds = tf.data.Dataset.from_generator(
            generator=gen_batches,
            output_types=output_types,
            output_shapes=output_shapes,
            args=batch_args)

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
def serialize_ast_traj(ds: tf.data.Dataset):
    """Serialize one asteroid trajectory to a proto message"""
    # iterator for the dataset
    it = iter(ds)

    # Unpack inputs and outputs from iterator
    inputs, outputs = next(it)
    
    # Unpack inputs (orbital elements)
    a = inputs['a']
    e = inputs['e']
    inc = inputs['inc']
    Omega = inputs['Omega']
    omega = inputs['omega']
    f = inputs['f']
    epoch = inputs['epoch']
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
        'epoch': _float_feature(epoch),
        'ts': _float_vec_feature(ts),
        # q and v must be flattened
        'q': _float_vec_feature(tf.reshape(q, [-1])),
        'v': _float_vec_feature(tf.reshape(v, [-1])),
    }
    
    # create a message using tf.train.Example
    proto_msg = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto_msg.SerializeToString()

# ********************************************************************************************************************* 
def deserialize_ast_traj(proto_msg):
    """Deserialize an asteroid trajectory stored as a proto message"""
    pass


