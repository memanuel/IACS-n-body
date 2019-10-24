"""
Harvard IACS Masters Thesis
observation_data.py: Synthetic observations of asteroid directions based on their orbits.

Michael S. Emanuel
Thu Oct 17 09:20:20 2019
"""

# Library imports
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
from tqdm.auto import tqdm

# Local imports
from utils import range_inc
from astro_utils import datetime_to_mjd
from asteroid_data import make_data_one_file, get_earth_pos_file
from asteroid_integrate import load_data as load_data_asteroids

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()
ast_num_all = ast_elt.Num.to_numpy()

# Load earth position data
q_earth, ts = get_earth_pos_file()
space_dims = 3

# ********************************************************************************************************************* 
def make_synthetic_obs_data(n0: int, n1: int, dt0: datetime, dt1: datetime, 
                            p0: float, p1: float, noise: float, 
                            frac_real: float):
    """
    Genereate synthetic data of asteroid observations.
    INPUTS:
        n0: asteroid number of first asteroid to consider; inclusive
        n1: asteroid number of last asteroid to consider; exclusive
        dt0: starting date range for observations
        dt1: ending date range for observations
        p0: probability of an observation for asteroid n0
        p1: probability of an observation for asteroid n1 (interpolates from p0 to p1)
        noise: standard deviation of noise applied to observations
        frac_real: the fraction of the observations that are real; rest are uniformly distributed bogus observations
    OUTPUTS:
        t: mjd of this observation
        u: asteroid direction (ux, uy, uz)
        ast_num: asteroid number; 0 for bogus observations
    """
    # Seed RNG for reproducible results
    np.random.seed(seed=42)
    
    # Data type for floating point
    dtype = np.float32

    # Range of MJDs
    t0: float = datetime_to_mjd(dt0)
    t1: float = datetime_to_mjd(dt1)
    
    # Preallocate storage
    num_obs_max: int = int((n1-n0) * (t1-t0) / (1.0 - frac_real))
    t: np.array = np.zeros(num_obs_max, dtype=dtype)
    u: np.array = np.zeros((num_obs_max, space_dims), dtype=dtype)
    ast_num_obs: np.array = np.zeros(num_obs_max, dtype=np.int32)
    
    # Load data pages for selected asteroids
    page_size: int = 1000
    page0: int = n0 // page_size
    page1: int = (n1-1) // page_size
    # Initialize counter for number of observations
    num_obs: int = 0
    # Iterate through pages of asteroid data
    page: int
    for page in tqdm(range_inc(page0, page1)):
        # Load file for this page
        n0_file: int = page * page_size
        n1_file: int = n0_file + page_size
        inputs, outputs = make_data_one_file(n0=n0_file, n1=n1_file)
        # print(f'Loaded file for asteroids {n0_file} to {n1_file}.')

        # Asteroid numbers on this page
        mask_page: np.array = (n0_file < ast_num_all) & (ast_num_all < n1_file)
        ast_num_page: np.array = ast_num_all[mask_page]
        # print(f'ast_num_page={ast_num_page}')
        
        # Extract positions; times of asteroid positions synchronized to those for earth
        q: np.array = outputs['q']

        # Iterate through asteroids in this page
        idx: int
        ast_num_i: int
        # print(f'n0={n0}, n1={n1}')
        for idx, ast_num_i in enumerate(ast_num_page):
            # Only process asteroids in [n0, n1)
            if not (n0 <= ast_num_i and ast_num_i < n1):
                continue
            # Interpolate the observation acceptance probability
            p: float = np.interp(ast_num_i, [n0, n1], [p0, p1])
            # print(f'ast_num_i={ast_num_i}, p={p}')
            # Direction and observation times for this asteroid
            ts_obs, u_obs = get_synthetic_data_one_asteroid(
                            idx=idx, t0=t0, t1=t1, p=p, noise=noise, q=q, ts=ts)
            # Number of observations generated; end index of this data slice
            ni: int = len(ts_obs)
            slice_end = num_obs + ni
            # print(f'ni={ni}')
            # Copy this page into t and u
            t[num_obs:slice_end] = ts_obs
            u[num_obs:slice_end] = u_obs
            # The observed asteroid number is the same 
            ast_num_obs[num_obs:slice_end] = ast_num_i
            # Update num_obs for this asteroid
            num_obs += ni
    
    # Add bogus data points
    num_real: int = num_obs
    num_total: int = int(np.round(num_real / frac_real))
    num_bogus: int = num_total - num_real
    ts_bogus = np.arange(t0, t1)
    t[num_real:num_total] = np.random.choice(ts_bogus, size=num_bogus)
    u[num_real:num_total] = random_direction(num_bogus)
    # Use placeholder asteroid number = 0 for bogus observations
    ast_num_obs[num_real:num_total] = 0
    
    # Prune array size down and sort it by time t
    idx = np.argsort(t[0:num_total])
    t = t[idx]
    u = u[idx]
    ast_num_obs = ast_num_obs[idx]
    
    return t, u, ast_num_obs

# ********************************************************************************************************************* 
def get_synthetic_data_one_asteroid(idx: int, t0: float, t1: float, p: float, noise: float,
                                    q: np.array, ts: np.array):
    """
    Get synthetic observation data for one asteroid
    INPUTS:
        idx: Index number of desired asteroid in this data page
        t0: Start time for observation (MJD, inclusive)
        t1: End time for observation (MJD, exclusive)
        p: Probability a given candidate observation is made
        noise: Standard deviation of noise applied to observations
        q: Position of asteroids on this data page
        ts: Time of asteroid position snapshots on this data page
    OUTUTS:
        ts_obs: observation times
        u_obs: observed direction of asteroid
    """
    # mask for select candidate observation times
    time_mask = (t0 <= ts) & (ts < t1)
    
    # observation time and position data for all possible observations
    ts_cand = ts[time_mask]
    q_earth_cand = q_earth[time_mask]
    q_ast_cand = q[idx][time_mask]
    
    # unit vector (direction) from earth to asteroid on all candidate dates
    q_rel = q_ast_cand - q_earth_cand
    r_earth = np.linalg.norm(q_rel, axis=1, keepdims=True)
    u_cand = q_rel / r_earth
    
    # apply random noise
    sigma = noise / np.sqrt(space_dims)
    u_cand += np.random.normal(scale=sigma, size=u_cand.shape)
    # renormalize u
    u_cand /= np.linalg.norm(u_cand, axis=1, keepdims=True)
    
    # mask for randomly sampling the candidate observations
    mask_picked = (np.random.uniform(size=len(u_cand)) < p)

    # the observation times and directions are the candidates that have been picked
    ts_obs = ts_cand[mask_picked]
    u_obs = u_cand[mask_picked]
    return ts_obs, u_obs

# ********************************************************************************************************************* 
def random_direction(num_bogus: int):
    """Return a random direction on the unit sphere as a 3D vector u."""
    # See http://mathworld.wolfram.com/SpherePointPicking.html
    # or http://corysimon.github.io/articles/uniformdistn-on-sphere/
    theta: np.array = np.random.uniform(low=0.0, high=2.0*np.pi, size=num_bogus)
    v: np.array = np.random.uniform(low=0.0, high=1.0, size=num_bogus)
    phi: np.array = np.arccos(2.0 * v - 1.0)
    u = np.zeros((num_bogus,space_dims))
    r = np.sin(phi)
    # x, y and z coordinates respectively
    u[:, 0] = r * np.cos(theta)
    u[:, 1] = r * np.sin(theta)
    u[:, 2] = np.cos(phi)
    return u

# ********************************************************************************************************************* 
def load_synthetic_obs_data(n0: int, n1: int):
    """Load synthetic observation data"""
    # File name for this data set
    fname: str = f'../data/observations/synthetic_n_{n0:06}_{n1:06}.npz'
    # Load numpy data and unpack into variables
    data = np.load(fname)
    t = data['t']
    u = data['u']
    ast_num = data['ast_num']
    return t, u, ast_num

# ********************************************************************************************************************* 
def run_batch(n0: int, n1: int) -> None:
    """Run one batch of synthetic asteroid data"""
    # Date range for simulation
    dt0 = datetime(2000,1,1)
    dt1 = datetime(2019,10,1)
    # Acceptance probability of observations; interpolate linearly from n0 to n1
    p0: float = 1.0
    p1: float = 0.10
    # Noise to add to observations; distance on the unit sphere
    noise_arcsec: float = 1.0
    noise: float = np.deg2rad(noise_arcsec / 3600)
    # Fraction of real observations
    frac_real: float = 0.90
    
    # File name for this data set
    fname: str = f'../data/observations/synthetic_n_{n0:06}_{n1:06}.npz'

    # Quit early if file already exists
    if os.path.isfile(fname):
        print(f'File {fname} already exists.')
        return

    # Generate the synthetic data if file does not exists
    print(f'Generating synthetic observation data for asteroids {n0} to {n1} with p0={p0}, p1={p1}...')
    t, u, ast_num = make_synthetic_obs_data(n0=n0, n1=n1, dt0=dt0, dt1=dt1, p0=p0, p1=p1, 
                                            noise=noise, frac_real=frac_real)
    
    # Save to numpy
    np.savez(fname, t=t, u=u, ast_num=ast_num)

# ********************************************************************************************************************* 
def make_ragged_tensors(t_np: np.array, u_np: np.array, ast_num_np: np.array):
    """
    Convert t, u, ast_num into ragged tensors
    INPUTS:
        t_np: A numpy array with observation times as MJDs; size (N,)
        u_np: A numpy array with directions as MJDs; size (N,3)
        ast_num_np: A numpy array with the asteroid numbers; size (N,)
    """
    # Unique times and their indices
    t_unq, inv_idx, = np.unique(t_np, return_inverse=True)

    # The row IDs for the ragged tensorflow are what numpy calls the inverse indices    
    value_rowids = inv_idx    

    # Tensor with distinct times
    t = tf.convert_to_tensor(value=t_unq)
    # Ragged tensors for direction u and asteroid number ast_num
    u = tf.RaggedTensor.from_value_rowids(values=u_np, value_rowids=inv_idx)
    ast_num = tf.RaggedTensor.from_value_rowids(values=ast_num_np, value_rowids=value_rowids)

    # Return the tensors for t, u, ast_num
    return t, u, ast_num

# ********************************************************************************************************************* 
def make_synthetic_obs_dataset(n0: int, n1: int):
    """Create a tf.Dataset with synthetic data"""
    # Load data up asteroid number n1
    t_np, u_np, ast_num_np = load_synthetic_obs_data(n0=n0, n1=n1)
    
    # Convert to tensors (regular for t, ragged for u and ast_num)
    t, u_r, ast_num_r = make_ragged_tensors(t_np=t_np, u_np=u_np, ast_num_np=ast_num_np)
    
    # Get row lengths
    row_len = u_r.row_lengths()
    
    # Pad u_r into a regular tensor
    pad_default = np.array([0.0, 0.0, 65536.0])
    u = u_r.to_tensor(default_value=pad_default)
    
    # Alias array of times to ts; it is a separate output
    ts = t
    
    # Pad ast_num_r into a regular tensor; use default -1 to indicate padded observation
    ast_num = ast_num_r.to_tensor(default_value=-1)
    
    # Wrap into tensorflow Dataset
    inputs = {
        't' : t, 
        'u': u, 
        'row_len': row_len,
    }
    outputs = {
        'ast_num': ast_num,
    }
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))
    
    # Return the dataset as well as the time array
    return ds, ts

# ********************************************************************************************************************* 
def make_synthetic_obs_tensors(n0: int, n1: int):
    """Create tensors t, u, ast_num for the specified asteroid numbers"""
    # Load data up asteroid number n1
    t_np, u_np, ast_num_np = load_synthetic_obs_data(n0=n0, n1=n1)
    
    # Convert to tensors (regular for t, ragged for u and ast_num)
    t, u, ast_num = make_ragged_tensors(t_np=t_np, u_np=u_np, ast_num_np=ast_num_np)
    
    return t, u, ast_num

# ********************************************************************************************************************* 
def main():
    # Simulate a few different size ranges with default parameters
    run_batch(1, 10)
    run_batch(1, 100)
    run_batch(1, 1000)
    run_batch(1, 10000)
    run_batch(1, 100000)

# ********************************************************************************************************************* 
def test_ragged_tensors():
    t_np, u_np, ast_num_np = load_synthetic_obs_data(n1=10)
    t, u, ast_num = make_synthetic_obs_tensors(t_np, u_np, ast_num_np)

# ********************************************************************************************************************* 
def test_synthetic_dataset():
    n0: int = 1
    n1: int = 10
    make_synthetic_obs_dataset(n0=n0, n1=n1)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()

# test_synthetic_dataset()

ds, ts = make_synthetic_obs_dataset(1, 10)
