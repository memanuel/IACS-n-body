"""
Harvard IACS Masters Thesis
Three Body Problem - Perturbed Sun-Earth-Jupiter System
Generate training data (trajectories)

Michael S. Emanuel
Tue Aug 20 16:40:29 2019
"""

# Library imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import rebound
import numpy as np
import pickle
from tqdm.auto import tqdm
import argparse
from typing import List

# Local imports
from g3b_data import make_traj_from_sim
from utils import hash_id_crc32

# Aliases
keras = tf.keras


# ********************************************************************************************************************* 
def make_sim_cart(m, q0, v0, sample_freq, integrator, debug_print):
    """
    Make a Rebound simulation for the 3 body problem using Cartesian position and velocity.
    INPUTS:
    m: masses of of 3 bodies in solar units; array of size 3
    q0: the starting positions of the 3 bodies in AU; array of size (3, 3,) = (num_body, space_dims)
    v0: the starting velocities of the 3 bodies in AU/yr; array of size (3, 3,)
    sample_freq: number of sample points per year, e.g. 365
    integrator: the name of the rebound integrator.  one of 'ias15' or 'whfast'
    RETURNS:
    sim: an instance of the simulator that is ready to integrate
    """
    # Unpack the position components
    qx = q0[:, 0]
    qy = q0[:, 1]
    qz = q0[:, 2]
    
    # Unpack the velocity components
    vx = v0[:, 0]
    vy = v0[:, 1]
    vz = v0[:, 2]
    
    # Create a simulation
    sim = rebound.Simulation()

    # Set units
    sim.units = ('yr', 'AU', 'Msun')

    # Set integrator.  if acceleration output required, must use ias15
    sim.integrator = integrator

    # Set the simulation time step based on sample_freq
    sim.dt = 1.0 / sample_freq

    # Add the 3 bodies in the provided order
    for i in range(3):
        sim.add(m=m[i], x=qx[i], y=qy[i], z=qz[i], vx=vx[i], vy=vy[i], vz=vz[i])
        if debug_print:
            print(f'particle {i}, m={m[i]}, x={qx[i]}, y={qy[i]}, z={qz[i]}, vx={vx[i]}, vy={vy[i]}, vz={vz[i]}')
    
    # Move to the center-of-momentum coordinate system
    sim.move_to_com()
    
    return sim   
            
# ********************************************************************************************************************* 
def make_traj_cart(m, q0, v0, n_years, sample_freq, integrator='ias15', debug_print = False):
    """
    Make an array of training data points from an initial configuration in Cartesian coordinates
    Creates one trajectory with the same initial configuration.
    INPUTS:
    m: masses of of 3 bodies in solar units; array of size 3
    q0: the starting positions of the 3 bodies in AU; array of size (3, 3,) = (num_body, space_dims)
    v0: the starting velocities of the 3 bodies in AU/yr; array of size (3, 3,)
    sample_freq: number of sample points per year, e.g. 365
    integrator: the name of the rebound integrator.  one of 'ias15' or 'whfast'
    NOTE:
    the orbital element inputs a, e, inc, Omega, omega, f are all arrays of size 3
    RETURNS:
    input_dict: a dictionary with the input fields t, q0, v0
    output_dict: a dictionary with the output fields q, v, a, q0_rec, v0_rec, T, H, H, L
    """
    # Build the simulation from the orbital elements
    sim = make_sim_cart(m=m, q0=q0, v0=v0,
                        sample_freq=sample_freq, integrator=integrator, debug_print=debug_print)
    
    # Create the trajectory from this simulation
    inputs, outputs = make_traj_from_sim(sim=sim, n_years=n_years, sample_freq=sample_freq, debug_print=debug_print)
    
    # Return the dicts
    return (inputs, outputs)

# ********************************************************************************************************************* 
def make_sim_horizons(object_names: List[str], horizon_date: str):
    """Create a new rebound simulation with initial data from the NASA Horizons system"""
    # Create a simulation
    sim = rebound.Simulation()
    
    # Set units
    sim.units = ('yr', 'AU', 'Msun')
    
    # Add these objects from Horizons
    sim.add(object_names, date=horizon_date)
          
    # Add hashes for the object names
    for i, particle in enumerate(sim.particles):
        particle.hash = rebound.hash(object_names[i])
        
    # Move to center of mass
    sim.move_to_com()
    
    return sim

# ********************************************************************************************************************* 
def make_archive(fname_archive, sim, n_years: int, sample_freq: int):
    """Create a rebound simulation archive and save it to disk"""
    # Number of samples including both start and end
    N = n_years*sample_freq + 1

    # Set the times for snapshots
    ts = np.linspace(0.0, n_years, N, dtype=np.float32)

    # Integrate the simulation up to tmax
    for i, t in tqdm(enumerate(ts)):
        # Integrate to the current time step with an exact finish time
        sim.integrate(t, exact_finish_time=1)
        # Save a snapshot to the archive file
        sim.simulationarchive_snapshot(filename=fname_archive)

    # print(f'Created simulation archive in {fname_archive} in {elapsed} seconds.') 

    # Load the updated simulation archive
    sa = rebound.SimulationArchive(fname_archive)
    
    # Return the simulation archive
    return sa

# ********************************************************************************************************************* 
def make_sa_sej(n_years: int, sample_freq:int ):
    """Create or load the sun-earth-jupiter system at start of J2000.0 frame"""
    
    # The name of the simulation archive
    fname_archive = '../data/sej/ss_sej.bin'
    
    # If this file already exists, load and return it
    try:
        sa = rebound.SimulationArchive(fname_archive)
        # print(f'Found simulation archive {fname_archive}')
    except:
        # Initialize a new simulation
        object_names = ['Sun', 'Earth', 'Jupiter']
        horizon_date = '2000-01-01 12:00'
        sim = make_sim_horizons(object_names=object_names, horizon_date=horizon_date)
        
        # Create a simulation archive from this simulation
        sa = make_archive(fname_archive=fname_archive, sim=sim, 
                          n_years=n_years, sample_freq=sample_freq)
    
    return sa

# ********************************************************************************************************************* 
def make_data_sej(n_traj: int, n_years: int, sample_freq: int,
                  sd_q: np.array, sd_v: np.array, seed: int):
    """
    Build a data set of perturbed instances of the sun-earth-jupiter system.
    INPUTS:
        n_traj: the number of trajectories to simulate
        n_years: the number of years in each trajectory
        sample_freq: the number of sample points per year
        sd_q: the standard deviation of perturbations applied to position qx, qy, and qz; array of shape (2,)
        sd_v: the standard deviation of perturbations applied to velocity vx, vy, and vz; array of shape (2,)
        seed: the random seed
    OUTPUTS:
        inputs: a dictionary of numpy arrays with inputs to the model
                keys are t, q0, v0, m
        outputs: a dictionary of numpy arrays with outputs to the model
                keys are q, v, a, 
                orb_a, orb_e, orb_inc, orb_Omega, orb_omega, orb_f,
                q0_rec, v0_rec, T, U, H, P, L
    """
    
    # Simulation archive for unperperturbed system
    sa = make_sa_sej(n_years, sample_freq)
    # Start of the unperturbed simulation
    sim0 = sa[0]
    # Extract particles for sun, earth and jupiter
    ps = sim0.particles
    p0, p1, p2 = ps
    
    # Unpack the masses
    mu = np.array([p0.m, p1.m, p2.m], dtype=np.float32)
    
    # Unpack the unperturbed position
    qu_0 = np.array([p0.x, p0.y, p0.z], dtype=np.float32)
    qu_1 = np.array([p1.x, p1.y, p1.z], dtype=np.float32)
    qu_2 = np.array([p2.x, p2.y, p2.z], dtype=np.float32)
    qu = np.array([qu_0, qu_1, qu_2])

    # Unpack the unperturbed velocity
    vu_0 = np.array([p0.vx, p0.vy, p0.vz], dtype=np.float32)
    vu_1 = np.array([p1.vx, p1.vy, p1.vz], dtype=np.float32)
    vu_2 = np.array([p2.vx, p2.vy, p2.vz], dtype=np.float32)
    vu = np.array([vu_0, vu_1, vu_2])

    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Number of particles and spatial dimensions
    num_particles = 3
    space_dims = 3

    # If sd_q and sd_v were passed as scalars, convert to arrays of size 3
    if type(sd_q) is float:
        sd_q = np.repeat(sd_q, 3)
    if type(sd_v) is float:
        sd_v = np.repeat(sd_v, 3)        

    # Initialize the shift to initial position and velocity by sampling according to the inputs
    shift_size = (n_traj, num_particles, space_dims)
    shift_comp_size = (n_traj, space_dims)
    delta_q = np.zeros(shape=shift_size, dtype=np.float32)
    delta_v = np.zeros(shape=shift_size, dtype=np.float32) 
    # Iterate over particle j to be shifted; indices are (traj_num, particle_num, space_dim)
    for j in range(3):
        delta_q[:, j, :] = np.random.normal(loc=0.0, scale=sd_q[j], size=shift_comp_size).astype(np.float32)
        delta_v[:, j, :] = np.random.normal(loc=0.0, scale=sd_v[j], size=shift_comp_size).astype(np.float32)

    # Compute perturbed position and velocity
    qp = qu + delta_q
    vp = vu + delta_v

    # Number of samples including both start and end in each trajectory
    traj_size = sample_freq*n_years + 1
    
    # The integrator to use
    integrator = 'ias15'

    # Shape of arrays for various inputs and outputs
    time_shape = (n_traj, traj_size)
    init_shape = (n_traj, num_particles, space_dims)
    mass_shape = (n_traj, num_particles)
    traj_shape = (n_traj, traj_size, num_particles, space_dims)
    elt_shape = (n_traj, traj_size, num_particles-1)
    mom_shape = (n_traj, traj_size, space_dims)
    
    # Initialize arrays for the data
    # Inputs
    t = np.zeros(time_shape, dtype=np.float32)
    q0 = np.zeros(init_shape, dtype=np.float32)
    v0 = np.zeros(init_shape, dtype=np.float32)
    m = np.zeros(mass_shape, dtype=np.float32)
    # Outputs - trajectory
    q = np.zeros(traj_shape, dtype=np.float32)
    v = np.zeros(traj_shape, dtype=np.float32)
    a = np.zeros(traj_shape, dtype=np.float32)
    # Outputs - orbital elements
    orb_a = np.zeros(elt_shape, dtype=np.float32)
    orb_e = np.zeros(elt_shape, dtype=np.float32)
    orb_inc = np.zeros(elt_shape, dtype=np.float32)
    orb_Omega = np.zeros(elt_shape, dtype=np.float32)
    orb_omega = np.zeros(elt_shape, dtype=np.float32)
    orb_f = np.zeros(elt_shape, dtype=np.float32)
    # Outputs- physics
    T = np.zeros(time_shape, dtype=np.float32)
    U = np.zeros(time_shape, dtype=np.float32)
    H = np.zeros(time_shape, dtype=np.float32)
    P = np.zeros(mom_shape, dtype=np.float32)
    L = np.zeros(mom_shape, dtype=np.float32)

    # Sample the trajectories
    for i in tqdm(range(n_traj)):
        # Generate one trajectory
        inputs_traj, outputs_traj = \
            make_traj_cart(m=mu, q0=qp[i], v0=vp[i], n_years=n_years, sample_freq=sample_freq, 
                           integrator=integrator, debug_print = False)

        # Copy results into main arrays
        # Inputs
        t[i] = inputs_traj['t']
        q0[i] = inputs_traj['q0']
        v0[i] = inputs_traj['v0']
        m[i] = inputs_traj['m']
        # Outputs - trajectory
        q[i] = outputs_traj['q']
        v[i] = outputs_traj['v']
        a[i] = outputs_traj['a']
        # Outputs - orbital elements
        orb_a[i] = outputs_traj['orb_a']
        orb_e[i] = outputs_traj['orb_e']
        orb_inc[i] = outputs_traj['orb_inc']
        orb_Omega[i] = outputs_traj['orb_Omega']
        orb_omega[i] = outputs_traj['orb_omega']
        orb_f[i] = outputs_traj['orb_f']
        # Outputs - physics
        T[i] = outputs_traj['T']
        U[i] = outputs_traj['U']
        H[i] = outputs_traj['H']
        P[i] = outputs_traj['P']
        L[i] = outputs_traj['L']
        
    # Assemble the input dict
    inputs = {
        't': t,
        'q0': q0,
        'v0': v0,
        'm': m,
        }

    # Assemble the output dict
    outputs = {
        # Trajectory
        'q': q,
        'v': v,
        'a': a,
        # Orbital Elements
        'orb_a': orb_a,
        'orb_e': orb_e,
        'orb_inc': orb_inc,
        'orb_Omega': orb_Omega,
        'orb_omega': orb_omega,
        'orb_f': orb_f,
        # Recovered initial configuration        
        'q0_rec': q0,
        'v0_rec': v0,
        # Physics quantities
        'T': T,
        'U': U,
        'H': H,
        'P': P,
        'L': L}

    # Return the dicts
    return (inputs, outputs)

# ********************************************************************************************************************* 
def make_filename_sej(n_traj: int, vt_split: float, n_years: int, sample_freq: int, 
                      sd_q: np.array, sd_v: np.array, seed: int):
    """Make file name for serializing datasets for the perturbed sen-earth-jupiter system"""
    
    # Create dictionary with attributes
    attributes = {
        'n_traj': n_traj,
        'vt_split': vt_split,
        'n_years': n_years,
        'sample_freq': sample_freq,
        'sd_q': sd_q,
        'sd_v': sd_v,
        'seed': seed,
        }

    # Create a non-negative hash ID of the attributes
    # attributes_bytes = bytes(str(attributes), 'utf-8')
    # hash_id = zlib.crc32(attributes_bytes)
    hash_id = hash_id_crc32(attributes)

    # Create the filename
    return f'../data/sej/{hash_id}.pickle'

# ********************************************************************************************************************* 
def load_data_sej(n_traj: int, vt_split: float, n_years: int, sample_freq: int, 
                  sd_q: np.array, sd_v: np.array, seed: int):
    """Load data for the perturbed sun-earth-jupiter problem for train, val and test"""
    # Get the filename for these arguments
    filename = make_filename_sej(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                                 sd_q=sd_q, sd_v=sd_v, seed=seed)
    # Attempt to load the file
    try:
        with open(filename, 'rb') as fh:
            vartbl = pickle.load(fh)
            inputs_trn = vartbl['inputs_trn']
            outputs_trn = vartbl['outputs_trn']
            inputs_val = vartbl['inputs_val']
            outputs_val = vartbl['outputs_val']
            inputs_tst = vartbl['inputs_tst']
            outputs_tst = vartbl['outputs_tst']
            print(f'Loaded data from {filename}.')
            data  = (inputs_trn, outputs_trn, inputs_val, outputs_val, inputs_tst, outputs_tst)
    # Generate the data and save it to the file
    except:
        # Status 
        print(f'Unable to load data from {filename}.')
        data = None

    # Return the data
    return data

# ********************************************************************************************************************* 
def make_datasets_sej(n_traj: int, vt_split: float, n_years: int, sample_freq: int,
                      sd_q: np.array, sd_v: np.array, seed: int, batch_size: int,
                      assemble_datasets: bool = True):
    """Make datasets for the perturbed sen-earth-jupiter problem for train, val and test"""

    # Attempt to load the data
    data = load_data_sej(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                         sd_q=sd_q, sd_v=sd_v, seed=seed)

    # Unpack data if available
    if data is not None:
        inputs_trn, outputs_trn, inputs_val, outputs_val, inputs_tst, outputs_tst = data
    # Otherwise generate the trajectories using make_data_sej
    else:
        # Set the number of trajectories for train, validation and test
        n_traj_trn = n_traj
        n_traj_val = int(n_traj * vt_split)
        n_traj_tst = n_traj_val

        # Set the random seeds
        seed_trn = seed + 0
        seed_val = seed + 1
        seed_tst = seed + 2

        # Generate inputs and outputs for orbits with input parameters
        inputs_trn, outputs_trn = make_data_sej(n_traj=n_traj_trn, n_years=n_years, sample_freq=sample_freq,
                                                sd_q=sd_q, sd_v=sd_v, seed=seed_trn)
        inputs_val, outputs_val = make_data_sej(n_traj=n_traj_val, n_years=n_years, sample_freq=sample_freq,
                                                sd_q=sd_q, sd_v=sd_v, seed=seed_val)
        inputs_tst, outputs_tst = make_data_sej(n_traj=n_traj_tst, n_years=n_years, sample_freq=sample_freq,
                                                sd_q=sd_q, sd_v=sd_v, seed=seed_tst)
        
        # Get the filename for these arguments
        filename = make_filename_sej(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                                     sd_q=sd_q, sd_v=sd_v, seed=seed)

        # Save these to file
        vartbl = dict()
        vartbl['inputs_trn'] = inputs_trn
        vartbl['outputs_trn'] = outputs_trn
        vartbl['inputs_val'] = inputs_val
        vartbl['outputs_val'] = outputs_val
        vartbl['inputs_tst'] = inputs_tst
        vartbl['outputs_tst'] = outputs_tst
        with open(filename, 'wb') as fh:
            pickle.dump(vartbl, fh)

    # Create DataSet objects for train, val and test sets if the assemble_dataset flag was passed
    if assemble_datasets:
        ds_trn = tf.data.Dataset.from_tensor_slices((inputs_trn, outputs_trn))
        ds_val = tf.data.Dataset.from_tensor_slices((inputs_val, outputs_val))
        ds_tst = tf.data.Dataset.from_tensor_slices((inputs_tst, outputs_tst))
        
        # Set shuffle buffer size
        buffer_size = batch_size * 256
    
        # Shuffle and batch data sets
        drop_remainder = True
        ds_trn = ds_trn.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size, drop_remainder=drop_remainder)
        ds_val = ds_val.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size, drop_remainder=drop_remainder)
        ds_tst = ds_tst.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size, drop_remainder=drop_remainder)
        
        data = ds_trn, ds_val, ds_tst
    else:
        data = None
    
    return data

# ********************************************************************************************************************* 
def combine_datasets_sej_impl(n_traj: int, vt_split: float, n_years: int, sample_freq: int, 
                              sd_q: np.array, sd_v: np.array, seeds: List[int], batch_size: int):
    """Combine a collection of SEJ data sets into one large data set."""
    
    # First dataset
    seed = seeds[0]
    # print(f'i=1 , seed={seed:3} ', end=None)
    ds_trn, ds_val, ds_tst = make_datasets_sej(
           n_traj=n_traj, vt_split=vt_split, 
           n_years=n_years, sample_freq=sample_freq,
           sd_q=sd_q, sd_v=sd_v, 
           seed=seed, batch_size=batch_size)
    # Concatenate remaining datasets
    for seed in tqdm(list(seeds[1:])):
        # Status update
        # print(f'i={i+1:2}, seed={seed:3} ', end=None)
        # The new batch of datasets
        ds_trn_new, ds_val_new, ds_tst_new = make_datasets_sej(
               n_traj=n_traj, vt_split=vt_split, 
               n_years=n_years, sample_freq=sample_freq,
               sd_q=sd_q, sd_v=sd_v, 
               seed=seed, batch_size=batch_size)
        # Concatenate the new datasets
        ds_trn = ds_trn.concatenate(ds_trn_new)
        ds_val = ds_val.concatenate(ds_val_new)
        ds_tst = ds_tst.concatenate(ds_tst_new)        
        
    # Return the three large concatenated datasets
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def make_perturbations(scale_factor: float):
    """Make default orbital perturbations given a scaling factor"""
    # Orbital perturbation scales on sun, earth and jupiter respectively
    sd_q = scale_factor * np.array([0.00, 0.01, 0.05])
    sd_v = scale_factor * np.array([0.00, 0.001, 0.0005])
    
    return sd_q, sd_v
    
# ********************************************************************************************************************* 
def combine_datasets_sej(num_data_sets: int, batch_size: int, seed0: int, scale_factor: float):
    """Combine a collection of SEJ data sets into one large data set."""
    # Number of trajectories in each constituent batch
    n_traj = 10000
    vt_split = 0.20

    # Time parameters
    n_years=100
    sample_freq=10

    # Orbital perturbation scales on sun, earth and jupiter respectively
    sd_q, sd_v = make_perturbations(scale_factor)
    
    # List of random seeds
    seeds = list(range(seed0, seed0+3*num_data_sets, 3))
    
    # Status update
    # print(f'Loading {num_data_sets} SEJ data sets with:\n', sej_sigma)
    
    # Delegate to conbine_datasets_g3b
    ds_trn, ds_val, ds_tst = combine_datasets_sej_impl(
        n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq,
        sd_q=sd_q, sd_v=sd_v,
        seeds=seeds, batch_size=batch_size)
    
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def orb_elts0(ds: tf.data.Dataset):
    """Get the initial orbital elements in a dataset"""
    # Get array with the starting values of the six orbital elements
    orb_a = np.concatenate([data[1]['orb_a'][:, 0, :] for i, data in ds.enumerate()], axis=0)
    orb_e = np.concatenate([data[1]['orb_e'][:, 0, :] for i, data in ds.enumerate()], axis=0)
    orb_inc = np.concatenate([data[1]['orb_inc'][:, 0, :] for i, data in ds.enumerate()], axis=0)
    orb_Omega = np.concatenate([data[1]['orb_Omega'][:, 0, :] for i, data in ds.enumerate()], axis=0)
    orb_omega = np.concatenate([data[1]['orb_omega'][:, 0, :] for i, data in ds.enumerate()], axis=0)
    orb_f = np.concatenate([data[1]['orb_f'][:, 0, :] for i, data in ds.enumerate()], axis=0)
    # H = np.concatenate([data[1]['H'][:] for i, data in ds.enumerate()], axis=0)
    
    # Combined orbital elements; array of shape num_trajectories, 12
    orb_elt = np.concatenate([orb_a, orb_e, orb_inc, orb_Omega, orb_omega, orb_f], axis=1)
    
    return orb_elt

# ********************************************************************************************************************* 
def orb_elt_summary(orb_elt):
    """Print a summary of the initial orbital elements in a data set"""
    # List of orbital elements in the cov matrix
    elt_names = ['a1', 'a2', 'e1', 'e2', 'inc1', 'inc2', 
                 'Omega1', 'Omega2', 'omega1', 'omega2', 'f1', 'f2']
    # limit to the interesting ones
    elt_names = elt_names[0:6]
    
    # Compute mean, std, min and max of orbital elemetns
    elt_mean = np.mean(orb_elt, axis=0)
    elt_std = np.std(orb_elt, axis=0)
    elt_min = np.min(orb_elt, axis=0)
    elt_max = np.max(orb_elt, axis=0)    
    
    # Display summary statistics of orbital elements
    print(f'element:  mean     :  std dev  :  min      :  max')
    for i, elt_name in enumerate(elt_names):
        print(f'{elt_name:6} : {elt_mean[i]:9.6f} : {elt_std[i]:9.6f} : '
              f'{elt_min[i]:9.6f} : {elt_max[i]:9.6f}')

# ********************************************************************************************************************* 
def main():
    """Main routine for making SEJ data sets"""
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Generate data for perturbed Sun-Earth-Jupiter system.')
    parser.add_argument('-num_batches', metavar='B', type=int, default=5,
                        help='the number of batches to run')
    parser.add_argument('-n_traj', metavar='N', type=int, default=10000,
                        help='the number of trajectories')
    parser.add_argument('-scale_factor', metavar='F', type=float, default=1.0,
                        help='scale factor that multiplies base perturbation size')
    parser.add_argument('-seed', metavar='S', type=int, default=42,
                        help='the first seed for the random number generator')
    args = parser.parse_args()
    
    # Unpack command line arguments
    num_batches = args.num_batches
    n_traj = args.n_traj
    seed0 = args.seed
    scale_factor = args.scale_factor

    # Status
    print(f'\nGenerating {num_batches} batches of {n_traj} trajectories each.')
    print(f'Scale factor = {scale_factor}.')
    print(f'Initial random seed = {seed0}.')
    
    # Length and sample frequency
    n_years = 100
    sample_freq = 10
    
    # Make the simulation archive for the unperturbed (real) sun-earth-jupiter system
    make_sa_sej(n_years=n_years, sample_freq=sample_freq)
    
    # Batch size and number of train and validation samples
    batch_size = 256
    vt_split = 0.20
    
    # Orbital perturbation scales on sun, earth and jupiter respectively
    sd_q, sd_v = make_perturbations(scale_factor)
    
    # List of seeds to use for datasets
    seed1 = seed0 + num_batches * 3
    seeds = list(range(seed0, seed1, 3))
    
    # Set assemble_datasets flag to false; don't need datasets,
    # only want to save the numpy arrays to disk
    assemble_datasets = False
    
    # Run perturbed simulation
    for i, seed in tqdm(enumerate(seeds)):
        make_datasets_sej(n_traj=n_traj, vt_split=vt_split, 
                          n_years=n_years, sample_freq=sample_freq,
                          sd_q=sd_q, sd_v=sd_v,
                          seed=seed, batch_size=batch_size, 
                          assemble_datasets=assemble_datasets)
# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
