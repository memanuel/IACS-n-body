"""
Harvard IACS Masters Thesis
Three Body Problem - Perturbed Sun-Earth-Jupiter System
Generate training data (trajectories)

Michael S. Emanuel
Tue Aug 20 16:40:29 2019
"""

# Library imports
import tensorflow as tf
import rebound
import numpy as np
import pickle
from tqdm.auto import tqdm
import warnings
from typing import List

# Local imports
from g3b_data import make_traj_g3b
from utils import hash_id_crc32

# Aliases
keras = tf.keras

# Suppress the annoying and verbose tensorflow warnings
warnings.simplefilter("ignore")

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
                  sd_log_a: float, sd_log_e: float, sd_log_inc: float,
                  sd_Omega: float, sd_omega: float, sd_f: float,
                  seed: int):
    """
    Build a data set of perturbed instances of the sun-earth-jupiter system.
    INPUTS:
        n_traj: the number of trajectories to simulate
        n_years: the number of years in each trajectory
        sample_freq: the number of sample points per year
        sd_a: the standard deviation of log(a)
        sd_log_e: the standard deviation of log(e)
        sd_log_inc: the standard deviation of log(inc)
        sd_Omega: the standard deviation of Omega
        sd_omega: the standard deviation of omega
        sd_f: the standard deviation of f
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
    # Start of the simulation
    sim = sa[0]
    # Extract particles for sun, earth and jupiter
    ps = sim.particles
    p0, p1, p2 = ps
    
    # Unpack the masses
    m0 = np.array([p0.m, p1.m, p2.m], dtype=np.float32)
    
    # Unpack the unperturbed orbital elements
    a0 = np.array([p1.a, p2.a], dtype=np.float32)
    e0 = np.array([p1.e, p2.e], dtype=np.float32)
    inc0 = np.array([p1.inc, p2.inc], dtype=np.float32)
    Omega0 = np.array([p1.Omega, p2.Omega], dtype=np.float32)
    omega0 = np.array([p1.omega, p2.omega], dtype=np.float32)
    f0 = np.array([p1.f, p2.f], dtype=np.float32)

    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Initialize the shift to orbital element by sampling according to the inputs
    elt_size = (n_traj, 2)
    delta_log_a = np.random.normal(loc=0.0, scale=sd_log_a, size=elt_size).astype(np.float32)
    delta_log_e = np.random.normal(loc=0.0, scale=sd_log_e, size=elt_size).astype(np.float32)
    delta_log_inc = np.random.normal(loc=0.0, scale=sd_log_inc, size=elt_size).astype(np.float32)
    delta_Omega = np.random.normal(loc=0.0, scale=sd_Omega, size=elt_size).astype(np.float32)
    delta_omega = np.random.normal(loc=0.0, scale=sd_omega, size=elt_size).astype(np.float32)
    delta_f = np.random.normal(loc=0.0, scale=sd_f, size=elt_size).astype(np.float32)

    # Compute perturbed orbital elements
    orb_a0 = a0 * np.exp(delta_log_a)
    orb_e0 = e0 * np.exp(delta_log_e)
    orb_inc0 = inc0 * np.exp(delta_log_inc)
    orb_Omega0 = Omega0 + delta_Omega
    orb_omega0 = omega0 + delta_omega
    orb_f0 = f0 + delta_f

    # Number of samples including both start and end in each trajectory
    traj_size = sample_freq*n_years + 1
    
    # The integrator to use
    integrator = 'ias15'

    # Number of particles and spatial dimensions
    num_particles = 3
    space_dims = 3

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
        inputs_traj, outputs_traj = make_traj_g3b(m=m0, 
                                                  a=orb_a0[i], e=orb_e0[i], inc=orb_inc0[i], 
                                                  Omega=orb_Omega0[i], omega=orb_omega0[i], f=orb_f0[i], 
                                                  n_years=n_years, sample_freq=sample_freq, 
                                                  integrator=integrator)

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
                      sd_log_a: float, sd_log_e: float, sd_log_inc: float,
                      sd_Omega: float, sd_omega: float, sd_f: float,
                      seed: int):
    """Make file name for serializing datasets for the perturbed sen-earth-jupiter system"""
    
    # Create dictionary with attributes
    attributes = {
        'n_traj': n_traj,
        'vt_split': vt_split,
        'n_years': n_years,
        'sample_freq': sample_freq,
        'sd_log_a': sd_log_a,
        'sd_log_e': sd_log_e,
        'sd_log_inc': sd_log_inc,
        'sd_Omega': sd_Omega,
        'sd_omega': sd_omega,
        'sd_f': sd_f,
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
                  sd_log_a: float, sd_log_e: float, sd_log_inc: float,
                  sd_Omega: float, sd_omega: float, sd_f: float,
                  seed: int):
    """Load data for the perturbed sun-earth-jupiter problem for train, val and test"""
    # Get the filename for these arguments
    filename = make_filename_sej(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                                 sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                                 sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                                 seed=seed)
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
                      sd_log_a: float, sd_log_e: float, sd_log_inc: float,
                      sd_Omega: float, sd_omega: float, sd_f: float,
                      seed: int, batch_size: int):
    """Make datasets for the perturbed sen-earth-jupiter problem for train, val and test"""

    # Attempt to load the data
    data = load_data_sej(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                         sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                         sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                         seed=seed)

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
                                                sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                                                sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                                                seed=seed_trn)
        inputs_val, outputs_val = make_data_sej(n_traj=n_traj_val, n_years=n_years, sample_freq=sample_freq,
                                                sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                                                sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                                                seed=seed_val)
        inputs_tst, outputs_tst = make_data_sej(n_traj=n_traj_tst, n_years=n_years, sample_freq=sample_freq,
                                                sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                                                sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                                                seed=seed_tst)
        
        # Get the filename for these arguments
        filename = make_filename_sej(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                                     sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                                     sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                                     seed=seed)
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

    # Create DataSet objects for train, val and test sets
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
    
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def combine_datasets_sej(n_traj: int, vt_split: float, n_years: int, sample_freq: int, 
                         sd_log_a: float, sd_log_e: float, sd_log_inc: float,
                         sd_Omega: float, sd_omega: float, sd_f: float,
                         seeds: List[int], batch_size: int):
    """Make datasets for the general 3 body problem for train, val and test"""
    
    # First dataset
    seed = seeds[0]
    # print(f'i=1 , seed={seed:3} ', end=None)
    ds_trn, ds_val, ds_tst = make_datasets_sej(
           n_traj=n_traj, vt_split=vt_split, 
           n_years=n_years, sample_freq=sample_freq,
           sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
           sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
           seed=seed,
           batch_size=batch_size)
    # Concatenate remaining datasets
    for seed in tqdm(list(seeds[1:])):
        # Status update
        # print(f'i={i+1:2}, seed={seed:3} ', end=None)
        # The new batch of datasets
        ds_trn_new, ds_val_new, ds_tst_new = make_datasets_sej(
               n_traj=n_traj, vt_split=vt_split, 
               n_years=n_years, sample_freq=sample_freq,
               sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
               sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
               seed=seed,
               batch_size=batch_size)
        # Concatenate the new datasets
        ds_trn = ds_trn.concatenate(ds_trn_new)
        ds_val = ds_val.concatenate(ds_val_new)
        ds_tst = ds_tst.concatenate(ds_tst_new)        
        
    # Return the three large concatenated datasets
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def main():
    """Main routine for making SEJ data sets"""
    # Length and sample frequency
    n_years = 100
    sample_freq = 10
    
    # Make the simulation archive for the unperturbed (real) sun-earth-jupiter system
    make_sa_sej(n_years=n_years, sample_freq=sample_freq)
    
    # Number and size of trajectories
    num_batches = 20
    n_traj = 10000
    batch_size = 64
    vt_split = 0.20
    
    # Orbital perturbation scales
    sd_log_a = 0.01
    sd_log_e = 0.10
    sd_log_inc = 0.10
    sd_Omega = np.pi * 0.02
    sd_omega = np.pi * 0.02
    sd_f = np.pi * 0.02
    
    # List of seeds to use for datasets
    seed0 = 42
    seed1 = seed0 + num_batches * 3
    seeds = list(range(seed0, seed1, 3))
    
    # Run perturbed simulation
    for i, seed in tqdm(enumerate(seeds)):
        make_datasets_sej(n_traj=n_traj, vt_split=vt_split, 
                          n_years=n_years, sample_freq=sample_freq,
                          sd_log_a=sd_log_a, sd_log_e=sd_log_e, sd_log_inc=sd_log_inc,
                          sd_Omega=sd_Omega, sd_omega=sd_omega, sd_f=sd_f,
                          seed=seed, batch_size=batch_size)
# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
