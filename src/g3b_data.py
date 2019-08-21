"""
Harvard IACS Masters Thesis
General Three Body Problem
Generate training data (trajectories)

Michael S. Emanuel
Mon Aug 05 10:21:00 2019
"""

# Library imports
import tensorflow as tf
import rebound
import numpy as np
import pickle
from tqdm.auto import tqdm
from typing import List

# Local imports
from utils import hash_id_crc32

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def make_traj_g3b(m, a, e, inc, Omega, omega, f, n_years, sample_freq, integrator='ias15', debug_print = False):
    """
    Make an array of training data points for the general 3 body problem.
    Creates one trajectory with the same initial configuration.
    INPUTS:
    m: masses of of 3 bodies in solar units; array of size 3
    a: the semi-major axis of the orbit in AU;
    e: the eccentricity of the orbit; must be in the range [0, 1) for an elliptic orbit
    inc: the inclination
    Omega: the longitude of the ascending node
    omega: the argument of pericenter
    f: the true anomaly at epoch    
    n_years: the number of years of data to simulate, e.g. 2
    sample_freq: number of sample points per year, e.g. 365
    integrator: the name of the rebound integrator.  one of 'ias15' or 'whfast'
    NOTE:
    the orbital element inputs a, e, inc, Omega, omega, f are all arrays of size 3
    RETURNS:
    input_dict: a dictionary with the input fields t, q0, v0
    output_dict: a dictionary with the output fields q, v, a, q0_rec, v0_rec, T, H, H, L
    """
    # Number of samples including both start and end
    N = n_years*sample_freq + 1

    # Create a simulation
    sim = rebound.Simulation()

    # Set units
    sim.units = ('yr', 'AU', 'Msun')

    # Set integrator.  if acceleration output required, must use ias15
    sim.integrator = integrator

    # Set the simulation time step based on sample_freq
    sim.dt = 1.0 / sample_freq

    # Get sorted list of indices for adding bodies so a is in ascending order
    js = np.argsort(a)
    
    # Unpack the masses; note that m[0] is the primary and indices on m must be shifted
    # by 1 compared to indices on the orbital elements
    m0 = m[0]
    m1 = m[js[0]+1]
    m2 = m[js[1]+1]
    
    # Mass output - must sort along with a! Also convert to float32 type.
    # m_ = np.array([m0] + [m[j+1] for j in js], dtype=np.float32)
    m_ = np.array([m0, m1, m2], dtype=np.float32)

    # Add primary with specified mass at origin with 0 velocity
    sim.add(m=m0)

    # Add the bodies in ascending order of a
    # j is the index position of the body that is the ith from the center
    # The mass is m[j+1] because the mass vector has one extra entry for the primary at the beginning
    for j in js:
        sim.add(m=m[j+1], a=a[j], e=e[j], inc=inc[j], Omega=Omega[j], omega=omega[j], f=f[j])
        if debug_print:
            print(f'particle {j}, m={m[j+1]}, a={a[j]}, e={e[j]}, inc={inc[j]}, '
                  f'Omega={Omega[j]}, omega={omega[j]}, f={f[j]}')
            
    # Move to the center-of-momentum coordinate system
    sim.move_to_com()

    if debug_print:
        print(f'\nStatus after shift to COM:\n')
        print(sim.status())

    # Array shapes
    num_particles = 3
    space_dims = 3
    traj_shape = (N, num_particles, space_dims)
    # the number of orbital elements is the number of particles minus 1; no elements for primary
    elt_shape = (N, num_particles-1)
    mom_shape = (N, 3)

    # Initialize cartesian entries to zero vectors
    q = np.zeros(traj_shape, dtype=np.float32)
    v = np.zeros(traj_shape, dtype=np.float32)
    acc = np.zeros(traj_shape, dtype=np.float32)
    
    # Initialize orbital elements over time to zero vectors
    orb_a = np.zeros(elt_shape, dtype=np.float32)
    orb_e = np.zeros(elt_shape, dtype=np.float32)
    orb_inc = np.zeros(elt_shape, dtype=np.float32)
    orb_Omega = np.zeros(elt_shape, dtype=np.float32)
    orb_omega = np.zeros(elt_shape, dtype=np.float32)
    orb_f = np.zeros(elt_shape, dtype=np.float32)

    # Initialize placeholders for kinetic and potential energy
    T = np.zeros(N, dtype=np.float32)
    U = np.zeros(N, dtype=np.float32)

    # Initialize momentum and angular momentum
    P = np.zeros(mom_shape, dtype=np.float32)
    L = np.zeros(mom_shape, dtype=np.float32)

    # The coefficients for gravitational potential energy on particle pairs
    k_01 = sim.G * m0 * m1
    k_02 = sim.G * m0 * m2
    k_12 = sim.G * m1 * m2

    # Set the times for snapshots
    ts = np.linspace(0.0, n_years, N, dtype=np.float32)

    # The particles for the 3 bodies
    p0 = sim.particles[0]
    p1 = sim.particles[1]
    p2 = sim.particles[2]

    # Simulate the orbits
    # Start by integrating backward, then forward for a small step
    # This allows rebound to correctly initialize the acceleration
    sim.integrate(-1E-6, exact_finish_time=1)
    sim.integrate(1E-6, exact_finish_time=1)
    for i, t in enumerate(ts):
        # Integrate to the current time step with an exact finish time
        sim.integrate(t, exact_finish_time=1)

        # Save the position
        q[i] = [[p0.x, p0.y, p0.z],
                [p1.x, p1.y, p1.z],
                [p2.x, p2.y, p2.z]]
        # Save the velocity
        v[i] = [[p0.vx, p0.vy, p0.vz],
                [p1.vx, p1.vy, p1.vz],
                [p2.vx, p2.vy, p2.vz]]
        # Save the acceleration
        acc[i] = [[p0.ax, p0.ay, p0.az],
                  [p1.ax, p1.ay, p1.az],
                  [p2.ax, p2.ay, p2.az]]
        
        # Calculate the two orbits
        orb1 = p1.calculate_orbit()
        orb2 = p2.calculate_orbit()
        
        # Save the orbital elements
        orb_a[i] = [orb1.a, orb2.a]
        orb_e[i] = [orb1.e, orb2.e]
        orb_inc[i] = [orb1.inc, orb2.inc]
        orb_Omega[i] = [orb1.Omega, orb2.Omega]
        orb_omega[i] = [orb1.omega, orb2.omega]
        orb_f[i] = [orb1.f, orb2.f]

        # Extract the 3 positions 
        q0 = q[i,0]
        q1 = q[i,1]
        q2 = q[i,2]

        # Extract the 3 velocities
        v0 = v[i,0]
        v1 = v[i,1]
        v2 = v[i,2]

        # Kinetic energy
        T0 = 0.5 * m0 * np.sum(np.square(v0))
        T1 = 0.5 * m1 * np.sum(np.square(v1))
        T2 = 0.5 * m2 * np.sum(np.square(v2))
        T[i] = T0 + T1 + T2
        
        # Potential energy; 3 pairs of interacting particles (3 choose 2)
        r_01 = np.linalg.norm(q1 - q0)
        r_02 = np.linalg.norm(q2 - q0)
        r_12 = np.linalg.norm(q2 - q1)
        U[i] = -(k_01/r_01 + k_02/r_02 + k_12/r_12)

        # The momentum vector; should be zero in the COM frame
        mv0 = m0 * v0
        mv1 = m1 * v1
        mv2 = m2 * v2
        P[i] = mv0 + mv1 + mv2

        # The angular momentum vector; should be constant by conservation of angular momentum
        L[i] =  np.cross(q0, mv0) + np.cross(q1, mv1) + np.cross(q2, mv2)

    # The total energy is the sum of kinetic and potential; should be constant by conservation of energy
    H = T + U
    
    # The initial position and velocity
    q0 = q[0]
    v0 = v[0]
    
    # Assemble the input dict
    inputs = {
        't': ts,
        'q0': q0,
        'v0': v0,
        'm': m_
        }

    # Assemble the output dict
    outputs = {
        # the trajectory
        'q': q,
        'v': v,
        'a': acc,

        # the orbital elements over time
        'orb_a': orb_a,
        'orb_e': orb_e,
        'orb_inc': orb_inc,
        'orb_Omega': orb_Omega,
        'orb_omega': orb_omega,
        'orb_f': orb_f,
        
        # the initial conditions, which should be recovered
        'q0_rec': q0,
        'v0_rec': v0,

        # the energy, momentum and angular momentum, which should be conserved
        'T': T,
        'U': U,
        'H': H,
        'P': P,
        'L': L}

    # Return the dicts
    return (inputs, outputs)


# ********************************************************************************************************************* 
def repeat_array(x, batch_size: int):
    """Repeat an array into a batch of copies"""
    return np.broadcast_to(x, (batch_size,) + x.shape)

# ********************************************************************************************************************* 
def traj_to_batch(inputs, outputs, batch_size: int):
    """Repeat arrays in one example trajectory into a full batch of identical trajectories."""
    batch_in = dict()
    for field in inputs:
        batch_in[field] = repeat_array(inputs[field], batch_size)
        
    batch_out = dict()
    for field in outputs:
        batch_out[field] = repeat_array(outputs[field], batch_size)
        
    return batch_in, batch_out

# ********************************************************************************************************************* 
def make_data_g3b(n_traj: int, n_years: int, sample_freq: int,
                  m_min: float=1.0E-9, m_max: float=1.0, 
                  a_min: float = 0.50, a_max: float = 32.0, 
                  e_max = 0.20, inc_max = 0.04, seed = 42):
    """
    Make a set of training data for the restricted two body problem
    INPUTS:
    n_traj: the number of trajectories to sample
    n_years: the number of years for each trajectory, e.g. 2
    integrator: the integrator used.  'ias15' or 'whfast'
    m_min: minimum mass of the second (lighter) body in solar masses 
    m_max: maximum mass of the second (lighter) body in solar masses 
    a_min: minimum semi-major axis in AU, e.g. 0.50
    a_max: maximum semi-major axis in AU, e.g. 32.0
    e_max: maximum eccentricity, e.g. 0.20
    inc_max: maximum inclination, e.g. pi/4
    """
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
    
    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Initialize masses; m1 always has mass 1.0, m2 has mass at most m_max
    m1 = np.ones(shape=n_traj)
    # Draw m2 from a log uniform distribution
    log_m2 = np.random.uniform(low=np.log(m_min), high=np.log(m_max), size=n_traj)
    m2 = np.exp(log_m2)
    # Draw m3 from a log uniform distribution
    log_m3 = np.random.uniform(low=np.log(m_min), high=np.log(m_max), size=n_traj)
    m3 = np.exp(log_m3)
    # Assemble into m0
    m0 = np.stack([m1, m2, m3], axis=1).astype(np.float32)

    # Initialize orbital element by sampling according to the inputs
    elt_size = (n_traj, 2)
    orb_a0 = np.random.uniform(low=a_min, high=a_max, size=elt_size).astype(np.float32)
    orb_e0 = np.random.uniform(low=0.0, high=e_max, size=elt_size).astype(np.float32)
    orb_inc0 = np.random.uniform(low=0.0, high=inc_max, size=elt_size).astype(np.float32)
    orb_Omega0 = np.random.uniform(low=-np.pi, high=np.pi, size=elt_size).astype(np.float32)
    orb_omega0 = np.random.uniform(low=-np.pi, high=np.pi, size=elt_size).astype(np.float32)
    orb_f0 = np.random.uniform(low=-np.pi, high=np.pi, size=elt_size).astype(np.float32)

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
        inputs_traj, outputs_traj = make_traj_g3b(m=m0[i], a=orb_a0[i], e=orb_e0[i], inc=orb_inc0[i], 
                                        Omega=orb_Omega0[i], omega=orb_omega0[i], f=orb_f0[i], 
                                        n_years=n_years, sample_freq=sample_freq, integrator=integrator)
        
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
def make_filename_g3b(n_traj: int, vt_split: float, n_years: int, sample_freq: int, 
                      m_min: float, m_max: float, a_min: float, a_max: float, e_max: float, inc_max: float, 
                      seed: int):
    """Make file name for serializing datasets for the general 3 body problem"""
    
    # Create dictionary with attributes
    attributes = {
        'n_traj': n_traj,
        'vt_split': vt_split,
        'n_years': n_years,
        'sample_freq': sample_freq,
        'm_min': m_min,
        'm_max': m_max,
        'a_min': a_min,
        'a_max': a_max,
        'e_max': e_max,
        'inc_max': inc_max,
        'seed': seed,
        }
    
    # Create a non-negative hash ID of the attributes
    # attributes_bytes = bytes(str(attributes), 'utf-8')
    # hash_id = zlib.crc32(attributes_bytes)
    hash_id = hash_id_crc32(attributes)
    
    # Create the filename
    return f'../data/g3b/{hash_id}.pickle'

# ********************************************************************************************************************* 
def load_data_g3b(n_traj: int, vt_split: float, n_years: int, sample_freq: int, m_min: float, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seed: int):
    """Load data for the general 3 body problem for train, val and test"""
    # Get the filename for these arguments
    filename = make_filename_g3b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                                 m_min=m_min, m_max=m_max, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, 
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
def make_datasets_g3b(n_traj: int, vt_split: float, n_years: int, sample_freq: int, m_min: float, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seed: int, batch_size: int):
    """Make datasets for the general 3 body problem for train, val and test"""

    # Attempt to load the data
    data = load_data_g3b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                         m_min=m_min, m_max=m_max, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed)

    # Unpack data if available
    if data is not None:
        inputs_trn, outputs_trn, inputs_val, outputs_val, inputs_tst, outputs_tst = data
    # Otherwise generate the trajectories using make_data_g3b
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
        inputs_trn, outputs_trn = make_data_g3b(n_traj=n_traj_trn, n_years=n_years, sample_freq=sample_freq,
                                                m_min=m_min, m_max=m_max, a_min=a_min, a_max=a_max, 
                                                e_max=e_max, inc_max=inc_max, seed=seed_trn)
        inputs_val, outputs_val = make_data_g3b(n_traj=n_traj_val, n_years=n_years, sample_freq=sample_freq,
                                                m_min=m_min, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_val)
        inputs_tst, outputs_tst = make_data_g3b(n_traj=n_traj_tst, n_years=n_years, sample_freq=sample_freq,
                                                m_min=m_min, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_tst)
        
        # Get the filename for these arguments
        filename = make_filename_g3b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq, 
                                     m_min=m_min, m_max=m_max, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, 
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
def combine_datasets_g3b(n_traj: int, vt_split: float, n_years: int, sample_freq: int, m_min: float, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seeds: List[int], batch_size: int):
    """Combine a collection of g3b data sets into one large data set."""
    
    # First dataset
    seed = seeds[0]
    # print(f'i=1 , seed={seed:3} ', end=None)
    ds_trn, ds_val, ds_tst = make_datasets_g3b(
           n_traj=n_traj, vt_split=vt_split, 
           n_years=n_years, sample_freq=sample_freq,
           m_min=m_min, m_max=m_max,
           a_min=a_min, a_max=a_max,
           e_max=e_max, inc_max=inc_max,
           seed=seed,
           batch_size=batch_size)
    # Concatenate remaining datasets
    for seed in tqdm(list(seeds[1:])):
        # Status update
        # print(f'i={i+1:2}, seed={seed:3} ', end=None)
        # The new batch of datasets
        ds_trn_new, ds_val_new, ds_tst_new = make_datasets_g3b(
               n_traj=n_traj, vt_split=vt_split, 
               n_years=n_years, sample_freq=sample_freq,
               m_min=m_min, m_max=m_max,
               a_min=a_min, a_max=a_max,
               e_max=e_max, inc_max=inc_max,
               seed=seed,
               batch_size=batch_size)
        # Concatenate the new datasets
        ds_trn = ds_trn.concatenate(ds_trn_new)
        ds_val = ds_val.concatenate(ds_val_new)
        ds_tst = ds_tst.concatenate(ds_tst_new)        
        
    # Return the three large concatenated datasets
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def make_datasets_solar(n_traj=1000, vt_split=0.20, n_years=100, sample_freq=10, batch_size=64, seed=42):
    """Make 3 data sets for solar-type systems with a range of a, e, and inclinations."""
    # Set the parameters for solar-system -like orbits
    # https://en.wikipedia.org/wiki/Planetary_mass
    # mass of Mercury is 1.7E-7 solar masses
    # mass of Jupiter is 9.5E-4 solar masses 
    m_min = 1.0E-7 
    m_max = 2.0E-3 
    # http://www.met.rdg.ac.uk/~ross/Astronomy/Planets.html
    # a ranges from 0.39 for Mercury to 30.0 for Neptune
    a_min = 0.50
    a_max = 32.0
    # Largest eccentricity is 0.206 for Mercury.  Heavier planets have eccentricities 0.007 to 0.054
    e_max = 0.08
    # largest inclination to the invariable plane is Mercury at 6 degrees = 0.10 radians
    # the heavier planets have inclinations in the 0-2 degree range = 0.035 radians
    inc_max = 0.04 

    # Delegate to make_datasets_g2b
    return make_datasets_g3b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq,
                             m_min=m_min, m_max=m_max, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, 
                             seed=seed, batch_size=batch_size)
    
# ********************************************************************************************************************* 
def combine_datasets_solar(num_data_sets: int, batch_size: int =64, seed0: int =42):
    """Combine a collection of solar data sets into one large data set."""

    # Number of trajectories in each constituent batch
    n_traj = 10000
    vt_split = 0.20

    # Time parameters
    n_years=100
    sample_freq=10

    # Configuration of system
    m_min = 1.0E-7 
    m_max = 2.0E-3 
    a_min = 0.50
    a_max = 32.0
    e_max = 0.08
    inc_max = 0.04
    
    # List of random seeds
    seeds = list(range(seed0, seed0+3*num_data_sets, 3))
    
    # Delegate to conbine_datasets_g3b
    ds_trn, ds_val, ds_tst = combine_datasets_g3b(
        n_traj=n_traj, vt_split=vt_split, n_years=n_years, sample_freq=sample_freq,
        m_min=m_min, m_max=m_max, a_min=a_min, a_max=a_max,
        e_max=e_max, inc_max=inc_max, seeds=seeds, batch_size=batch_size)
    
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def make_datasets_hard(n_traj=1000, vt_split=0.20, n_years=100, sample_freq=10, batch_size=64, seed=42):
    """Make 3 data sets for systems with more difficult parameter ranges."""
    # Set the parameters for solar-system -like orbits
    m_min = 1.0E-4 
    m_max = 1.0E-1 
    a_min = 0.50
    a_max = 32.0
    e_max = 0.20
    inc_max = 0.10 

    # Delegate to make_datasets_g3b
    return make_datasets_g3b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, m_min=m_min, m_max=m_max,
                             a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed, batch_size=batch_size)
    
# ********************************************************************************************************************* 
def main():
    """Main routine for making datasets"""
    # Inputs for make_datasets_g2b
    vt_split = 0.20
    n_years = 100
    sample_freq = 10
    batch_size = 64
    # Set seeds; for sizes up to large use the same seed for convenience
    seed_tiny = 42
    seed_small = 42
    seed_large = 42
    # Number of small data sets
    small_data_sets = 50
    
    n_traj_tiny = 64
    n_traj_small = 10000
    n_traj_large = 50000
    
    # Create a tiny data set with 100 solar type orbits
    print(f'Generating tiny data set for solar-type systems ({n_traj_tiny} orbits)...')
    make_datasets_solar(n_traj=n_traj_tiny, vt_split=1.0, n_years=n_years, sample_freq=sample_freq,
                        batch_size=batch_size, seed=seed_tiny)

    # Create a small data set with 10,000 solar type orbits
    print(f'Generating small data set for solar-type systems ({n_traj_small} orbits) ...')
    make_datasets_solar(n_traj=n_traj_small, vt_split=vt_split, n_years=n_years, 
                        batch_size=batch_size, seed=seed_small)
        
    # Create a large data set with 50,000 binary type orbits
    print(f'Generating large data set for binary-type systems ({n_traj_large} orbits) ...')
    make_datasets_solar(n_traj=n_traj_large, vt_split=vt_split, n_years=n_years, 
                        batch_size=batch_size, seed=seed_large)
    
    # Create a whole batch of small data sets with different seeds
    seeds = seed_small + 3*(np.arange(small_data_sets) + 1)
    print(f'Creating a batch of small data sets for solar-type systems with {n_traj_small} orbits')
    for i, seed in enumerate(seeds):
        print(f'Generating small data {i} with seed {seed}...')
        make_datasets_solar(n_traj=n_traj_small, vt_split=vt_split, n_years=n_years, 
                            batch_size=batch_size, seed=seed)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
