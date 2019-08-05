"""
Harvard IACS Masters Thesis
General Two Body Problem
Generate training data (trajectories)

Michael S. Emanuel
Mon Aug 05 10:21:00 2019
"""

# Library imports
import tensorflow as tf
import rebound
import numpy as np
import zlib
import pickle
from tqdm.auto import tqdm

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def make_traj_g2b(m1, m2, a, e, inc, Omega, omega, f, n_years):
    """
    Make an array of training data points for the general 2 body problem.
    Creates one trajectory with the same initial configuration.
    INPUTS:
    m1: mass of the primary in solar units
    m2: mass of the secondary in solar units
    a: the semi-major axis of the orbit in AU
    e: the eccentricity of the orbit; must be in the range [0, 1) for an elliptic orbit
    inc: the inclination
    Omega: the longitude of the ascending node
    omega: the argument of pericenter
    f: the true anomaly at epoch    
    n_years: the number of years of data to simulate
    RETURNS:
    input_dict: a dictionary with the input fields t, q0, v0
    output_dict: a dictionary with the output fields q, v, a, q0_rec, v0_rec, T, H, H, L
    """
    # The sample frequency is 365 per year (dt approximately = 1 day)
    sample_freq = 365
    # Number of samples including both start and end
    N = sample_freq*n_years + 1

    # Create a simulation
    sim = rebound.Simulation()

    # Set units
    sim.units = ('yr', 'AU', 'Msun')

    # Set integrator to ias15; need acceleration output
    sim.integrator = 'ias15'

    # Set the simulation time step based on sample_freq
    sim.dt = 1.0 / sample_freq

    # Add primary with specified mass at origin with 0 velocity
    sim.add(m=m1)

    # Add the orbiting body
    sim.add(m=m2, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)
    
    # Move to the center-of-momentum coordinate system
    sim.move_to_com()

    # Initialize cartesian entries to zero vectors for particle 1
    q1 = np.zeros((N,3), dtype=np.float32)
    v1 = np.zeros((N,3), dtype=np.float32)
    a1 = np.zeros((N,3), dtype=np.float32)

    # Initialize cartesian entries to zero vectors for particle 2
    q2 = np.zeros((N,3), dtype=np.float32)
    v2 = np.zeros((N,3), dtype=np.float32)
    a2 = np.zeros((N,3), dtype=np.float32)

    # Initialize placeholders for kinetic and potential energy
    T = np.zeros(N, dtype=np.float32)
    U = np.zeros(N, dtype=np.float32)

    # Initialize momentum and angular momentum
    P = np.zeros((N,3), dtype=np.float32)
    L = np.zeros((N,3), dtype=np.float32)

    # The coefficient for gravitational potential energy
    k = sim.G * m1 * m2

    # Set the times for snapshots
    ts = np.linspace(0.0, n_years, N)

    # The particles for the primary and orbiting body
    p1 = sim.particles[0]
    p2 = sim.particles[1]

    # Simulate the orbits
    # Start by integrating backward, then forward for a small step
    # This allows rebound to correctly initialize the acceleration
    sim.integrate(-1E-6, exact_finish_time=1)
    sim.integrate(1E-6, exact_finish_time=1)
    for i, t in enumerate(ts):
        # Integrate to the current time step with an exact finish time
        sim.integrate(t, exact_finish_time=1)
        # Save the position
        q1[i] = [p1.x, p1.y, p1.z]
        q2[i] = [p2.x, p2.y, p2.z]
        # Save the velocity
        v1[i] = [p1.vx, p1.vy, p1.vz]
        v2[i] = [p2.vx, p2.vy, p2.vz]
        # Save the acceleration
        a1[i] = [p1.ax, p1.ay, p1.az]
        a2[i] = [p2.ax, p2.ay, p2.az]

        # Kinetic energy
        T1 = 0.5 * m1 * np.sum(v1[i] * v1[i])
        T2 = 0.5 * m2 * np.sum(v2[i] * v2[i])
        T[i] = T1 + T2
        
        # Potential energy
        r = np.linalg.norm(q2[i] - q1[i])
        U[i] = -k / r

        # The momentum vector; should be zero in the COM frame
        P[i] = m1 * v1[i] + m2 * v2[i]

        # The angular momentum vector; should be constant by conservation of angular momentum
        L[i] = np.cross(q1[i], m1*v1[i]) + np.cross(q2[i], m2*v2[i])

    # The total energy is the sum of kinetic and potential; should be constant by conservation of energy
    H = T + U
    
    # The initial position and velocity
    q1_init = q1[0]
    q2_init = q2[0]
    v1_init = v1[0]
    v2_init = v2[0]
    
    # Assemble the input dict
    inputs = {
        't': ts,
        'q1_init': q1_init,
        'q2_init': q2_init,
        'v1_init': v1_init,
        'v2_init': v2_init,
        'm1': m1,
        'm2': m2}

    # Assemble the output dict
    outputs = {
        'q1': q1,
        'q2': q2,
        'v1': v1,
        'v2': v2,
        'a1': a1,
        'a2': a2,
        # the initial conditions, which should be recovered
        'q1_rec': q1_init,
        'q2_rec': q2_init,
        'v1_rec': v1_init,
        'v2_rec': v2_init,
        # the energy and angular momentum, which should be conserved
        'T': T,
        'U': U,
        'H': H,
        'P': P,
        'L': L}

    # Return the dicts
    return (inputs, outputs)


# ********************************************************************************************************************* 
def make_data_g2b(n_traj: int, n_years: int, m_max: float=1.0, 
                  a_min: float = 0.50, a_max: float = 32.0, 
                  e_max = 0.20, inc_max = 0.0, seed = 42):
    """
    Make a set of training data for the restricted two body problem
    INPUTS:
    n_traj: the number of trajectories to sample
    n_years: the number of years for each trajectory, e.g. 2
    m_max: maximum mass of the second (lighter) body in solar masses 
    a_min: minimum semi-major axis in AU, e.g. 0.50
    a_max: maximum semi-major axis in AU, e.g. 32.0
    e_max: maximum eccentricity, e.g. 0.20
    inc_max: maximum inclination, e.g. pi/4
    """
    # The sample frequency is 365 per year (dt approximately = 1 day)
    sample_freq = 365
    # Number of samples including both start and end in each trajectory
    traj_size = sample_freq*n_years + 1
    # Number of spatial dimensions
    space_dims = 3

    # Shape of arrays for various inputs and outputs
    # scalar_shape = (n_traj, )
    init_shape = (n_traj, space_dims)
    time_shape = (n_traj, traj_size)
    traj_shape = (n_traj, traj_size, space_dims)
    
    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Initialize masses; m1 always has mass 1.0, m2 has mass at most m_max
    m1 = np.ones(shape=n_traj, dtype=np.float32)
    m2 = np.random.uniform(low=1.0E-9, high=m_max, size=n_traj).astype(np.float32)

    # Initialize orbital element by sampling according to the inputs
    orb_a = np.random.uniform(low=a_min, high=a_max, size=n_traj).astype(np.float32)
    orb_e = np.random.uniform(low=0.0, high=e_max, size=n_traj).astype(np.float32)
    orb_inc = np.random.uniform(low=0.0, high=inc_max, size=n_traj).astype(np.float32)
    orb_Omega = np.random.uniform(low=-np.pi, high=np.pi, size=n_traj).astype(np.float32)
    orb_omega = np.random.uniform(low=-np.pi, high=np.pi, size=n_traj).astype(np.float32)
    orb_f = np.random.uniform(low=-np.pi, high=np.pi, size=n_traj).astype(np.float32)

    # Initialize arrays for the data
    t = np.zeros(time_shape, dtype=np.float32)
    q1_init = np.zeros(init_shape, dtype=np.float32)
    q2_init = np.zeros(init_shape, dtype=np.float32)
    v1_init = np.zeros(init_shape, dtype=np.float32)
    v2_init = np.zeros(init_shape, dtype=np.float32)
    q1 = np.zeros(traj_shape, dtype=np.float32)
    q2 = np.zeros(traj_shape, dtype=np.float32)
    v1 = np.zeros(traj_shape, dtype=np.float32)
    v2 = np.zeros(traj_shape, dtype=np.float32)
    a1 = np.zeros(traj_shape, dtype=np.float32)
    a2 = np.zeros(traj_shape, dtype=np.float32)
    T = np.zeros(time_shape, dtype=np.float32)
    U = np.zeros(time_shape, dtype=np.float32)
    H = np.zeros(time_shape, dtype=np.float32)
    P = np.zeros(traj_shape, dtype=np.float32)
    L = np.zeros(traj_shape, dtype=np.float32)
    
    # Sample the trajectories
    for i in tqdm(range(n_traj)):
        # Generate one trajectory
        inputs, outputs = make_traj_g2b(m1=m1[i], m2=m2[i], a=orb_a[i], e=orb_e[i], inc=orb_inc[i], 
                                        Omega=orb_Omega[i], omega=orb_omega[i], f=orb_f[i], n_years=n_years)
        
        # Copy results into main arrays
        t[i, :] = inputs['t']
        q1_init[i, :] = inputs['q1_init']
        q2_init[i, :] = inputs['q2_init']
        v1_init[i, :] = inputs['v1_init']
        v2_init[i, :] = inputs['v2_init']
        # m1[i] = inputs['m1']
        # m2[i] = inputs['m2']
        q1[i, :, :] = outputs['q1']
        q2[i, :, :] = outputs['q2']
        v1[i, :, :] = outputs['v1']
        v2[i, :, :] = outputs['v2']
        a1[i, :, :] = outputs['a1']
        a2[i, :, :] = outputs['a2']
        T[i, :] = outputs['T']
        U[i, :] = outputs['U']
        H[i, :] = outputs['H']
        P[i, :] = outputs['P']
        L[i, :] = outputs['L']

    # Assemble the input dict
    inputs = {
        't': t,
        'q1_init': q1_init,
        'q2_init': q2_init,
        'v1_init': v1_init,
        'v2_init': v2_init,
        'm1': m1,
        'm2': m2}

    # Assemble the output dict
    outputs = {
        'q1': q1,
        'q2': q2,
        'v1': v1,
        'v2': v2,
        'a1': a1,
        'a2': a2,
        'q1_rec': q1_init,
        'q2_rec': q2_init,
        'v1_rec': v1_init,
        'v2_rec': v2_init,
        'T': T,
        'U': U,
        'H': H,
        'P': P,
        'L': L}

    # Return the dicts
    return (inputs, outputs)

# ********************************************************************************************************************* 
def make_filename_g2b(n_traj: int, vt_split: float, n_years: int, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seed: int):
    """Make file name for serializing datasets for the restricted 2 body problem"""
    
    # Create dictionary with attributes
    attributes = {
        'n_traj': n_traj,
        'vt_split': vt_split,
        'n_years': n_years,
        'm_max': m_max,
        'a_min': a_min,
        'a_max': a_max,
        'e_max': e_max,
        'inc_max': inc_max,
        'seed': seed,
        # don't need to include batch_size because it doesn't affect data contents, only tf.data.Dataset
        }
    
    # Create a non-negative hash ID of the attributes
    # hash_id = hash(frozenset(attributes.items())) & sys.maxsize
    attributes_bytes = bytes(str(attributes), 'utf-8')
    hash_id = zlib.crc32(attributes_bytes)
    
    # Create the filename
    return f'../data/g2b/{hash_id}.pickle'

# ********************************************************************************************************************* 
def make_datasets_g2b(n_traj: int, vt_split: float, n_years: int, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seed: int, batch_size: int):
    """Make datasets for the restricted 2 body problem for train, val and test"""
    # Get the filename for these arguments
    filename = make_filename_g2b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, 
                                 m_max=m_max, a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed)
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
    # Generate the data and save it to the file
    except:
        # Status 
        print(f'Unable to load data from {filename}.')
        # Set the number of trajectories for train, validation and test
        n_traj_trn = n_traj
        n_traj_val = int(n_traj * vt_split)
        n_traj_tst = n_traj_val

        # Set the random seeds
        seed_trn = seed + 0
        seed_val = seed + 1
        seed_tst = seed + 2

        # Generate inputs and outputs for orbits with input parameters
        inputs_trn, outputs_trn = make_data_g2b(n_traj=n_traj_trn, n_years=n_years, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_trn)
        inputs_val, outputs_val = make_data_g2b(n_traj=n_traj_val, n_years=n_years, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_val)
        inputs_tst, outputs_tst = make_data_g2b(n_traj=n_traj_tst, n_years=n_years, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_tst)
        
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
    ds_trn = ds_trn.shuffle(buffer_size=buffer_size).batch(batch_size)
    ds_val = ds_val.shuffle(buffer_size=buffer_size).batch(batch_size)
    ds_tst = ds_tst.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    return ds_trn, ds_val, ds_tst

# ********************************************************************************************************************* 
def make_datasets_solar(n_traj=1000, vt_split=0.20, n_years=2, batch_size=64, seed=42):
    """Make 3 data sets for solar-system -like orbits with a range of a, e, and inclinations."""
    # Set the parameters for solar-system -like orbits
    m_max = 0.002
    a_min = 0.50
    a_max = 32.0
    e_max = 0.20
    inc_max = np.pi / 4.0
    
    # Delegate to make_datasets_g2b
    return make_datasets_g2b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, m_max=m_max,
                             a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed, batch_size=batch_size)
    
# ********************************************************************************************************************* 
def make_datasets_binary(n_traj=1000, vt_split=0.20, n_years=2, batch_size=64, seed=42):
    """Make 3 data sets for binary star -like orbits with a range of a, e, and inclinations."""
    # Set the parameters for solar-system -like orbits
    m_max = 1.0
    a_min = 0.50
    a_max = 32.0
    e_max = 0.20
    inc_max = np.pi / 4.0
    
    # Delegate to make_datasets_g2b
    return make_datasets_g2b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, m_max=m_max,
                             a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed, batch_size=batch_size)