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
import zlib
import pickle
from tqdm.auto import tqdm

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def make_traj_g3b(m, a, e, inc, Omega, omega, f, n_years):
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
    n_years: the number of years of data to simulate
    the orbital element inputs a, e, inc, Omega, omega, f are all arrays of size 3
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

    # Unpack masses of the 3 objects
    m1, m2, m3 = m
    
    # Unpack the orbital elements
    a2, a3 = a
    e2, e3 = e
    inc2, inc3 = inc
    Omega2, Omega3 = Omega
    omega2, omega3 = omega
    f2, f3 = f
    
    # Test if a2 and a3 are out of order; if so, flip them so a2 <= a3
    if a2 > a3:
        elts2 = (a2, e2, inc2, Omega2, omega2, f2)
        elts3 = (a3, e3, inc3, Omega3, omega3, f3)
        (elts2, elts3) = (elts3, elts2)

    # Add primary with specified mass at origin with 0 velocity
    sim.add(m=m1)

    # Add body 2
    sim.add(m=m2, a=a2, e=e2, inc=inc2, Omega=Omega2, omega=omega2, f=f2)
    
    # Add body 3
    sim.add(m=m3, a=a3, e=e3, inc=inc3, Omega=Omega3, omega=omega3, f=f3)

    # Move to the center-of-momentum coordinate system
    sim.move_to_com()

    # Initialize cartesian entries to zero vectors
    num_particles = 3
    space_dims = 3
    traj_shape = (N, num_particles, space_dims)
    q = np.zeros(traj_shape, dtype=np.float32)
    v = np.zeros(traj_shape, dtype=np.float32)
    acc = np.zeros(traj_shape, dtype=np.float32)

    # Mass vector
    m = np.array(m, dtype=np.float32)

    # Initialize placeholders for kinetic and potential energy
    T = np.zeros(N, dtype=np.float32)
    U = np.zeros(N, dtype=np.float32)

    # Initialize momentum and angular momentum
    mom_shape = (N, 3)
    P = np.zeros(mom_shape, dtype=np.float32)
    L = np.zeros(mom_shape, dtype=np.float32)

    # The coefficients for gravitational potential energy on particle pairs
    k_12 = sim.G * m1 * m2
    k_13 = sim.G * m1 * m3
    k_23 = sim.G * m2 * m3

    # Set the times for snapshots
    ts = np.linspace(0.0, n_years, N)

    # The particles for the 3 bodies
    p1 = sim.particles[0]
    p2 = sim.particles[1]
    p3 = sim.particles[2]

    # Simulate the orbits
    # Start by integrating backward, then forward for a small step
    # This allows rebound to correctly initialize the acceleration
    sim.integrate(-1E-6, exact_finish_time=1)
    sim.integrate(1E-6, exact_finish_time=1)
    for i, t in enumerate(ts):
        # Integrate to the current time step with an exact finish time
        sim.integrate(t, exact_finish_time=1)
        # Save the position
        q[i] = [[p1.x, p1.y, p1.z],
                [p2.x, p2.y, p2.z],
                [p3.x, p3.y, p3.z]]
        # Save the velocity
        v[i] = [[p1.vx, p1.vy, p1.vz],
                [p2.vx, p2.vy, p2.vz],
                [p3.vx, p3.vy, p3.vz]]
        # Save the acceleration
        acc[i] = [[p1.ax, p1.ay, p1.az],
                  [p2.ax, p2.ay, p2.az],
                  [p3.ax, p3.ay, p3.az]]
        
        # Extract the 3 positions 
        q1 = q[i,0]
        q2 = q[i,1]
        q3 = q[i,2]

        # Extract the 3 velocities
        v1 = v[i,0]
        v2 = v[i,1]
        v3 = v[i,2]

        # Kinetic energy
        T1 = 0.5 * m1 * np.sum(np.square(v1))
        T2 = 0.5 * m2 * np.sum(np.square(v2))
        T3 = 0.5 * m3 * np.sum(np.square(v3))
        T[i] = T1 + T2 + T3
        
        # Potential energy; 3 pairs of interacting particles (3 choose 2)
        r_12 = np.linalg.norm(q2 - q1)
        r_13 = np.linalg.norm(q3 - q1)
        r_23 = np.linalg.norm(q3 - q2)
        U[i] = -(k_12/r_12 + k_13/r_13 + k_23/r_23)

        # The momentum vector; should be zero in the COM frame
        mv1 = m1 * v1
        mv2 = m2 * v2
        mv3 = m3 * v3
        P[i] = mv1 + mv2 + mv3

        # The angular momentum vector; should be constant by conservation of angular momentum
        L[i] = np.cross(q1, mv1) + np.cross(q2, mv2) + np.cross(q3, mv3)

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
        'm': m
        }

    # Assemble the output dict
    outputs = {
        # the trajectory
        'q': q,
        'v': v,
        'a': acc,
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
def make_data_g3b(n_traj: int, n_years: int, 
                  m_min: float=1.0E-9, m_max: float=1.0, 
                  a_min: float = 0.50, a_max: float = 32.0, 
                  e_max = 0.20, inc_max = 0.0, seed = 42):
    """
    Make a set of training data for the restricted two body problem
    INPUTS:
    n_traj: the number of trajectories to sample
    n_years: the number of years for each trajectory, e.g. 2
    m_min: minimum mass of the second (lighter) body in solar masses 
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

    # Number of particles and spatial dimensions
    num_particles = 3
    space_dims = 3

    # Shape of arrays for various inputs and outputs
    time_shape = (n_traj, traj_size)
    init_shape = (n_traj, num_particles, space_dims)
    mass_shape = (n_traj, num_particles)
    traj_shape = (n_traj, traj_size, num_particles, space_dims)
    mom_shape = (n_traj, traj_size, space_dims)
    
    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Initialize masses; m1 always has mass 1.0, m2 has mass at most m_max
    m1 = np.ones(shape=n_traj, dtype=np.float32)
    # Draw m2 from a log uniform distribution
    log_m2 = np.random.uniform(low=np.log(m_min), high=np.log(m_max), size=n_traj).astype(np.float32)
    m2 = np.exp(log_m2)
    # Draw m3 from a log uniform distribution
    log_m3 = np.random.uniform(low=np.log(m_min), high=np.log(m_max), size=n_traj).astype(np.float32)
    m3 = np.exp(log_m3)
    # Assemble into m
    orb_m = np.stack([m1, m2, m3], axis=1)

    # Initialize orbital element by sampling according to the inputs
    elt_size = (n_traj, 2)
    orb_a = np.random.uniform(low=a_min, high=a_max, size=elt_size).astype(np.float32)
    orb_e = np.random.uniform(low=0.0, high=e_max, size=elt_size).astype(np.float32)
    orb_inc = np.random.uniform(low=0.0, high=inc_max, size=elt_size).astype(np.float32)
    orb_Omega = np.random.uniform(low=-np.pi, high=np.pi, size=elt_size).astype(np.float32)
    orb_omega = np.random.uniform(low=-np.pi, high=np.pi, size=elt_size).astype(np.float32)
    orb_f = np.random.uniform(low=-np.pi, high=np.pi, size=elt_size).astype(np.float32)

    # Initialize arrays for the data
    # Inputs
    t = np.zeros(time_shape, dtype=np.float32)
    q0 = np.zeros(init_shape, dtype=np.float32)
    v0 = np.zeros(init_shape, dtype=np.float32)
    m = np.zeros(mass_shape, dtype=np.float32)
    # Outputs
    q = np.zeros(traj_shape, dtype=np.float32)
    v = np.zeros(traj_shape, dtype=np.float32)
    acc = np.zeros(traj_shape, dtype=np.float32)
    T = np.zeros(time_shape, dtype=np.float32)
    U = np.zeros(time_shape, dtype=np.float32)
    H = np.zeros(time_shape, dtype=np.float32)
    P = np.zeros(mom_shape, dtype=np.float32)
    L = np.zeros(mom_shape, dtype=np.float32)
    
    # Sample the trajectories
    for i in tqdm(range(n_traj)):
        # Generate one trajectory
        inputs, outputs = make_traj_g3b(m=orb_m[i], a=orb_a[i], e=orb_e[i], inc=orb_inc[i], 
                                        Omega=orb_Omega[i], omega=orb_omega[i], f=orb_f[i], n_years=n_years)
        
        # Copy results into main arrays
        t[i] = inputs['t']
        q0[i] = inputs['q0']
        v0[i] = inputs['v0']
        m[i] = inputs['m']
        q[i] = outputs['q']
        v[i] = outputs['v']
        acc[i] = outputs['a']
        T[i] = outputs['T']
        U[i] = outputs['U']
        H[i] = outputs['H']
        P[i] = outputs['P']
        L[i] = outputs['L']

    # Assemble the input dict
    inputs = {
        't': t,
        'q0': q0,
        'v0': v0,
        'm': m,
        }

    # Assemble the output dict
    outputs = {
        'q': q,
        'v': v,
        'a': acc,
        'q0_rec': q0,
        'v0_rec': v0,
        'T': T,
        'U': U,
        'H': H,
        'P': P,
        'L': L}

    # Return the dicts
    return (inputs, outputs)

# ********************************************************************************************************************* 
def make_filename_g3b(n_traj: int, vt_split: float, n_years: int, m_min: float, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seed: int):
    """Make file name for serializing datasets for the restricted 2 body problem"""
    
    # Create dictionary with attributes
    attributes = {
        'n_traj': n_traj,
        'vt_split': vt_split,
        'n_years': n_years,
        'm_min': m_min,
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
    return f'../data/g3b/{hash_id}.pickle'

# ********************************************************************************************************************* 
def make_datasets_g2b(n_traj: int, vt_split: float, n_years: int, m_min: float, m_max: float,
                      a_min: float, a_max: float, e_max: float, inc_max: float, seed: int, batch_size: int):
    """Make datasets for the restricted 2 body problem for train, val and test"""
    # Get the filename for these arguments
    filename = make_filename_g3b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, m_min=m_min, m_max=m_max, 
                                 a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed)
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
        inputs_trn, outputs_trn = make_data_g3b(n_traj=n_traj_trn, n_years=n_years, m_min=m_min, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_trn)
        inputs_val, outputs_val = make_data_g3b(n_traj=n_traj_val, n_years=n_years, m_min=m_min, m_max=m_max,
                                                a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed_val)
        inputs_tst, outputs_tst = make_data_g3b(n_traj=n_traj_tst, n_years=n_years, m_min=m_min, m_max=m_max,
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
    m_min = 1.0E-7
    m_max = 0.002
    a_min = 0.50
    a_max = 32.0
    e_max = 0.20
    inc_max = np.pi / 4.0
    
    # Delegate to make_datasets_g2b
    return make_datasets_g2b(n_traj=n_traj, vt_split=vt_split, n_years=n_years, m_min=m_min, m_max=m_max,
                             a_min=a_min, a_max=a_max, e_max=e_max, inc_max=inc_max, seed=seed, batch_size=batch_size)
    
# ********************************************************************************************************************* 
def main():
    """Main routine for making datasets"""
    # Inputs for make_datasets_g2b
    vt_split = 0.20
    n_years = 2
    batch_size = 64
    seed = 42
    
    n_traj_small = 100
    n_traj_medium = 10000
    n_traj_large = 100000
    
    # Create DataSet objects for toy size problem
    print(f'Generating small data set for solar-type systems ({n_traj_small} orbits)...')
    make_datasets_solar(n_traj=n_traj_small, vt_split=vt_split, n_years=n_years, batch_size=batch_size, seed=seed)

    # Create a medium data set with 10,000 solar type orbits
    print(f'Generating medium data set for solar-type systems ({n_traj_medium} orbits) ...')
    make_datasets_solar(n_traj=n_traj_medium, vt_split=vt_split, n_years=n_years, batch_size=batch_size, seed=seed)
        
    # Create a large data set with 50,000 binary type orbits
    print(f'Generating large data set for binary-type systems ({n_traj_large} orbits) ...')
    make_datasets_solar(n_traj=n_traj_large, vt_split=vt_split, n_years=n_years, batch_size=batch_size, seed=seed)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
