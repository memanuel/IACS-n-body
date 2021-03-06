"""
Harvard IACS Masters Thesis
Restricted Two Body Problem - Special Case of Circular Orbits (Eccentricity = 0)
Generate training data (trajectories)

Michael S. Emanuel
Tue Jun  18 15:29 2019
"""

# Library imports
import tensorflow as tf
import numpy as np

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def make_traj_r2bc(r0, theta0, n_years):
    """
    Make an array of training data points for the restricted 2 body circular problem.
    Creates one trajectory with the same initial configuration.
    INPUTS:
    r0: The radius of the orbit in AU
    theta0: the angle of the planet in the ecliptic at time 0 years
    n_years: the number of years of data to simulate
    RETURNS:
    input_dict: a dictionary with the input fields t, q0, v0
    output_dict: a dictionary with the output fields q, v, a
    """
    # The sample frequency is 365 per year (dt approximately = 1 day)
    sample_freq = 365
    # Number of samples including both start and end
    N = sample_freq*n_years + 1
    # Array of sample times at the specified frequency over the specified length of time
    t = np.arange(N, dtype=np.float32) / sample_freq

    # The gravitational constant in units of (AU, years, solar_mass)
    G = (2.0 * np.pi)**2
    # The mass of the sun in this coordinate system
    m0 = 1.0
    # The gravitational coefficient G*(m0 + m1) for the restricted two body problem in these units
    mu = G * m0

    # Set the angular frequency omega based on the radius using the relationship
    # mu = omega^2 r^3 --> omega = sqrt(mu / r^3)
    omega = np.sqrt(mu / r0**3)
    # Theta is offset by theta0
    theta = (omega * t) + theta0

    # Compute qx and qy from r and theta; wrap into position q
    qx = r0 * np.cos(theta, dtype=np.float32)
    qy = r0 * np.sin(theta, dtype=np.float32)
    q = np.stack([qx, qy], axis=1)
    
    # Compute vx and vy; wrap into velocity v
    vx = -omega * qy
    vy = +omega * qx
    v = np.stack([vx, vy], axis=1)

    # Compute ax and ay; wrap into acceleration a
    ax = -omega**2 * qx
    ay = -omega**2 * qy
    a = np.stack([ax, ay], axis=1)

    # Initial position q0
    qx0 = qx[0]
    qy0 = qy[0]
    q0 = np.stack([qx0, qy0], axis=0)
    
    # Initial velocity v0
    vx0 = vx[0]
    vy0 = vy[0]
    v0 = np.stack([vx0, vy0], axis=0)
    
    # Initial kinetic and potential energy; total energy
    T0 = 0.5 * np.sum(v0*v0)
    U0 = -mu / r0
    H0 = T0 + U0

    # Initial angular momentum
    L0 = qx * vy - qy*vx

    # Repeat the energy traj_size times
    ones_vec = np.ones(shape=(N,))
    T = T0 * ones_vec
    U = U0 * ones_vec
    H = H0 * ones_vec
    L = L0 * ones_vec
    
    # Assemble the input dict
    inputs = {
        't': t,
        'q0': q0,
        'v0': v0,
        'mu': mu}

    # Assemble the output dict
    outputs = {
        'q': q,
        'v': v,
        'a': a,
        # the initial conditions, which should be recovered
        'q0_rec': q0,
        'v0_rec': v0,
        # the energy and angular momentum, which should be conserved
        'T': T,
        'U': U,
        'H': H,
        'L': L}

    # Return the dicts
    return (inputs, outputs)

# ********************************************************************************************************************* 
def make_train_r2bc(n_traj: int, n_years: int, r_min: float = 0.25, r_max: float = 32.0, seed = 42):
    """
    Make a set of training data for the restricted two body problem
    INPUTS:
    n_traj: the number of trajectories to sample
    n_years: the number of years for each trajectory, e.g. 2
    r_min: minimal distance in AU, e.g. 0.25
    r_max: maximal distance in AU, e.g. 32.0    
    """
    # The sample frequency is 365 per year (dt approximately = 1 day)
    sample_freq = 365
    # Number of samples including both start and end in each trajectory
    traj_size = sample_freq*n_years + 1
    # Number of spatial dimensions
    space_dims = 2

    # Shape of arrays for various inputs and outputs
    scalar_shape = (n_traj, 1)
    init_shape = (n_traj, space_dims)
    time_shape = (n_traj, traj_size)
    traj_shape = (n_traj, traj_size, space_dims)
    
    # Initialize arrays for the data
    t = np.zeros(time_shape, dtype=np.float32)
    q0 = np.zeros(init_shape, dtype=np.float32)
    v0 = np.zeros(init_shape, dtype=np.float32)
    mu = np.zeros(scalar_shape, dtype=np.float32)
    q = np.zeros(traj_shape, dtype=np.float32)
    v = np.zeros(traj_shape, dtype=np.float32)
    a = np.zeros(traj_shape, dtype=np.float32)
    T = np.zeros(time_shape, dtype=np.float32)
    U = np.zeros(time_shape, dtype=np.float32)
    H = np.zeros(time_shape, dtype=np.float32)
    L = np.zeros(time_shape, dtype=np.float32)
    
    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Sample the trajectories
    for i in range(n_traj):
        # Sample r0
        r0 = np.random.uniform(low=r_min, high=r_max)
        # Sample theta0
        theta0 = np.random.uniform(low=-np.pi, high=np.pi)
        # Generate one trajectory
        inputs, outputs = make_traj_r2bc(r0=r0, theta0=theta0, n_years=n_years)
        
        # Copy results into main arrays
        t[i, :] = inputs['t']
        q0[i, :] = inputs['q0']
        v0[i, :] = inputs['v0']
        mu[i, :] = inputs['mu']
        q[i, :, :] = outputs['q']
        v[i, :, :] = outputs['v']
        a[i, :, :] = outputs['a']
        T[i, :] = outputs['T']
        U[i, :] = outputs['U']
        H[i, :] = outputs['H']
        L[i, :] = outputs['L']

    # Assemble the input dict
    inputs = {
        't': t,
        'q0': q0,
        'v0': v0,
        'mu': mu}

    # Assemble the output dict
    outputs = {
        'q': q,
        'v': v,
        'a': a,
        'q0_rec': q0,
        'v0_rec': v0,
        'T': T,
        'U': U,
        'H': H,
        'L': L}

    # Return the dicts
    return (inputs, outputs)

# ********************************************************************************************************************* 
def make_datasets_r2bc(n_traj, vt_split, n_years, r_min, r_max, seed, batch_size):
    """Make datasets for the restricted 2 body problem for train, val and test"""
    # Set the number of trajectories for train, validation and test
    n_traj_trn = n_traj
    n_traj_val = int(n_traj * vt_split)
    n_traj_tst = n_traj_val
    
    # Set the random seeds
    seed_trn = seed + 0
    seed_val = seed + 1
    seed_tst = seed + 2

    # Generate inputs and outputs for orbits with input parameters
    inputs_trn, outputs_trn = make_train_r2bc(n_traj=n_traj_trn, n_years=n_years, 
                                              r_min=r_min, r_max=r_max, seed=seed_trn)
    inputs_val, outputs_val = make_train_r2bc(n_traj=n_traj_val, n_years=n_years, 
                                              r_min=r_min, r_max=r_max, seed=seed_val)
    inputs_tst, outputs_tst = make_train_r2bc(n_traj=n_traj_tst, n_years=n_years, 
                                              r_min=r_min, r_max=r_max, seed=seed_tst)

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
def make_datasets_earth(n_traj=1000, vt_split=0.20, n_years=2, batch_size=64):
    """Make 3 data sets for earth-like orbits with a=1"""
    # Set the parameters for earth-like orbits
    r_min = 1.0
    r_max = 1.0
    seed = 42
    
    # Delegate to make_datasets_r2bc
    return make_datasets_r2bc(n_traj, vt_split, n_years, r_min, r_max, seed, batch_size)

# ********************************************************************************************************************* 
def make_datasets_solar(n_traj=10000, vt_split=0.20, n_years=2, batch_size=64):
    """Make 3 data sets for typical solar system orbits with a in [0.5, 32.0]"""
    # Set the parameters for solar system-like orbits
    r_min = 0.5
    r_max = 32.0
    seed = 42
    
    # Delegate to make_datasets_r2bc
    return make_datasets_r2bc(n_traj, vt_split, n_years, r_min, r_max, seed, batch_size)

