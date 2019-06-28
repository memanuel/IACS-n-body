"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Generate and plot training data (trajectories)

Michael S. Emanuel
Tue Jun  18 15:29 2019
"""

# Library imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

# ********************************************************************************************************************* 
def plot_orbit_q(data):
    """Plot the orbit position in a training sample"""
    # Unpack data
    t = data['t']
    q = data['q']
    qx = q[:, 0]
    qy = q[:, 1]
    # Compute the distance r
    r = np.linalg.norm(q, axis=1)

    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Position')
    ax.set_xlabel('t - years')
    ax.set_ylabel('q - AU')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, qx, color='b', label='qx')
    ax.plot(t, qy, color='r', label='qy')
    ax.plot(t, r,  color='purple', label='r')
    ax.grid()
    ax.legend()
    
    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_v(data):
    """Plot the orbit velocity in a training sample"""
    # Unpack data
    t = data['t']
    v = data['v']
    vx = v[:, 0]
    vy = v[:, 1]
    spd = np.linalg.norm(v, axis=1)
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Velocity')
    ax.set_xlabel('t - years')
    ax.set_ylabel('v - AU / year')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, vx, color='b', label='vx')
    ax.plot(t, vy, color='r', label='vy')
    ax.plot(t, spd, color='purple', label='spd')
    ax.grid()
    ax.legend()

    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_a(data):
    """Plot the orbit acceleration in a training sample"""
    # Unpack data
    t = data['t']
    a = data['a']
    ax = a[:, 0]
    ay = a[:, 1]
    acc = np.linalg.norm(a, axis=1)
    
    # Plot the x and y coordinate
    # Name the axes object ax_ rather than ax to avoid a name collision with the x component of acceleration, ax
    fig, ax_ = plt.subplots(figsize=[16, 9])
    ax_.set_title('Orbit Acceleration')
    ax_.set_xlabel('t - years')
    ax_.set_ylabel('a - $AU / year^2$')
    ax_.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax_.plot(t, ax, color='b', label='ax')
    ax_.plot(t, ay, color='r', label='ay')
    ax_.plot(t, acc, color='purple', label='acc')
    ax_.grid()
    ax_.legend()

    return fig, ax_

# ********************************************************************************************************************* 
def plot_orbit_energy(data):
    """Plot the orbit energy in a training sample"""
    # Unpack data
    t = data['t']
    q = data['q']
    v = data['v']
    r = np.linalg.norm(q, axis=1)
    spd = np.linalg.norm(v, axis=1)
    
    # Compute the kinetic energy over m1
    T = 0.5 * spd * spd
    
    # Compute the potential energy over m1
    mu = (2.0 * np.pi)**2
    U = -mu / r
    
    # The total energy
    E = T + U
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Energy')
    ax.set_xlabel('t - years')
    ax.set_ylabel('Energy / m1 in $(au/year)^2$')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, T, color='b', label='T')
    ax.plot(t, U, color='r', label='U')
    ax.plot(t, E, color='purple', label='E')
    ax.grid()
    ax.legend()

    return fig, ax

