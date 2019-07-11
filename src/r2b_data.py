"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Generate training data (trajectories)

Michael S. Emanuel
Thu Jul 11 16:28:58 2019
"""

# Library imports
import tensorflow as tf
import rebound
import numpy as np
from tqdm.auto import tqdm

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def make_traj_r2b(a, e, inc, Omega, omega, f, n_years):
    """
    Make an array of training data points for the restricted 2 body problem.
    Creates one trajectory with the same initial configuration.
    INPUTS:
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

    # Set integrator to whfast: will be perfect for 2 body problem
    sim.integrator = 'whfast'

    # Set the simulation time step based on sample_freq
    sim.dt = 1.0 / sample_freq

    # Add primary with 1 solar mass at origin with 0 velocity
    sim.add(m=1.0)

    # Add the orbiting body
    sim.add(m=0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, f=f)

    # Initialize cartesian entries to zero vectors; these are placeholders
    q = np.zeros((N,3), dtype=np.float32)
    v = np.zeros((N,3), dtype=np.float32)
    a = np.zeros((N,3), dtype=np.float32)

    # Initialize placeholders for kinetic, potential, and total energy
    T = np.zeros(N, dtype=np.float32)
    U = np.zeros(N, dtype=np.float32)

    # Initialize angular momentum
    L = np.zeros((N,3), dtype=np.float32)

    # The gravitational constant mu as a scalar; assume the small particles have mass 0
    mu = sim.G * sim.particles[0].m

    # Set the times for snapshots
    ts = np.linspace(0.0, n_years, N)

    # The orbiting body
    p = sim.particles[1]

    # Simulate the orbits
    for i, t in enumerate(ts):
        # Take one time step; this is one day
        sim.integrate(t, exact_finish_time=0)
        # Save the position
        q[i] = [p.x, p.y, p.z]
        # Save the velocity
        v[i] = [p.vx, p.vy, p.vz]
        # Save the acceleration
        a[i] = [p.ax, p.ay, p.az]

        # Save the energy terms
        T[i] = 0.5 * np.sum(v[i] * v[i])
        U[i] = -mu / np.linalg.norm(q[i])

        # Save the angular momentum vector
        L[i] = np.cross(q[i], v[i])

    # The total energy is the sum of kinetic and potential
    H = T + U
    
    # The initial position and velocity
    q0 = q[0]
    v0 = v[0]
    
    # Assemble the input dict
    inputs = {
        't': ts,
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
def make_train_r2b(n_traj: int, n_years: int, a_min: float = 0.50, a_max: float = 32.0, 
                   e_max = 0.20, inc_max = 0.0, seed = 42):
    """
    Make a set of training data for the restricted two body problem
    INPUTS:
    n_traj: the number of trajectories to sample
    n_years: the number of years for each trajectory, e.g. 2
    a_min: minimal semi-major axis in AU, e.g. 0.50
    a_max: maximal semi-major axis in AU, e.g. 32.0
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
    scalar_shape = (n_traj, 1)
    init_shape = (n_traj, space_dims)
    time_shape = (n_traj, traj_size)
    traj_shape = (n_traj, traj_size, space_dims)
    
    # Set random seed for reproducible results
    np.random.seed(seed=seed)

    # Initialize orbital element by sampling according to the inputs
    orb_a = np.random.uniform(low=a_min, high=a_max, size=n_traj).astype(np.float32)
    orb_e = np.random.uniform(low=0.0, high=e_max, size=n_traj).astype(np.float32)
    orb_inc = np.random.uniform(low=0.0, high=inc_max, size=n_traj).astype(np.float32)
    orb_Omega = np.random.uniform(low=-np.pi, high=np.pi, size=n_traj).astype(np.float32)
    orb_omega = np.random.uniform(low=-np.pi, high=np.pi, size=n_traj).astype(np.float32)
    orb_f = np.random.uniform(low=-np.pi, high=np.pi, size=n_traj).astype(np.float32)

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
    L = np.zeros(traj_shape, dtype=np.float32)
    
    # Sample the trajectories
    for i in tqdm(range(n_traj)):
        # Generate one trajectory
        inputs, outputs = make_traj_r2b(a=orb_a[i], e=orb_e[i], inc=orb_inc[i], 
                                        Omega=orb_Omega[i], omega=orb_omega[i], 
                                        f=orb_f[i], n_years=n_years)
        
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