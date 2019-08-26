"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import rebound
# from tqdm.auto import tqdm as tqdm_auto
from tqdm import tqdm as tqdm_console
from typing import List

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
    for i, t in tqdm_console(enumerate(ts)):
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
def make_sa_ceres(n_years: int, sample_freq:int ):
    """Create or load the sun-earth-jupiter system at start of J2000.0 frame"""
    
    # The name of the simulation archive
    fname_archive = '../data/asteroid/ss_ceres.bin'
    
    # If this file already exists, load and return it
    try:
        sa = rebound.SimulationArchive(fname_archive)
        # print(f'Found simulation archive {fname_archive}')
    except:
        # Initialize a new simulation
        object_names = ['Sun', 'Ceres']
        
        # The epoch as a horizon date string
        Epoch = 58600
        jd = Epoch + 2400000.5
        horizon_date = f'JD{jd:.8f}'
        
        # Initialize simulation
        sim = make_sim_horizons(object_names=object_names, horizon_date=horizon_date)
        
        # Create a simulation archive from this simulation
        sa = make_archive(fname_archive=fname_archive, sim=sim, 
                          n_years=n_years, sample_freq=sample_freq)
    
    return sa

# ********************************************************************************************************************* 
def test_element_recovery():
    # Load data from file
    Epoch = 58600
    a = 2.7691652
    e = 0.07600903
    i_deg = 10.59407
    w_deg = 73.59769
    Node_deg=80.30553
    M_deg=77.3720959
    
    # Date conversion
    jd = Epoch + 2400000.5
    horizon_date = f'JD{jd:.8f}'
    
    # Convert from degrees to radians
    inc = np.radians(i_deg)
    Omega = np.radians(Node_deg)
    omega = np.radians(w_deg)
    M=np.radians(M_deg)
    
    # Create simulation using Horizons in automatic mode
    # sim1 = rebound.Simulation()
    # sim1.add('Solar System Barycenter', date=horizon_date)
    # sim1.add('Sun', date=horizon_date)
    # sim1.add('Ceres', date=horizon_date)
    sa = make_sa_ceres(1, 1)
    sim1 = sa[0]    
    p1 = sim1.particles[1]
    orb1 = p1.calculate_orbit()
    
    # Create simulation using data from file
    sim2 = rebound.Simulation()
    # sim2.add('Sun', date=horizon_date)
    sim2.add(sim1.particles[0])
    sim2.add(m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M)
    p2 = sim2.particles[1]
    orb2 = p2.calculate_orbit()

    # Compute errors
    print(f'Errors in recovered orbital elements:')
    print(f'a    : {np.abs(orb2.a - orb1.a):5.3e}')
    print(f'e    : {np.abs(orb2.e - orb1.e):5.3e}')
    print(f'inc  : {np.abs(orb2.inc - orb1.inc):5.3e}')
    print(f'Omega: {np.abs(orb2.Omega - orb1.Omega):5.3e}')
    print(f'omega: {np.abs(orb2.omega - orb1.omega):5.3e}')
    print(f'f    : {np.abs(orb2.f - orb1.f):5.3e}')

# ********************************************************************************************************************* 
# Create simulation using horizons
sa = make_sa_ceres(1, 1)
# sim = sa[0]

test_element_recovery()
