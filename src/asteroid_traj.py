"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import rebound
import tqdm
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
# Create simulation using horizons
sim = rebound.Simulation()
object_names = ['Sun', 'Ceres']
horizon_date = '2000-01-01 12:00'
