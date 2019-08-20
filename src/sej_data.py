"""
Harvard IACS Masters Thesis
Three Body Problem - Perturbed Sun-Earth-Jupiter System
Generate training data (trajectories)

Michael S. Emanuel
Tue Aug 20 16:40:29 2019
"""

# Library imports
# import tensorflow as tf
import rebound
import numpy as np
# import os
# import zlib
# import pickle
from tqdm.auto import tqdm
from typing import List

# Aliases
# keras = tf.keras

# ********************************************************************************************************************* 
def make_sim_horizons(object_names: List[str], horizon_date: str):
    """Create a new simulation with initial data from the NASA Horizons system"""
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
    """Create a simulation archive and save it to disk"""
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
        print(f'Found simulation archive {fname_archive}')
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
def main():
    """Main routine for making SEJ data sets"""
    # Make the simulation archive for the unperturbed (real) sun-earth-jupiter system
    n_years = 100
    sample_freq = 10
    make_sa_sej(n_years=n_years, sample_freq=sample_freq)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
