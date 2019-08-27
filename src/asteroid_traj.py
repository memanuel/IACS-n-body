"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import rebound
from datetime import datetime
import os
from tqdm import tqdm as tqdm_console
from typing import List

# Local imports
from astro_utils import mjd_to_horizons, datetime_to_horizons, datetime_to_mjd, mjd_to_datetime
from utils import rms

# ********************************************************************************************************************* 
def make_sim_horizons(object_names: List[str], horizon_date: str):
    """Create a new rebound simulation with initial data from the NASA Horizons system"""
    # Create a simulation
    sim = rebound.Simulation()
    
    # Set units
    sim.units = ('day', 'AU', 'Msun')
    
    # Add these objects from Horizons
    sim.add(object_names, date=horizon_date)
          
    # Add hashes for the object names
    for i, particle in enumerate(sim.particles):
        particle.hash = rebound.hash(object_names[i])
        
    # Move to center of mass
    sim.move_to_com()
    
    return sim

# ********************************************************************************************************************* 
def make_sim_planets(t: datetime):
    """Create a simulation with the sun and 8 planets at the specified time"""
    # Filename for archive
    file_date = t.strftime('%Y-%m-%d_%H-%M')
    fname_archive: str = f'../data/asteroid/sim_planets_{file_date}'

    # If this file already exists, load and return it
    try:
        sa = rebound.SimulationArchive(fname_archive)
        sim = sa[0]
    except:        
        # List of object names
        # Use codes Earth = 399 and Moon = 301; otherwise 'Earth' returns Earth-Moon Barycenter
        object_names = ['Sun', 'Mercury', 'Venus', '399', '301', 
                        'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

        # Convert t to a horizon date string
        horizon_date: str = datetime_to_horizons(t)
    
        # Initialize simulation
        sim = make_sim_horizons(object_names=object_names, horizon_date=horizon_date)

        # Move to center of momentum
        sim.move_to_com()

        # Save a snapshot to the archive file
        sim.simulationarchive_snapshot(filename=fname_archive)

    # Return the simulation
    return sim

# ********************************************************************************************************************* 
def make_archive_impl(fname_archive: str, sim_epoch: rebound.Simulation, 
                      epoch_dt: datetime, dt0: datetime, dt1: datetime, 
                      time_step: int):
    """
    Create a rebound simulation archive and save it to disk.
    INPUTS:
        fname_archive: the file name to save the archive to
        sim_epoch: rebound simulation object as of the epoch time; to be integrated in both directions
        epoch_dt: a datetime corresponding to sim_epoch
        dt0: the earliest datetime to simulate back to
        dt1: the latest datetime to simulate forward to
        time_step: the time step in days for the simulation
    """
    
    # Convert epoch, start and end times relative to a base date of the simulation start
    # This way, time is indexed from t0=0 to t1 = (dt1-dt0)
    epoch_t = datetime_to_mjd(epoch_dt, dt0)
    t0 = datetime_to_mjd(dt0, dt0)
    t1 = datetime_to_mjd(dt1, dt0)
    
    # Create copies of the simulation to integrate forward and backward
    sim_fwd = sim_epoch.copy()
    sim_back = sim_epoch.copy()
    
    # Set the time counter on both simulation copies to the epoch time
    sim_fwd.t = epoch_t
    sim_back.t = epoch_t
    
    # Set the times for snapshots in both directions
    ts_fwd = np.arange(epoch_t, t1+time_step, time_step)
    ts_back = np.arange(epoch_t, t0-time_step, -time_step)

    # File names for forward and backward integrations
    fname_fwd = fname_archive.replace('.bin', '_fwd.bin')
    fname_back = fname_archive.replace('.bin', '_back.bin')

    # Integrate the simulation forward in time
    for t in tqdm_console(ts_fwd):
        # Integrate to the current time step with an exact finish time
        sim_fwd.integrate(t, exact_finish_time=1)
        # Save a snapshot to the archive file
        sim_fwd.simulationarchive_snapshot(filename=fname_fwd)

    # Integrate the simulation backward in time
    for t in tqdm_console(ts_back):
        # Integrate to the current time step with an exact finish time
        sim_back.integrate(t, exact_finish_time=1)
        # Save a snapshot to the archive file
        sim_back.simulationarchive_snapshot(filename=fname_back)

    # Load the archives with the forward and backward snapshots
    sa_fwd = rebound.SimulationArchive(fname_fwd)
    sa_back = rebound.SimulationArchive(fname_back)
    
    # Combine the forward and backward archives in forward order from t0 to t1
    for sim in reversed(sa_back):
        sim.simulationarchive_snapshot(fname_archive)
    for sim in sa_fwd:
        sim.simulationarchive_snapshot(fname_archive)

    # Load the updated simulation archive
    sa = rebound.SimulationArchive(fname_archive)
    
    # Delete the forward and backward archives
    os.remove(fname_fwd)
    os.remove(fname_back)
    
    # Return the combined simulation archive
    return sa

# ********************************************************************************************************************* 
def make_archive(fname_archive: str, sim_epoch: rebound.Simulation, 
                 epoch_dt: datetime, dt0: datetime, dt1: datetime, 
                 time_step: int):
    """
    Load a rebound archive if available; otherwise generate it and save it to disk.
    INPUTS:
        fname_archive: the file name to save the archive to
        sim_epoch: rebound simulation object as of the epoch time; to be integrated in both directions
        epoch_dt: a datetime corresponding to sim_epoch
        dt0: the earliest datetime to simulate back to
        dt1: the latest datetime to simulate forward to
        time_step: the time step in days for the simulation
    """
    try:
        sa = rebound.SimulationArchive(filename=fname_archive)
    except:
        sa = make_archive_impl(fname_archive=fname_archive, sim_epoch=sim_epoch,
                               epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step)
    
    return sa
    
# ********************************************************************************************************************* 
def make_sa_ceres_v1(n_years: int, sample_freq:int ):
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
        Epoch = 58600.0
        # jd = mjd_to_jd(Epoch)
        # horizon_date = f'JD{jd:.8f}'
        horizon_date = mjd_to_horizons(Epoch)
        
        # Initialize simulation
        sim = make_sim_horizons(object_names=object_names, horizon_date=horizon_date)
        
        # Create a simulation archive from this simulation
        sa = make_archive(fname_archive=fname_archive, sim=sim, 
                          n_years=n_years, sample_freq=sample_freq)
    
    return sa

# ********************************************************************************************************************* 
def sim_cfg_array(sim):
    """Extract the Cartesian configuration of each body in the simulation"""
    # Allocate array of elements
    cfgs = np.zeros(shape=(sim.N-1, 6))
    
    # Iterate through particles AFTER the primary
    ps = sim.particles
    for i, p in enumerate(ps[1:]):
        cfgs[i] = np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz])
        
    return cfgs

# ********************************************************************************************************************* 
def sim_elt_array(sim):
    """Extract the orbital elements of each body in the simulation"""
    # Allocate array of elements
    elts = np.zeros(shape=(sim.N-1, 6))
    
    # Iterate through particles AFTER the primary
    ps = sim.particles
    for i, p in enumerate(ps[1:]):
        elts[i] = np.array([p.a, p.e, p.inc, p.Omega, p.omega, p.f])
        
    return elts

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
    jd = mjd_to_jd(Epoch)
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
def report_sim_difference(sim0, sim1):
    """Generate selected asteroid simulations"""
    # Extract orbital element arrays directrly from Horizons
    elt0 = sim_elt_array(sim0)
    elt1 = sim_elt_array(sim1)

    # Take differences
    elt_diff = (elt1- elt0)

    # Compute RMS difference by orbital element
    elt_rms = rms(elt_diff, axis=0)

    # Names of selected elements
    elt_names = ['a', 'e', 'inc', 'Omega', 'omega', 'f']

    # Report RMS orbital element differences
    print('\nRMS Difference of elements:')
    for i, elt in enumerate(elt_names):
        print(f'{elt:5} : {elt_rms[i]:5.2e}')

    # Extract orbital element arrays directrly from Horizons
    cfg0 = sim_cfg_array(sim0)
    cfg1 = sim_cfg_array(sim1)

    # Take differences
    cfg_diff = (cfg1 - cfg0)

    # Compute RMS difference of configuration
    cfg_rms = rms(cfg_diff, axis=0)

    # Names of Cartesian configuration variables
    cfg_names = ['qx', 'qy', 'qz', 'vx', 'vy', 'vz']

    # Report RMS Cartesian differences
    print('\nRMS Difference of configuration:')
    for i, var in enumerate(cfg_names):
        print(f'{var:2} : {cfg_rms[i]:5.2e}')

# ********************************************************************************************************************* 
def test_planet_integration(sa, sim0, sim1):
    """Generate selected asteroid simulations"""
    
    print(f'Difference at start t0:')
    report_sim_difference(sim0, sa[0])
    
    print(f'Difference at start t1:')
    report_sim_difference(sim1, sa[-1])
    
# ********************************************************************************************************************* 
def main():
    """Generate selected asteroid simulations"""
    
    # Reference epoch for asteroids file
    epoch_mjd: float = 58600.0
    # Convert to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    epoch_hrzn: str = datetime_to_horizons(epoch_dt)
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040, 1, 1)
    
    # Create simulations snapshots of planets
    print(f'Epoch {epoch_hrzn} / mjd={epoch_mjd}.')
    sim_epoch = make_sim_planets(epoch_dt)
    print(f'Start {datetime_to_horizons(dt0)}.')
    sim0 = make_sim_planets(dt0)
    print(f'End   {datetime_to_horizons(dt1)}.')
    sim1 = make_sim_planets(dt1)
    
    # Integrate the planets from dt0 to dt1
    fname_archive = '../data/asteroid/sim_planets_2000_2040.bin'
    time_step = 1
    sa = make_archive(fname_archive=fname_archive, sim_epoch=sim_epoch,
                      epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step)
   
    # Test integration of planetrs
    test_planet_integration(sa, sim0, sim1)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
    pass


