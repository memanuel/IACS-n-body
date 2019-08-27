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
from typing import List, Dict

# Local imports
from astro_utils import mjd_to_horizons, datetime_to_horizons, datetime_to_mjd, mjd_to_datetime
from utils import rms

# ********************************************************************************************************************* 
def object_to_horizon_names(object_names):
    """Convert from user friendly object names to Horizons names"""
    # See https://ssd.jpl.nasa.gov/horizons.cgi#top for looking up IDs
    # Initialize every name to map to itself
    object_name_to_hrzn : Dict['str', 'str'] = {nm: nm for nm in object_names}

    # Earth and the Moon
    object_name_to_hrzn['Earth Barycenter'] = '3'
    object_name_to_hrzn['Earth Geocenter'] = '399'
    # object_name_to_hrzn['Moon'] = '301'
    
    # Mars and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Mars
    # object_name_to_hrzn['Mars'] = '4'
    object_name_to_hrzn['Mars Barycenter'] = '499'
    object_name_to_hrzn['Mars Geocenter'] = '499'
    # object_name_to_hrzn['Phobos'] = '401'
    # object_name_to_hrzn['Deimos'] = '402'

    # Jupiter and its moons
    # https://en.wikipedia.org/wiki/Galilean_moons
    object_name_to_hrzn['Jupiter Barycenter'] = '5'
    object_name_to_hrzn['Jupiter Geocenter'] = '599'
    # object_name_to_hrzn['Io'] = '501'
    # object_name_to_hrzn['Europa'] = '502'
    # object_name_to_hrzn['Ganymede'] = '503'
    # object_name_to_hrzn['Callisto'] = '504'

    # Saturn and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Saturn
    object_name_to_hrzn['Saturn Barycenter'] = '6'
    object_name_to_hrzn['Saturn Geocenter'] = '699'
    
    # Uranus and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Uranus
    object_name_to_hrzn['Uranus Barycenter'] = '7'
    object_name_to_hrzn['Uranus Geonenter'] = '799'

    # Neptune and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Neptune
    object_name_to_hrzn['Neptune Barycenter'] = '8'
    object_name_to_hrzn['Neptune Geocenter'] = '899'

    # Pluto and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Neptune
    object_name_to_hrzn['Pluto Barycenter'] = '9'
    object_name_to_hrzn['Pluto Geocenter'] = '999'

    # List of horizon object names corresponding to these input object names
    horizon_names = [object_name_to_hrzn[nm] for nm in object_names]

    return horizon_names

# ********************************************************************************************************************* 
def make_sim_horizons(object_names: List[str], horizon_date: str):
    """Create a new rebound simulation with initial data from the NASA Horizons system"""
    # Create a simulation
    sim = rebound.Simulation()
    
    # Set units
    sim.units = ('day', 'AU', 'Msun')
    
    # Convert from user friendly object names to Horizons names
    horizon_names = object_to_horizon_names(object_names)

    # Add these objects from Horizons
    print(f'Searching Horizons as of {horizon_date}.')
    sim.add(horizon_names, date=horizon_date)
    
    # Add hashes for the object names
    for i, particle in enumerate(sim.particles):
        particle.hash = rebound.hash(object_names[i])
        
    # Move to center of mass
    sim.move_to_com()
    
    return sim

# ********************************************************************************************************************* 
def make_sim(sim_name: str, object_names, t: datetime):
    """Create or load simulation with the specified objects at the specified time"""
    # Filename for archive
    file_date = t.strftime('%Y-%m-%d_%H-%M')
    fname_archive: str = f'../data/planets/{sim_name}_{file_date}.bin'

    # If this file already exists, load and return it
    try:
        sa = rebound.SimulationArchive(fname_archive)
        sim = sa[0]
    except:        
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
def make_sim_planets(t: datetime):
    """Create a simulation with the sun and 8 planets at the specified time"""
    # Simulation name
    sim_name = 'planets'

    # List of object names
    object_names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 
                    'Jupiter', 'Saturn', 'Uranus', 'Neptune']

    # Return the simulation
    return make_sim(sim_name=sim_name, object_names=object_names, t=t)

# ********************************************************************************************************************* 
def make_sim_moons(t: datetime):
    """Create a simulation with the sun and 8 planets plus selected moons at the specified time"""
    # Simulation name
    sim_name = 'moons'

    # List of object names
    object_names = \
    ['Sun', 
     'Mercury', 'Venus', 
     'Earth Geocenter', 'Moon', 
     'Mars Geocenter', 'Phobos', 'Deimos',
     'Jupiter Geocenter', 'Io', 'Europa', 'Ganymede', 'Callisto',
     'Saturn Geocenter', 'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Iapetus',
     'Uranus', 'Ariel', 'Unbriel', 'Titania', 'Oberon', 'Miranda',
     'Neptune', 'Triton',
     'Pluto']

    # Return the simulation
    return make_sim(sim_name=sim_name, object_names=object_names, t=t)

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
def sim_cfg_array(sim):
    """Extract the Cartesian configuration of each body in the simulation"""
    # Allocate array of elements
    cfgs = np.zeros(shape=(sim.N, 6))
    
    # Iterate through particles AFTER the primary
    ps = sim.particles
    for i, p in enumerate(ps):
        cfgs[i] = np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz])
        
    return cfgs

# ********************************************************************************************************************* 
def report_sim_difference(sim0, sim1, object_names, object_name):
    """Generate selected asteroid simulations"""
    # Extract orbital element arrays directrly from Horizons
    elt0 = sim_elt_array(sim0)
    elt1 = sim_elt_array(sim1)

    # Take differences
    elt_diff = (elt1- elt0)
    # Compute RMS difference by orbital element
    elt_rms = rms(elt_diff, axis=0)

    # Body i
    object_num = object_names.index(object_name)
    i = object_num-1
    elt_i = elt_diff[object_num-1]

    # Names of selected elements
    elt_names = ['a', 'e', 'inc', 'Omega', 'omega', 'f']

    # Report RMS orbital element differences
    print(f'RMS Difference of elements and body {object_name}:')
    for j, elt in enumerate(elt_names):
        idx = np.argmax(np.abs(elt_diff[j]))+1
        worse = object_names[idx]
        print(f'{elt:5} : {elt_rms[j]:5.2e} : {worse:10} : {elt_i[j]:+5.2e} : {elt0[i, j]:11.8f} : {elt1[i, j]:11.8f}')
    print(f'Overall RMS = {rms(elt_diff):5.2e}')

    # Extract orbital element arrays directrly from Horizons
    cfg0 = sim_cfg_array(sim0)
    cfg1 = sim_cfg_array(sim1)

    # Take differences
    cfg_diff = (cfg1 - cfg0)
    # pos_dif = cif_diff[:, 0:3]
    # Compute RMS difference of configuration
    cfg_rms = rms(cfg_diff, axis=0)
    # Body i
    cfg_i = cfg_diff[object_num]

    # Names of Cartesian configuration variables
    cfg_names = ['qx', 'qy', 'qz', 'vx', 'vy', 'vz']

    # Report RMS Cartesian differences
    print(f'\nRMS Difference of configuration and {object_name}:')
    i = object_num
    for j, var in enumerate(cfg_names):
        idx = np.argmax(np.abs(cfg_diff[j]))
        worse = object_names[idx]
        print(f'{var:2} : {cfg_rms[j]:5.2e} : {worse:10}: {cfg_i[j]:+5.2e} : {cfg0[i, j]:11.8f} : {cfg1[i, j]:11.8f}')
    print(f'Overall RMS = {rms(cfg_diff):5.2e}')

# ********************************************************************************************************************* 
def test_planet_integration(sa):
    """Generate selected asteroid simulations"""
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040, 1, 1)
    
    sim0 = make_sim_planets(dt0)
    sim1 = make_sim_planets(dt1)

    # Planet to display
    object_names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Moon', 
                    'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']    
    object_name = 'Earth'

    # print(f'\nDifference at start t0:')
    # report_sim_difference(sim0, sa[0], planet_num)
    
    print(f'\nDifference at end t1:')
    report_sim_difference(sim1, sa[-1], object_names, object_name)
    
    # test_years = [2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035, 2040]
    # test_years = [2018, 2019, 2020]
    test_years = []
    for year in test_years:
        dt_t = datetime(year, 1, 1)
        t = (dt_t - dt0).days
        sim_t = make_sim_planets(dt_t)
        print(f'\nDifference on January 1, {year}:')
        report_sim_difference(sim_t, sa[t], object_name)
    
# ********************************************************************************************************************* 
def main():
    """Generate selected asteroid simulations"""
    
    # Reference epoch for asteroids file
    epoch_mjd: float = 58600.0
    # Convert to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040, 1, 1)
    
    # Create simulations snapshots of planets
    # print(f'Epoch {epoch_hrzn} / mjd={epoch_mjd}.')
    # print(f'Start {datetime_to_horizons(dt0)}.')
    # print(f'End   {datetime_to_horizons(dt1)}.')
    sim_epoch = make_sim_planets(epoch_dt)
    sim_epoch = make_sim_moons(epoch_dt)

    # Integrate the planets from dt0 to dt1
    fname_archive = '../data/asteroid/sim_planets_2000_2040.bin'
    time_step = 1
    sa = make_archive(fname_archive=fname_archive, sim_epoch=sim_epoch,
                      epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step)
   
    # Test integration of planetrs
    # test_planet_integration(sa)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
    pass


