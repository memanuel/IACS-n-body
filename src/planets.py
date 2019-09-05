"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import rebound
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm as tqdm_console
from typing import List, Dict

# Local imports
from astro_utils import datetime_to_horizons, datetime_to_mjd, mjd_to_datetime, cart_to_sph
from utils import rms

# ********************************************************************************************************************* 
# Collections of objects
# The sun and 8 planets
object_names_planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars',  'Jupiter', 'Saturn', 'Uranus', 'Neptune']
# The sun, 8 planets, and the most significant moons
# See https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
object_names_moons = \
    ['Sun', 
     'Mercury', 'Venus', 
     'Earth Geocenter', 'Moon', 
     'Mars Geocenter', 'Phobos', 'Deimos',
     'Jupiter Geocenter', 'Io', 'Europa', 'Ganymede', 'Callisto',
     'Saturn Geocenter', 'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Iapetus',
     'Uranus Geocenter', 'Ariel', 'Umbriel', 'Titania', 'Oberon', 'Miranda',
     'Neptune Geocenter', 'Triton',
     'Pluto', 'Charon'
     # These objects don't have mass on Horizons
     # 'Eris', 'Haumea', 'Makemake', 'Quaoar'
     ]

test_objects_moons = ['Sun', 'Mercury', 'Venus', 'Earth Geocenter', 'Mars Geocenter', 
                      'Jupiter Geocenter', 'Saturn Geocenter', 'Uranus Geocenter', 'Neptune Geocenter']

# ********************************************************************************************************************* 
def object_to_horizon_names(object_names):
    """Convert from user friendly object names to Horizons names"""
    # See https://ssd.jpl.nasa.gov/horizons.cgi#top for looking up IDs
    # Initialize every name to map to itself
    object_name_to_hrzn : Dict['str', 'str'] = {nm: nm for nm in object_names}

    # Earth and the Moon
    object_name_to_hrzn['Earth Barycenter'] = '3'
    object_name_to_hrzn['Earth Geocenter'] = '399'
    object_name_to_hrzn['Moon'] = '301'
    
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
    object_name_to_hrzn['Uranus Geocenter'] = '799'

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
        # sa = rebound.SimulationArchive(fname_archive)
        # sim = sa[0]
        sim = rebound.Simulation(fname_archive)
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
    object_names = object_names_planets

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, t=t)
    
    # Set integrator and time step
    # sim.integrator = 'whfast'
    # sim.dt = 1.0

    return sim

# ********************************************************************************************************************* 
def make_sim_moons(t: datetime):
    """Create a simulation with the sun and 8 planets plus selected moons at the specified time"""
    # Simulation name
    sim_name = 'moons'

    # List of object names
    object_names = object_names_moons

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, t=t)
    
    # Set integrator and time step
    sim.integrator = 'ias15'
    # sim.dt = 1.0/1024.0
    ias15 = sim.ri_ias15
    ias15.min_dt = 1.0 / 2.0
    
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

    # Set the times for snapshots in both directions;
    ts = np.arange(t0, t1+time_step, time_step)
    idx = np.searchsorted(ts, epoch_t, side='left')
    ts_fwd = ts[idx:]
    ts_back = reversed(ts[:idx])

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
def sim_elt_array(sim, object_names=None):
    """Extract the orbital elements of each body in the simulation"""
    # Allocate array of elements
    num_objects = sim.N-1 if object_names is None else len(object_names)
    elts = np.zeros(shape=(num_objects, 9))
    
    # Iterate through particles AFTER the primary
    ps = sim.particles
    if object_names is None:
        for i, p in enumerate(ps[1:]):
            elts[i] = np.array([p.a, p.e, p.inc, p.Omega, p.omega, p.f, p.M, p.pomega, p.l])
    else:
        for i, object_name in enumerate(object_names):
            # look up the particle with this name
            p = ps[object_name]
            elts[i] = np.array([p.a, p.e, p.inc, p.Omega, p.omega, p.f, p.M, p.pomega, p.l])
    return elts

# ********************************************************************************************************************* 
def sim_cfg_array(sim, object_names=None):
    """Extract the Cartesian configuration of each body in the simulation"""
    # Allocate array of configurations
    num_objects = sim.N if object_names is None else len(object_names)
    cfgs = np.zeros(shape=(num_objects, 6))
    
    # Iterate through particles
    ps = sim.particles
    if object_names is None:
        for i, p in enumerate(ps):
            cfgs[i] = np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz])
    else:
        for i, object_name in enumerate(object_names):
            # look up the particle with this name
            p = ps[object_name]
            cfgs[i] = np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz])
    return cfgs

# ********************************************************************************************************************* 
def report_sim_difference(sim0, sim1, object_names, verbose=False):
    """Report the difference between two simulations on a summary basis"""
    # Extract configuration arrays for the two simulations
    cfg0 = sim_cfg_array(sim0, object_names)
    cfg1 = sim_cfg_array(sim1, object_names)

    # Compute RMS difference of configuration
    # cfg_rms = rms(cfg_diff, axis=0)
    # cfg_err = np.abs(cfg_diff)

    # Names of Cartesian configuration variables
    # cfg_names = ['qx', 'qy', 'qz', 'vx', 'vy', 'vz']

    # Report RMS Cartesian differences
    # print(f'\nRMS Difference of configuration:')
    # for j, var in enumerate(cfg_names):
    #    idx = np.argmax(np.abs(cfg_diff[:, j]))
    #    worse = object_names[idx]
    #    print(f'{var:2} : {cfg_rms[j]:5.2e} : {worse:10}: {cfg_err[idx, j]:+5.2e}')
    
    # Displacement of each body to earth
    try:
        earth_idx = object_names.index('Earth')
    except:
        earth_idx = object_names.index('Earth Geocenter')
    q0 = cfg0[:, 0:3] - cfg0[earth_idx, 0:3]
    q1 = cfg1[:, 0:3] - cfg1[earth_idx, 0:3]
    
    # Right Ascension and Declination
    r0, asc0, dec0 = cart_to_sph(q0)
    r1, asc1, dec1 = cart_to_sph(q1)

    # Error in asc and dec
    asc_err = np.abs(asc1-asc0)
    dec_err = np.abs(dec1-dec0)

    # Take differences
    cfg_diff = (cfg1 - cfg0)
    pos_diff = cfg_diff[:, 0:3]
    vel_diff = cfg_diff[:, 3:6]

    pos_err = np.linalg.norm(pos_diff, axis=1)
    pos_err_rel = pos_err / np.linalg.norm(cfg0[:, 0:3], axis=1)
    vel_err = np.linalg.norm(vel_diff, axis=1)
    vel_err_rel = vel_err / np.linalg.norm(cfg0[:, 3:6], axis=1)

    print(f'\nPosition difference - absolute & relative')
    print(f'Body       : ASC     : DEC     : AU      : Rel     : Vel Rel')
    if verbose:
        object_names_short = [nm.replace(' Geocenter', '') for nm in object_names]
        for i, nm in enumerate(object_names_short):
            print(f'{nm:10} : {asc_err[i]:5.2e}: {dec_err[i]:5.2e}: {pos_err[i]:5.2e}: '
                  f'{pos_err_rel[i]:5.2e}: {vel_err_rel[i]:5.2e}')
    print(f'Overall    : {rms(asc_err):5.2e}: {rms(dec_err):5.2e}: {rms(pos_err):5.2e}: '
          f'{rms(pos_err_rel):5.2e}: {rms(vel_err_rel):5.2e}')

    # Extract orbital element arrays from the two simulations
    elt0 = sim_elt_array(sim0, object_names[1:])
    elt1 = sim_elt_array(sim1, object_names[1:])

    # Take differences
    elt_diff = (elt1 - elt0)
    # Compute RMS difference by orbital element
    elt_rms = rms(elt_diff, axis=0)
    elt_err = np.abs(elt_diff)

    # Names of selected elements
    elt_names = ['a', 'e', 'inc', 'Omega', 'omega', 'f', 'M', 'pomega', 'long']

    # Report RMS orbital element differences
    print(f'\nOrbital element errors:')
    if verbose:
        print(f'elt    : RMS      : worst      : max_err  : HRZN        : REB')
        for j, elt in enumerate(elt_names):
            idx = np.argmax(elt_err[:, j])
            worse = object_names_short[idx+1]
            print(f'{elt:6} : {elt_rms[j]:5.2e} : {worse:10} : {elt_err[idx, j]:5.2e} : '
                  f'{elt0[idx, j]:11.8f} : {elt1[idx, j]:11.8f}')
    print(f'RMS (a, e, inc) =          {rms(elt_diff[:,0:3]):5.2e}')
    print(f'RMS (f, M, pomega, long) = {rms(elt_diff[:,5:9]):5.2e}')      

    # One summary error statistic
    ang_err = rms(np.array([asc_err, dec_err]))
    print(f'\nRMS Angle error:\n{ang_err:5.2e}')
    
    # Return the angle errors
    return asc_err, dec_err
    
# ********************************************************************************************************************* 
def test_integration(sa, object_names, make_sim_func = make_sim_planets, make_plot=False):
    """Test the integration of the planets against Horizons data"""
    
    # Start time of simulation
    dt0: datetime = datetime(2000, 1, 1)
    
    # Dates to be tested
    test_years = [2040]
    # test_years = list(range(2000, 2015, 5)) + list(range(2015, 2025)) + list(range(2025, 2041, 5))
    test_dates = [datetime(year, 1, 1) for year in test_years]
    
    # Errors on these dates
    asc_errs = []
    dec_errs = []
    
    # Test the dates
    for dt_t in test_dates:
        t = (dt_t - dt0).days
        sim_t = make_sim_func(dt_t)
        print(f'\nDifference on {dt_t}:')
        asc_err, dec_err = report_sim_difference(sim_t, sa[t], object_names, True)
        asc_errs.append(asc_err)
        dec_errs.append(dec_err)
    
    # Plot error summary
    asc_err_rms = np.array([rms(x) for x in asc_errs])
    dec_err_rms = np.array([rms(x) for x in dec_errs])
    ang_err_rms = rms(np.stack([asc_err_rms, dec_err_rms]), axis=0)
    if make_plot:
        fig, ax = plt.subplots()
        ax.set_title('Angle Error vs. Time')
        ax.plot(test_years, ang_err_rms, marker='o', color='blue')
        
    print(f'\nAngle Error by Date:')
    for i, dt_t in enumerate(test_dates):
        # print(f'{dt_t.year}-{dt_t.month:2}-{dt_t.day:2} : {ang_err_rms[i]:5.2e}')
        print(f'{dt_t.date()} : {ang_err_rms[i]:5.2e}')
       
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
    sim_planets = make_sim_planets(epoch_dt)
    sim_moons = make_sim_moons(epoch_dt)
    
    time_step = 1
    # Integrate the planets from dt0 to dt1
    fname_planets = '../data/planets/sim_planets_2000-2040.bin'
    sa_planets = make_archive(fname_archive=fname_planets, sim_epoch=sim_planets,
                              epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step)
    
    # Integrate the planets and moons from dt0 to dt1
    fname_moons = '../data/planets/sim_moons_2000-2040_ias15_dt2.bin'
    sa_moons = make_archive(fname_archive=fname_moons, sim_epoch=sim_moons,
                           epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step)
       
    # Test integration of planets
    test_integration(sa_planets, object_names_planets, make_sim_planets, True)
    # test_integration(sa_moons, test_objects_moons, make_sim_moons, True)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
    pass
