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
import shutil
from itertools import chain
from tqdm import tqdm as tqdm_console

# Local imports
from astro_utils import datetime_to_mjd, mjd_to_datetime, cart_to_sph
from horizons import datetime_to_horizons, make_sim_horizons, extend_sim_horizons, mass_tbl
from utils import rms, plot_style

# ********************************************************************************************************************* 
# Collections of objects
# The sun and 8 planets
object_names_planets = ['Sun', 'Mercury Barycenter', 'Venus Barycenter', 'Earth Barycenter', 
                        'Mars Barycenter',  'Jupiter Barycenter', 'Saturn Barycenter', 
                        'Uranus Barycenter', 'Neptune Barycenter']

# The sun, 8 planets, and the most significant moons
# See https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
object_names_moons = \
    ['Sun', 'Mercury', 'Venus', 
     'Earth', 'Moon', 
     'Mars', 'Phobos', 'Deimos',
     'Jupiter', 'Io', 'Europa', 'Ganymede', 'Callisto',
     'Saturn', 'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Iapetus',
     'Uranus', 'Ariel', 'Umbriel', 'Titania', 'Oberon', 'Miranda',
     'Neptune', 'Triton',
     'Pluto', 'Charon',
     # These objects don't have mass on Horizons; mass added with table
     'Eris', 'Makemake', 'Haumea', '2007 OR10', 'Quaoar',
     'Ceres', 'Orcus', 'Hygiea', 'Varuna', 'Varda', 'Vesta', 'Pallas', '229762', '2002 UX25'
     ]

# These are the objects tested for the moon integration
test_objects_moons = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

# Set plot style
plot_style()

# ********************************************************************************************************************* 
def make_sim(sim_name: str, object_names, t: datetime):
    """Create or load simulation with the specified objects at the specified time"""
    # Filename for archive
    file_date = t.strftime('%Y-%m-%d_%H-%M')
    fname_archive: str = f'../data/planets/{sim_name}_{file_date}.bin'

    # Convert t to a horizon date string
    horizon_date: str = datetime_to_horizons(t)

    # If this file already exists, load it and check for both extra and missing bodies
    try:
        # Attempt to load the named file
        sim = rebound.Simulation(fname_archive)
        # print(f'Loaded {fname_archive}')

        # Generate list of missing object names
        objects_missing = [nm for nm in object_names if nm not in sim.particles]

        # Extend the simulation and save it with the augmented bodies
        if objects_missing:
            print(f'Found missing objects in {fname_archive}:')
            print(objects_missing)
            extend_sim_horizons(sim, object_names = objects_missing, horizon_date=horizon_date)
            sim.simulationarchive_snapshot(filename=fname_archive)

        # Sets of named and input object hashes
        hashes_sim = set(p.hash.value for p in sim.particles)
        hashes_input = set(rebound.hash(nm).value for nm in object_names)

        # Filter the simulation so ONLY the named objects are included
        hashes_remove = [h for h in hashes_sim if h not in hashes_input]        
        for h in hashes_remove:
            sim.remove(hash=h)

    except:           
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
def make_sim_moons(t: datetime, steps_per_day: int = 4):
    """Create a simulation with the sun and 8 planets plus selected moons at the specified time"""
    # Simulation name
    sim_name = 'moons'

    # List of object names
    object_names = object_names_moons

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, t=t)
    
    # Supply missing masses
    for i, p in enumerate(sim.particles):
        if p.m == 0:
            object_name = object_names[i]
            try:
                m = mass_tbl[object_name]
                p.m = m
                # print(f'Set mass of {object_name} = {m:5.3e} Msun from lookup table.')
            except:
                print(f'Unable to find mass of {object_name}.')
    
    # Set integrator and time step
    sim.integrator = 'ias15'
    ias15 = sim.ri_ias15
    ias15.min_dt = 1.0 / steps_per_day
    
    return sim

# ********************************************************************************************************************* 
def make_archive_impl(fname_archive: str, sim_epoch: rebound.Simulation, 
                      epoch_dt: datetime, dt0: datetime, dt1: datetime, 
                      time_step: int, save_step: int):
    """
    Create a rebound simulation archive and save it to disk.
    INPUTS:
        fname_archive: the file name to save the archive to
        sim_epoch: rebound simulation object as of the epoch time; to be integrated in both directions
        epoch_dt: a datetime corresponding to sim_epoch
        dt0: the earliest datetime to simulate back to
        dt1: the latest datetime to simulate forward to
        time_step: the time step in days for the simulation
        save_step: the interval for saving snapshots to the simulation archive
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
    # ts_back = reversed(ts[:idx])
    ts_back = ts[:idx][::-1]

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
    
    # Filename for numpy arrays of position and velocity
    fname_np = fname_archive.replace('.bin', '.npz')

    # Number of snapshots
    M = len(sa_back) + len(sa_fwd)
    # Number of particles
    N = sim_epoch.N

    # Save the object name hashes
    hashes = np.zeros(N, dtype=np.uint32)
    sim_epoch.serialize_particle_data(hash=hashes)

    # Initialize arrays for the position and velocity
    shape = (M, N, 3)
    q = np.zeros(shape, dtype=np.float64)
    v = np.zeros(shape, dtype=np.float64)
    
    # Combine the forward and backward archives in forward order from t0 to t1
    sims = chain(reversed(sa_back), sa_fwd)
    # Process each simulation snapshot in turn
    for i, sim in enumerate(sims):
        # Serialize the position and velocity
        sim.serialize_particle_data(xyz=q[i])
        sim.serialize_particle_data(vxvyvz=v[i])
        # Save a snapshot on multiples of save_step
        if i % save_step == 0:
            sim.simulationarchive_snapshot(fname_archive)        

    # Save the numpy arrays with the object hashes, position and velocity
    np.savez(fname_np, hashes=hashes, q=q, v=v)
    
    # Delete the forward and backward archives
    os.remove(fname_fwd)
    os.remove(fname_back)
    
# ********************************************************************************************************************* 
def make_archive(fname_archive: str, sim_epoch: rebound.Simulation, 
                 epoch_dt: datetime, dt0: datetime, dt1: datetime, 
                 time_step: int, save_step: int = 1):
    """
    Load a rebound archive if available; otherwise generate it and save it to disk.
    INPUTS:
        fname_archive: the file name to save the archive to
        sim_epoch: rebound simulation object as of the epoch time; to be integrated in both directions
        epoch_dt: a datetime corresponding to sim_epoch
        dt0: the earliest datetime to simulate back to
        dt1: the latest datetime to simulate forward to
        time_step: the time step in days for the simulation
        save_step: the interval for saving snapshots to the simulation archive
    """
    try:
        # First try to load the named archive
        sa = rebound.SimulationArchive(filename=fname_archive)
    except:
        # If the archive is not on disk, save it to disk
        make_archive_impl(fname_archive=fname_archive, sim_epoch=sim_epoch,
                          epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step, save_step=save_step)
        # Load the new archive into memory
        sa = rebound.SimulationArchive(filename=fname_archive)
    return sa
    
# ********************************************************************************************************************* 
def load_sim_np(fname_np: str):
    """Load numpy arrays for position, velocity, and hashes from the named file"""
    # Load the numpy data file
    with np.load(fname_np) as npz:
        # Extract position, velocity and hashes
        q = npz['q']
        v = npz['v']
        hashes = npz['hashes']
    return q, v, hashes

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

    # Create list of (i, p) pairs
    if object_names is None:
        ips = enumerate(ps)
    else:
        ips = [(i, ps[object_name]) for i, object_name in enumerate(object_names)]

    # Extract the configuration of each particle
    for i, p in ips:
        cfgs[i] = np.array([p.x, p.y, p.z, p.vx, p.vy, p.vz])

    return cfgs

# ********************************************************************************************************************* 
def report_sim_difference(sim0, sim1, object_names, verbose=False):
    """Report the difference between two simulations on a summary basis"""
    # Extract configuration arrays for the two simulations
    cfg0 = sim_cfg_array(sim0, object_names)
    cfg1 = sim_cfg_array(sim1, object_names)
    
    # Convert both arrays to heliocentric coordinates
    cfg0 = cfg0 - cfg0[0:1,:]
    cfg1 = cfg1 - cfg1[0:1,:]

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

    # Error in asc and dec; convert from radians to arcseconds
    asc_err = np.degrees(np.abs(asc1-asc0)) * 3600
    dec_err = np.degrees(np.abs(dec1-dec0)) * 3600

    # Take differences
    cfg_diff = (cfg1 - cfg0)
    pos_diff = cfg_diff[:, 0:3]
    vel_diff = cfg_diff[:, 3:6]

    # Error in position and velocity in heliocentric coordinates; skip the sun
    pos_err = np.linalg.norm(pos_diff[1:], axis=1)
    pos_err_rel = pos_err / np.linalg.norm(cfg0[1:, 0:3], axis=1)
    vel_err = np.linalg.norm(vel_diff[1:], axis=1)
    vel_err_rel = vel_err / np.linalg.norm(cfg0[1:, 3:6], axis=1)

    print(f'\nPosition difference - absolute & relative')
    print(f'Body       : ASC     : DEC     : AU      : Rel     : Vel Rel')
    if verbose:
        object_names_short = [nm.replace(' Geocenter', '') for nm in object_names]
        for i, nm in enumerate(object_names_short[1:]):
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
    print(f'\nRMS Angle error:\n{ang_err:5.3f}')
    
    # Return the angle errors and RMS position error
    return asc_err, dec_err, pos_err
    
# ********************************************************************************************************************* 
def test_integration(sa, object_names, make_sim_func = make_sim_planets, make_plot=False):
    """Test the integration of the planets against Horizons data"""
    
    # Start time of simulation
    dt0: datetime = datetime(2000, 1, 1)
    
    # Dates to be tested
    # test_years = [2019, 2020, 2025, 2030, 2040]
    # test_years = list(range(2000, 2015, 5)) + list(range(2015, 2025)) + list(range(2025, 2041, 5))
    test_years = list(range(2000, 2040))
    test_dates = [datetime(year, 1, 1) for year in test_years]
    
    # Errors on these dates
    asc_errs = []
    dec_errs = []
    pos_errs = []
    
    # Test the dates
    for dt_t in test_dates:
        t = (dt_t - dt0).days
        sim_t = make_sim_func(dt_t)
        print(f'\nDifference on {dt_t}:')
        verbose = (dt_t == test_dates[-1])
        asc_err, dec_err, pos_err = report_sim_difference(sim_t, sa[t], object_names, verbose=verbose)
        asc_errs.append(asc_err)
        dec_errs.append(dec_err)
        pos_errs.append(pos_err)
    
    # Plot error summary
    asc_err_rms = np.array([rms(x) for x in asc_errs])
    dec_err_rms = np.array([rms(x) for x in dec_errs])
    ang_err_rms = rms(np.stack([asc_err_rms, dec_err_rms]), axis=0)
    pos_err_rms = np.array([rms(x) for x in pos_errs])
    if make_plot:
        fig, ax = plt.subplots(figsize=[10,8])
        ax.set_title('Angle Error vs. Time')
        ax.set_ylabel('Arcseconds')
        ax.plot(test_years, ang_err_rms, marker='o', color='blue')

        fig, ax = plt.subplots(figsize=[10,8])
        ax.set_title('Position Error vs. Time')
        ax.set_ylabel('AU')
        ax.plot(test_years, pos_err_rms, marker='o', color='red')
        
    print(f'Error by Date:')
    print('DATE       : ANG   : AU  ')
    for i, dt_t in enumerate(test_dates):
        print(f'{dt_t.date()} : {ang_err_rms[i]:5.3f} : {pos_err_rms[i]:5.3e}')
    
    # Compute average error
    mean_ang_err = np.mean(ang_err_rms)
    mean_pos_err = np.mean(pos_err_rms)
    print(f'\nMean RMS error over dates:')
    print(f'angle: {mean_ang_err:5.3f}')
    print(f'AU   : {mean_pos_err:5.3e}')
       
# ********************************************************************************************************************* 
def main():
    """Integrate the orbits of the planets and major moons"""
    
    # Reference epoch for asteroids file
    epoch_mjd: float = 58600.0
    # Convert to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040,12,31)
    
    # Create simulations snapshots of planets
    # print(f'Epoch {epoch_hrzn} / mjd={epoch_mjd}.')
    # print(f'Start {datetime_to_horizons(dt0)}.')
    # print(f'End   {datetime_to_horizons(dt1)}.')

    # Initial configuration of planets
    sim_planets = make_sim_planets(t=epoch_dt)

    # Initial configuration of moons
    steps_per_day: int = 16
    sim_moons = make_sim_moons(t=epoch_dt, steps_per_day=steps_per_day)

    # Shared time_step and save_step
    time_step: int = 1
    save_step: int = 1

    # Integrate the planets from dt0 to dt1
    fname_planets = '../data/planets/sim_planets_2000-2040.bin'
    save_step: int = 1
    sa_planets = make_archive(fname_archive=fname_planets, sim_epoch=sim_planets,
                              epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                              time_step=time_step, save_step=save_step)
    
    # Integrate the planets and moons from dt0 to dt1
    fname_moons = f'../data/planets/sim_moons_2000-2040_ias15_sf{steps_per_day}.bin'
    sa_moons = make_archive(fname_archive=fname_moons, sim_epoch=sim_moons,
                            epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                            time_step=time_step, save_step=save_step)
    # Copy file to generically named one
    fname_moons_gen = f'../data/planets/sim_moons_2000-2040.bin'
    shutil.copy(fname_moons, fname_moons_gen)
       
    # Test integration of planets
    print(f'\n***** Testing integration of sun and 8 planets. *****')
    test_integration(sa_planets, object_names_planets, make_sim_planets, True)
    
    # Test integration of planets and moons
    num_obj = sim_moons.N
    print(f'\n***** Testing integration of {num_obj} objects in solar system: sun, planets, major moons. *****')
    test_integration(sa_moons, test_objects_moons, make_sim_moons, True)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
    pass
