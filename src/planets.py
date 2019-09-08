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
# from tqdm.auto import tqdm as tqdm_console
import argparse
from typing import List

# Local imports
from astro_utils import datetime_to_mjd, mjd_to_datetime, cart_to_sph
from horizons import make_sim_horizons, extend_sim_horizons
from utils import rms, plot_style

# ********************************************************************************************************************* 
# Set plot style
plot_style()

# ********************************************************************************************************************* 
# Collections of objects

# The sun and 8 planets
planets = [
    'Sun', 'Mercury Barycenter', 'Venus Barycenter', 
    'Earth', 'Moon',
    'Mars Barycenter',  'Jupiter Barycenter', 'Saturn Barycenter', 
    'Uranus Barycenter', 'Neptune Barycenter']
object_names_planets = planets
test_objects_planets = object_names_planets

# The sun, 8 planets, and the most significant moons
# See https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
moons = [
    'Sun', 
    'Mercury', 
    'Venus', 
    'Earth', 'Moon', 
    'Mars Barycenter',
    'Jupiter', 'Io', 'Europa', 'Ganymede', 'Callisto',
    'Saturn', 'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Iapetus', 'Phoebe',
    'Uranus', 'Ariel', 'Umbriel', 'Titania', 'Oberon', 'Miranda',
    'Neptune', 'Triton', 'Proteus',
    'Pluto', 'Charon']
object_names_moons = moons
# These are the objects tested for the moon integration
test_objects_moons = [
    'Sun', 'Mercury', 'Venus', 'Earth', 'Mars Barycenter', 
    'Jupiter', 'Saturn', 'Uranus', 'Neptune']

# Planet barycenters and selected dwarf planets above 1E-10 solar masses
# block 1: mass above 1E-9 solar masses; 4 of the 5 IAU recognized
dwarfs_09 = ['Pluto Barycenter', 'Eris', 'Makemake', 'Haumea']
# block 2: mass above 1E-10 solar masses
dwarfs_10 = [
    '2007 OR10', 'Quaoar', 'Hygiea', 'Ceres', 'Orcus', 
    'Salacia', 'Varuna', 'Varda', 'Vesta', 'Pallas']
# block 3: mass above 1E-11 solar masses
dwarfs_11 = [
    '229762', '2002 UX25', '2002 WC19', 'Davida', 'Interamnia', 
    'Eunomia', '2004 UX10', 'Juno', 'Psyche', '52 Europa']

# Selected dwarfs
dwarfs = dwarfs_09 + dwarfs_10
# Object collection for dwarfs integration: planet barycenters + selected dwarfs
object_names_dwarfs = object_names_planets + dwarfs
# Test objects for dwarfs integration - same as for planets
test_objects_dwarfs = test_objects_planets

# Objects in collection 'all'
object_names_all = object_names_moons + dwarfs

# Shared collection of test asteroids to integrate
test_asteroids = [
    'Ceres', 'Pallas', 'Juno', 'Vesta', 'Iris',
    'Hygiea', 'Egeria', 'Eunomia', 'Psyche', 'Fortuna']
test_objects_asteroids = ['Earth'] + test_asteroids

# ********************************************************************************************************************* 
def make_sim(sim_name: str, object_names: List[str], epoch: datetime, 
             integrator, steps_per_day: int, save_file: bool):
    """Create or load simulation with the specified objects at the specified time"""
    # Filename for archive
    file_date = epoch.strftime('%Y-%m-%d_%H-%M')
    fname_archive: str = f'../data/planets/{sim_name}_{file_date}.bin'

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
            extend_sim_horizons(sim, object_names = objects_missing, epoch=epoch)

        # Sets of named and input object hashes
        hashes_sim = set(p.hash.value for p in sim.particles)
        hashes_input = set(rebound.hash(nm).value for nm in object_names)

        # Filter the simulation so only the named objects are included
        hashes_remove = [h for h in hashes_sim if h not in hashes_input]        
        for h in hashes_remove:
            sim.remove(hash=h)

    except:           
        # Initialize simulation
        sim = make_sim_horizons(object_names=object_names, epoch=epoch)

    # Move to center of momentum
    sim.move_to_com()

    # Set integrator and time step
    sim.integrator = integrator
    dt = 1.0 / steps_per_day if steps_per_day > 0 else 0
    sim.dt = dt
    if integrator == 'ias15':
        ias15 = sim.ri_ias15
        ias15.min_dt = dt

    # Save a snapshot to the archive file if requested
    if save_file:
        sim.simulationarchive_snapshot(filename=fname_archive, deletefile=True)

    # Return the simulation
    return sim

# ********************************************************************************************************************* 
def add_test_asteroids(object_names: List[str], test_asteroids: List[str]) -> List[str]:
    """Augment a set of objects to integrate to include the test asteroids"""
    object_names_new = object_names.copy()
    for nm in test_asteroids:
        if nm not in object_names:
            object_names_new.append(nm)
    return object_names_new

def set_test_asteroid_masses(sim: rebound.Simulation, 
                             object_names_orig: List[str], 
                             test_asteroids: List[str]) -> None:
    """Zero out the masses of test asteroids to zero if they were not originally added as massive objects"""
    # List of objects to zero
    names_zero = [nm for nm in test_asteroids if nm not in object_names_orig]
    # Set masses of these objects to zero
    for nm in names_zero:
        try:
            sim.particles[nm].m = 0.0
        except rebound.ParticleNotFound:
            print(f'Particle {nm} not in simulation')
            print(f'object_names_orig:')
            print(object_names_orig)
            print(f'test_asteroids:')
            print(test_asteroids)
            exit()

# ********************************************************************************************************************* 
def make_sim_planets(epoch: datetime, integrator='ias15', steps_per_day: int = 256):
    """Create a simulation with the sun and 8 planets at the specified time"""
    # Arguments for make_sim
    sim_name = 'planets'
    object_names = add_test_asteroids(object_names_planets, test_asteroids)
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    # Zero out the test asteroid masses
    set_test_asteroid_masses(sim=sim, object_names_orig=object_names_planets, test_asteroids=test_asteroids)

    return sim

# ********************************************************************************************************************* 
def make_sim_moons(epoch: datetime, integrator='ias15', steps_per_day: int = 16):
    """Create a simulation with the sun and 8 planets plus selected moons"""
    # Arguments for make_sim
    sim_name = 'moons'
    object_names = add_test_asteroids(object_names_moons, test_asteroids)
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    # Zero out the test asteroid masses
    set_test_asteroid_masses(sim=sim, object_names_orig=object_names_moons, test_asteroids=test_asteroids)
    
    return sim

# ********************************************************************************************************************* 
def make_sim_dwarfs(epoch: datetime, integrator='ias15', steps_per_day: int = 16):
    """Create a simulation with the sun, 8 planets and selected dwarf planets"""
    # Arguments for make_sim
    sim_name = 'dwarfs'
    object_names = add_test_asteroids(object_names_dwarfs, test_asteroids)
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    # Zero out the test asteroid masses
    set_test_asteroid_masses(sim=sim, object_names_orig=object_names_dwarfs, test_asteroids=test_asteroids)
    
    return sim

# ********************************************************************************************************************* 
def make_sim_all(epoch: datetime, integrator='ias15', steps_per_day: int = 16):
    """Create a simulation with all the massive objects (planets, moons, dwarf planets)"""
    # Arguments for make_sim
    sim_name = 'all'
    object_names = add_test_asteroids(object_names_all, test_asteroids)
    save_file = False

    # Build a simulation with the selected objects
    sim = make_sim(sim_name=sim_name, object_names=object_names, epoch=epoch, 
                   integrator=integrator, steps_per_day=steps_per_day, save_file=save_file)

    # Zero out the test asteroid masses
    set_test_asteroid_masses(sim=sim, object_names_orig=object_names_all, test_asteroids=test_asteroids)
    
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
        print(f'Generating archive {fname_archive} from {dt0} to {dt1}, time_step={time_step}...')
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
    earth_idx = object_names.index('Earth')
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
    pos_err = np.linalg.norm(pos_diff, axis=1)
    pos_err_den = np.linalg.norm(cfg0[:, 0:3], axis=1)
    pos_err_den[0] = 1.0
    pos_err_rel = pos_err / pos_err_den
    vel_err = np.linalg.norm(vel_diff, axis=1)
    vel_err_den = np.linalg.norm(cfg0[:, 3:6], axis=1)
    vel_err_den[0] = 1.0
    vel_err_rel = vel_err / vel_err_den

    if verbose:
        print(f'\nPosition difference - absolute & relative')
        print(f'Body       : ASC     : DEC     : AU      : Rel     : Vel Rel')
        object_names_short = [nm.replace(' Barycenter', '') for nm in object_names]
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
    # Angle differences are mod two pi
    two_pi = 2.0 * np.pi
    elt_diff[:,2:] = (elt_diff[:,2:] +np.pi ) % two_pi - np.pi
    
    # Compute RMS difference by orbital element
    elt_rms = rms(elt_diff, axis=0)
    elt_err = np.abs(elt_diff)

    # Names of selected elements
    elt_names = ['a', 'e', 'inc', 'Omega', 'omega', 'f', 'M', 'pomega', 'long']

    # Report RMS orbital element differences
    if verbose:
        print(f'\nOrbital element errors:')
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
    
    # Return the angle errors and RMS position error
    return pos_err, ang_err
    
# ********************************************************************************************************************* 
def test_integration(sa: rebound.SimulationArchive, test_objects: List[str], 
                     sim_name: str, test_name: str, 
                     verbose: bool = False,
                     make_plot: bool = False):
    """Test the integration of the planets against Horizons data"""
    # Start time of simulation
    dt0: datetime = datetime(2000, 1, 1)
    
    # Dates to be tested
    test_years = list(range(2000, 2041))
    test_dates = [datetime(year, 1, 1) for year in test_years]
    verbose_dates = [test_dates[-1]]
    
    # Errors on these dates
    ang_errs = []
    pos_errs = []
    
    # Test the dates
    for dt_t in test_dates:
        t = (dt_t - dt0).days
        sim_t = make_sim_horizons(object_names=test_objects, epoch=dt_t)
        report_this_date = (dt_t in verbose_dates) and verbose
        if report_this_date:
            print(f'\nDifference on {dt_t}:')
        pos_err, ang_err = report_sim_difference(sim0=sim_t, sim1=sa[t], 
                                                 object_names=test_objects, verbose=report_this_date)
        pos_errs.append(pos_err)
        ang_errs.append(ang_err)
    
    # Plot error summary
    pos_err_rms = np.array([rms(x) for x in pos_errs])
    ang_err_rms = np.array([rms(x) for x in ang_errs])
        
    if make_plot:
        # Chart titles
        sim_name_chart = sim_name.title()
        test_name_chart = test_name.title() if test_name != 'planets_com' else 'Planets (COM)'

        # Error in the position
        fig, ax = plt.subplots(figsize=[10,8])
        ax.set_title(f'Position Error of {test_name} in {sim_name_chart} Integration')
        ax.set_ylabel(f'RMS Position Error in AU')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0,))
        ax.plot(test_years, pos_err_rms, marker='o', color='red')
        ax.grid()
        fname = f'../figs/planets_integration/sim_error_{sim_name}_{test_name}_pos.png'
        fig.savefig(fname=fname, bbox_inches='tight')

        # Error in the angle
        fig, ax = plt.subplots(figsize=[10,8])
        ax.set_title(f'Angle Error of {test_name_chart} in {sim_name_chart} Integration')
        ax.set_ylabel(f'RMS Angle Error vs. Earth in Arcseconds')
        ax.plot(test_years, ang_err_rms, marker='o', color='blue')
        ax.grid()
        fname = f'../figs/planets_integration/sim_error_{sim_name}_{test_name}_angle.png'
        fig.savefig(fname=fname, bbox_inches='tight')
    
    if verbose:
        print(f'\nError by Date:')
        print('DATE       : ANG   : AU  ')
        for i, dt_t in enumerate(test_dates):
            print(f'{dt_t.date()} : {ang_err_rms[i]:5.3f} : {pos_err_rms[i]:5.3e}')
    
    # Compute average error
    mean_ang_err = np.mean(ang_err_rms)
    mean_pos_err = np.mean(pos_err_rms)
    print(f'\nMean RMS error in {sim_name} integration of {test_name} test objects:')
    print(f'AU   : {mean_pos_err:5.3e}')
    print(f'angle: {mean_ang_err:5.3f}')
    
    # Return summary of errors in position and angles
    return pos_err_rms, ang_err_rms
       
# ********************************************************************************************************************* 
def run_one_sim(sim_name: str, sim_epoch: rebound.Simulation, 
                epoch_dt: datetime, dt0: datetime, dt1: datetime, 
                time_step: int, save_step: int, steps_per_day: int):
    """Run one simulation, saving it to a simulation archive"""
    integrator = sim_epoch.integrator
    fname = f'../data/planets/sim_{sim_name}_2000-2040_{integrator}_sf{steps_per_day}.bin'
    fname_gen = f'../data/planets/sim_{sim_name}_2000-2040.bin'
    sa = make_archive(fname_archive=fname, sim_epoch=sim_epoch,
                      epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                      time_step=time_step, save_step=save_step)
    # Copy file to generically named one
    shutil.copy(fname, fname_gen)
    
    return sa

# ********************************************************************************************************************* 
def main():
    """Integrate the orbits of the planets and major moons"""
    
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Integrate the orbits of planets and moons.')
    parser.add_argument('type', nargs='?', metavar='T', type=str, default='A',
                        help='type of integration: p- planets; d- dwarfs; m-moons; a-all, '
                             'A-all 4 strategies: n-none')
    args = parser.parse_args()
    
    # Unpack command line arguments
    integration_type = args.type
    
    # Flags for planets and moons
    run_planets: bool = integration_type in ('p', 'A')
    run_moons: bool = integration_type in ('m', 'A')
    run_dwarfs: bool = integration_type in ('d', 'A')
    run_all: bool = integration_type in ('a', 'A')

    # Reference epoch for asteroids file
    epoch_mjd: float = 58600.0
    # Convert to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    # epoch_dt: datetime = datetime(2019,4,27)
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040,12,31)
    
    # Integrator choices
    integrator_planets: str = 'ias15'
    integrator_moons: str = 'ias15'
    integrator_dwarfs: str = 'ias15'
    integrator_all: str = 'ias15'
    
    # Integrator time step
    steps_per_day_planets: int = 16
    steps_per_day_moons: int = 16
    steps_per_day_dwarfs: int = 16
    steps_per_day_all: int = 16

    # Initial configuration of planets, moons, and dwarfs
    sim_planets = make_sim_planets(epoch=epoch_dt, integrator=integrator_planets, 
                                   steps_per_day=steps_per_day_planets)
    sim_moons = make_sim_moons(epoch=epoch_dt, integrator=integrator_moons, 
                               steps_per_day=steps_per_day_moons)
    sim_dwarfs = make_sim_dwarfs(epoch=epoch_dt, integrator=integrator_dwarfs, 
                                 steps_per_day=steps_per_day_dwarfs)
    sim_all = make_sim_all(epoch=epoch_dt, integrator=integrator_all, 
                           steps_per_day=steps_per_day_all)

    # Shared time_step and save_step
    time_step: int = 1
    save_step: int = 1

    # Integrate the planets from dt0 to dt1
    if run_planets:
        sim_name = 'planets'
        fname = f'../data/planets/sim_{sim_name}_2000-2040_{integrator_planets}_sf{steps_per_day_planets}.bin'
        fname_gen = f'../data/planets/sim_{sim_name}_2000-2040.bin'
        sa_planets = make_archive(fname_archive=fname, sim_epoch=sim_planets,
                                  epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                                  time_step=time_step, save_step=save_step)
        # Copy file to generically named one
        shutil.copy(fname, fname_gen)
    
    # Integrate the planets and moons from dt0 to dt1
    if run_moons:
        sim_name = 'moons'
        fname = f'../data/planets/sim_{sim_name}_2000-2040_{integrator_planets}_sf{steps_per_day_planets}.bin'
        fname_gen = f'../data/planets/sim_{sim_name}_2000-2040.bin'
        sa_moons = make_archive(fname_archive=fname, sim_epoch=sim_moons,
                                epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                                time_step=time_step, save_step=save_step)
        # Copy file to generically named one
        shutil.copy(fname, fname_gen)
       
    # Integrate the planets and dwarf planets from dt0 to dt1
    if run_dwarfs:
        sim_name = 'dwarfs'
        fname = f'../data/planets/sim_{sim_name}_2000-2040_{integrator_planets}_sf{steps_per_day_planets}.bin'
        fname_gen = f'../data/planets/sim_{sim_name}_2000-2040.bin'
        sa_dwarfs = make_archive(fname_archive=fname, sim_epoch=sim_dwarfs,
                                 epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                                 time_step=time_step, save_step=save_step)
        # Copy file to generically named one
        shutil.copy(fname, fname_gen)
       
    # Integrate the all bodies from dt0 to dt1
    if run_all:
        sim_name = 'all'
        fname = f'../data/planets/sim_{sim_name}_2000-2040_{integrator_planets}_sf{steps_per_day_planets}.bin'
        fname_gen = f'../data/planets/sim_{sim_name}_2000-2040.bin'
        sa_all = make_archive(fname_archive=fname, sim_epoch=sim_dwarfs,
                              epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, 
                              time_step=time_step, save_step=save_step)
        # Copy file to generically named one
        shutil.copy(fname, fname_gen)
       
    # Test integration of planets
    if run_planets:
        print(f'\n***** Testing integration of sun and 8 planets. *****')
        print(f'Integrator = {integrator_planets}, steps per day = {steps_per_day_planets}.')
        test_integration(sa=sa_planets, test_objects=test_objects_planets, 
                         sim_name='planets', test_name='planets_com', make_plot=False)
        pos_err_planets, ang_err_planets = \
        test_integration(sa=sa_planets, test_objects=test_objects_asteroids, 
                         sim_name='planets', test_name='asteroids', make_plot=True, verbose=True)
    
    # Test integration of planets and moons
    if run_moons:
        num_obj = sim_moons.N
        print(f'\n***** Testing integration of {num_obj} objects in solar system: sun, planets, moons. *****')
        print(f'Integrator = {integrator_moons}, steps per day = {steps_per_day_moons}.')
        test_integration(sa=sa_moons, test_objects=test_objects_moons, 
                         sim_name='moons', test_name='planets', make_plot=False)
        pos_err_moons, ang_err_moons = \
        test_integration(sa=sa_moons, test_objects=test_objects_asteroids, 
                         sim_name='moons', test_name='asteroids', make_plot=False)
        
    # Test integration of planets and dwarfs
    if run_dwarfs:
        num_obj = sim_dwarfs.N
        print(f'\n***** Testing integration of {num_obj} objects in solar system: sun, planets, dwarf planets. *****')
        print(f'Integrator = {integrator_dwarfs}, steps per day = {steps_per_day_dwarfs}.')
        test_integration(sa=sa_dwarfs, test_objects=test_objects_dwarfs, 
                         sim_name='dwarfs', test_name='planets', make_plot=False)
        pos_err_dwarfs, ang_err_dwarfs = \
        test_integration(sa=sa_dwarfs, test_objects=test_objects_asteroids, 
                         sim_name='dwarfs', test_name='asteroids', make_plot=False)
        
    # Plot position error
    fig, ax = plt.subplots(figsize=[10,8])
    ax.set_title(f'Position Error on 10 Test Asteroids')
    ax.set_ylabel('RMS Position Error ')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0,))
    test_years = list(range(2000,2041))
    if run_planets:
        ax.plot(test_years, pos_err_planets, label='planets', marker='+', color='blue')
    if run_moons:
        ax.plot(test_years, pos_err_moons, label='moons', marker='x', color='green')
    if run_dwarfs:
        ax.plot(test_years, pos_err_dwarfs, label='dwarfs', marker='o', color='red')
    ax.grid()
    ax.legend()
    fig.savefig(fname=f'../figs/planets_integration/sim_error_comp.png', bbox_inches='tight')

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
#    epoch_dt: datetime = datetime(2019,4,27)
#    sim_planets = make_sim_planets(epoch=epoch_dt, integrator='ias15', steps_per_day=16)
#    sim = sim_planets