"""
Harvard IACS Masters Thesis
Utilities for working with Rebound simulations and archives

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import rebound
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from itertools import chain
from tqdm import tqdm as tqdm_console
from typing import List

# Local imports
from astro_utils import datetime_to_mjd, cart_to_sph
from horizons import make_sim_horizons, extend_sim_horizons
from utils import rms

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
def make_archive_impl(fname_archive: str, 
                      sim_epoch: rebound.Simulation, 
                      object_names: List[str],
                      epoch: datetime, dt0: datetime, dt1: datetime, 
                      time_step: int, save_step: int,
                      progbar: bool):
    """
    Create a rebound simulation archive and save it to disk.
    INPUTS:
        fname_archive: the file name to save the archive to
        sim_epoch: rebound simulation object as of the epoch time; to be integrated in both directions
        object_names: the user names of all the objects in the simulation
        epoch: a datetime corresponding to sim_epoch
        dt0: the earliest datetime to simulate back to
        dt1: the latest datetime to simulate forward to
        time_step: the time step in days for the simulation
        save_step: the interval for saving snapshots to the simulation archive
        progbar: flag - whether to display a progress bar
    """
    
    # Convert epoch, start and end times relative to a base date of the simulation start
    # This way, time is indexed from t0=0 to t1 = (dt1-dt0)
    epoch_t = datetime_to_mjd(epoch, dt0)
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
    ts_back = ts[:idx][::-1]
    # The epochs corresponding to the times in ts
    epochs = [dt0 + timedelta(t) for t in ts]

    # File names for forward and backward integrations
    fname_fwd = fname_archive.replace('.bin', '_fwd.bin')
    fname_back = fname_archive.replace('.bin', '_back.bin')

    # Number of snapshots
    M_back = len(ts_back)
    M_fwd = len(ts_fwd)
    M = M_back + M_fwd
    # Number of particles
    N = sim_epoch.N

    # Initialize arrays for the position and velocity
    shape = (M, N, 3)
    q = np.zeros(shape, dtype=np.float64)
    v = np.zeros(shape, dtype=np.float64)

    # Integrate the simulation forward in time
    idx_fwd= list(enumerate(ts_fwd))
    if progbar:
        idx_fwd = tqdm_console(idx_fwd)
    for i, t in idx_fwd:
        # Integrate to the current time step with an exact finish time
        sim_fwd.integrate(t, exact_finish_time=1)
        # Row index for position data
        row = M_back + i
        # Serialize the position and velocity
        sim_fwd.serialize_particle_data(xyz=q[row])
        sim_fwd.serialize_particle_data(vxvyvz=v[row])
        # Save a snapshot on multiples of save_step
        if (i % save_step == 0) or (row == M-1):
            sim_fwd.simulationarchive_snapshot(filename=fname_fwd)

    # Integrate the simulation backward in time
    idx_back = list(enumerate(ts_back))
    if progbar:
        idx_back = tqdm_console(idx_back)
    for i, t in idx_back:
        # Integrate to the current time step with an exact finish time
        sim_back.integrate(t, exact_finish_time=1)
        # Row index for position data
        row = M_back - i
        # Serialize the position and velocity
        sim_back.serialize_particle_data(xyz=q[row])
        sim_back.serialize_particle_data(vxvyvz=v[row])
        # Save a snapshot on multiples of save_step
        if (i % save_step == 0) or (row == 0):
            sim_back.simulationarchive_snapshot(filename=fname_back)

    # Load the archives with the forward and backward snapshots
    sa_fwd = rebound.SimulationArchive(fname_fwd)
    sa_back = rebound.SimulationArchive(fname_back)
    
    # Filename for numpy arrays of position and velocity
    fname_np = fname_archive.replace('.bin', '.npz')

    # Save the epochs as a numpy array
    epochs_np = np.array(epochs)
    # Save the object names as a numpy array of strings
    object_names_np = np.array(object_names)

    # Save the object name hashes
    hashes = np.zeros(N, dtype=np.uint32)
    sim_epoch.serialize_particle_data(hash=hashes)

    # Combine the forward and backward archives in forward order from t0 to t1
    sims = chain(reversed(sa_back), sa_fwd)
    # Process each simulation snapshot in turn
    for i, sim in enumerate(sims):
        # Serialize the position and velocity
        sim.serialize_particle_data(xyz=q[i])
        sim.serialize_particle_data(vxvyvz=v[i])
        # Save a snapshot on multiples of save_step
        sim.simulationarchive_snapshot(fname_archive)        

    # Save the numpy arrays with the object hashes, position and velocity
    np.savez(fname_np, 
             q=q, v=v, 
             ts=ts, epochs_np=epochs_np,
             hashes=hashes, object_names_np=object_names_np)
    
    # Delete the forward and backward archives
    os.remove(fname_fwd)
    os.remove(fname_back)
    
# ********************************************************************************************************************* 
def make_archive(fname_archive: str, 
                 sim_epoch: rebound.Simulation, 
                 object_names: List[str],
                 epoch: datetime, dt0: datetime, dt1: datetime, 
                 time_step: int, save_step: int = 1,
                 progbar: bool = False):
    """
    Load a rebound archive if available; otherwise generate it and save it to disk.
    INPUTS:
        fname_archive: the file name to save the archive to
        sim_epoch: rebound simulation object as of the epoch time; to be integrated in both directions
        object_names: the user names of all the objects in the simulation
        epoch: a datetime corresponding to sim_epoch
        dt0: the earliest datetime to simulate back to
        dt1: the latest datetime to simulate forward to
        time_step: the time step in days for the simulation
        save_step: the interval for saving snapshots to the simulation archive
        progbar: flag - whether to display a progress bar
    """
    try:
        # First try to load the named archive
        sa = rebound.SimulationArchive(filename=fname_archive)
    except:
        # If the archive is not on disk, save it to disk
        print(f'Generating archive {fname_archive}\n'
              f'from {dt0} to {dt1}, time_step={time_step}, save_step={save_step}...')
        make_archive_impl(fname_archive=fname_archive, sim_epoch=sim_epoch, object_names=object_names,
                          epoch=epoch, dt0=dt0, dt1=dt1, time_step=time_step, 
                          save_step=save_step, progbar=progbar)
        # Load the new archive into memory
        sa = rebound.SimulationArchive(filename=fname_archive)
    return sa
    
# ********************************************************************************************************************* 
def load_sim_np(fname_np: str):
    """Load numpy arrays for position, velocity, and hashes from the named file"""
    # Load the numpy data file
    with np.load(fname_np, allow_pickle=True) as npz:
        # Extract position, velocity and hashes
        q = npz['q']
        v = npz['v']
        ts = npz['ts']
        epochs_np = npz['epochs_np']
        epochs: List[datetime.datetime] = [nm for nm in epochs_np]
        hashes = npz['hashes']
        object_names_np = npz['object_names_np']
        object_names: List[str] = [nm for nm in object_names_np]
    return q, v, ts, epochs, hashes, object_names

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
def sim_elt_array(sim, object_names=None):
    """Extract the orbital elements of each body in the simulation"""
    # Allocate array of elements
    num_objects = sim.N-1 if object_names is None else len(object_names)
    elts = np.zeros(shape=(num_objects, 9))
    
    # Iterate through particles AFTER the primary
    ps = sim.particles
    primary = ps[0]
    if object_names is None:
        for i, p in enumerate(ps[1:]):
            orb = p.calculate_orbit(primary=primary)
            elts[i] = np.array([orb.a, orb.e, orb.inc, orb.Omega, orb.omega, 
                                orb.f, orb.M, orb.pomega, orb.l])
    else:
        for i, object_name in enumerate(object_names):
            # look up the particle with this name
            p = ps[object_name]
            orb = p.calculate_orbit(primary=primary)
            elts[i] = np.array([orb.a, orb.e, orb.inc, orb.Omega, orb.omega, 
                                orb.f, orb.M, orb.pomega, orb.l])
    return elts

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
        print(f'(Angle errors in arcseconds, position in AU)')
        print(f'Body       : Phi     : Theta   : Pos AU  : Pos Rel : Vel Rel')
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
        fig, ax = plt.subplots(figsize=[16,10])
        ax.set_title(f'Position Error of {test_name_chart} in {sim_name_chart} Integration')
        ax.set_ylabel(f'RMS Position Error in AU')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0,))
        ax.plot(test_years, pos_err_rms, marker='o', color='red')
        ax.grid()
        fname = f'../figs/planets_integration/sim_error_{sim_name}_{test_name}_pos.png'
        fig.savefig(fname=fname, bbox_inches='tight')

        # Error in the angle
        fig, ax = plt.subplots(figsize=[16,10])
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
       
