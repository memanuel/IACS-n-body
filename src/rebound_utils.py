"""
Harvard IACS Masters Thesis
Utilities for working with Rebound simulations and archives

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import rebound
from datetime import datetime, timedelta
from collections import namedtuple
import os
from itertools import chain
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_console
from typing import List, Tuple, Dict, Set, Optional

# Local imports
from astro_utils import datetime_to_mjd, cart_to_sph
from horizons import make_sim_horizons, extend_sim_horizons
from utils import rms

# Named tuple data type for orbital elements
OrbitalElement = namedtuple('OrbitalElement', 'a e inc Omega omega f')

# ********************************************************************************************************************* 
def extend_sim(sim: rebound.Simulation, 
               object_names_new: List[str], 
               epoch: datetime):
    """Extend an existing simulation"""
    # Generate list of missing object names
    hashes_present: Set[int] = set(p.hash.value for p in sim.particles)
    objects_missing: List[str] = [nm for nm in object_names_new if rebound.hash(nm).value not in hashes_present]

    # Extend the simulation and save it with the augmented bodies
    if objects_missing:
        extend_sim_horizons(sim, object_names=objects_missing, epoch=epoch)

    return sim

# ********************************************************************************************************************* 
def make_sim(sim_name: str, object_names: List[str], epoch: datetime, 
             integrator: str, steps_per_day: int, save_file: bool) -> rebound.Simulation:
    """Create or load simulation with the specified objects at the specified time"""
    # Filename for archive
    file_date: str = epoch.strftime('%Y-%m-%d_%H-%M')
    fname_archive: str = f'../data/planets/{sim_name}_{file_date}.bin'

    # If this file already exists, load it and check for both extra and missing bodies
    sim: rebound.Simulation
    try:
        # Attempt to load the named file
        sim = rebound.Simulation(fname_archive)
        # print(f'Loaded {fname_archive}')

        # Generate list of missing object names
        objects_missing: List[str] = [nm for nm in object_names if nm not in sim.particles]

        # Extend the simulation and save it with the augmented bodies
        if objects_missing:
            print(f'Found missing objects in {fname_archive}:')
            print(objects_missing)
            extend_sim_horizons(sim, object_names = objects_missing, epoch=epoch)

        # Sets of named and input object hashes
        hashes_sim: Set[int] = set(p.hash.value for p in sim.particles)
        hashes_input: Set[str] = set(rebound.hash(nm).value for nm in object_names)

        # Filter the simulation so only the named objects are included
        hashes_remove: List[str] = [h for h in hashes_sim if h not in hashes_input]        
        for h in hashes_remove:
            sim.remove(hash=h)

    except:           
        # Initialize simulation
        sim = make_sim_horizons(object_names=object_names, epoch=epoch)

    # Move to center of momentum
    sim.move_to_com()

    # Set integrator and time step
    sim.integrator = integrator
    dt: float = 1.0 / steps_per_day if steps_per_day > 0 else 0
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
                      save_elements: bool,
                      progbar: bool) -> None:
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
        save_elements: flag indiciting whether to save orbital elements
        progbar: flag - whether to display a progress bar
    """
    
    # Convert epoch, start and end times relative to a base date of the simulation start
    # This way, time is indexed from t0=0 to t1 = (dt1-dt0)
    epoch_t: float = datetime_to_mjd(epoch, dt0)
    t0: float = datetime_to_mjd(dt0, dt0)
    t1: float = datetime_to_mjd(dt1, dt0)
    
    # Create copies of the simulation to integrate forward and backward
    sim_fwd: rebound.Simulation = sim_epoch.copy()
    sim_back: rebound.Simulation = sim_epoch.copy()

    # Set the time counter on both simulation copies to the epoch time
    sim_fwd.t = epoch_t
    sim_back.t = epoch_t

    # Set the times for snapshots in both directions;
    ts: np.array = np.arange(t0, t1+time_step, time_step)
    idx: int = np.searchsorted(ts, epoch_t, side='left')
    ts_fwd: np.array = ts[idx:]
    ts_back: np.array = ts[:idx][::-1]
    # The epochs corresponding to the times in ts
    epochs: List[datetime] = [dt0 + timedelta(t) for t in ts]

    # File names for forward and backward integrations
    fname_fwd: str = fname_archive.replace('.bin', '_fwd.bin')
    fname_back: str = fname_archive.replace('.bin', '_back.bin')

    # Number of snapshots
    M_back: int = len(ts_back)
    M_fwd: int = len(ts_fwd)
    M: int = M_back + M_fwd
    # Number of particles
    N: int = sim_epoch.N

    # Initialize arrays for the position and velocity
    shape_qv: Tuple[int] = (M, N, 3)
    q: np.array = np.zeros(shape_qv, dtype=np.float64)
    v: np.array = np.zeros(shape_qv, dtype=np.float64)
    
    # Initialize arrays for orbital elements if applicable
    
    if save_elements:
        # Arrays for a, e, inc, Omega, omega, f
        shape_elt: Tuple[int] = (M, N)
        orb_a: np.array = np.zeros(shape_elt, dtype=np.float64)
        orb_e: np.array = np.zeros(shape_elt, dtype=np.float64)
        orb_inc: np.array = np.zeros(shape_elt, dtype=np.float64)
        orb_Omega: np.array = np.zeros(shape_elt, dtype=np.float64)
        orb_omega: np.array = np.zeros(shape_elt, dtype=np.float64)
        orb_f: np.array = np.zeros(shape_elt, dtype=np.float64)

        # Wrap these into a named tuple
        elts: OrbitalElement = \
            OrbitalElement(a=orb_a, e=orb_e, inc=orb_inc,
                           Omega=orb_Omega, omega=orb_omega, f=orb_f)
    else:
        elts = OrbitalElement()

    # Subfunction: process one row of the loop    
    def process_row(sim: rebound.Simulation, fname: str, 
                    t: float, row: int):
        # Integrate to the current time step with an exact finish time
        sim.integrate(t, exact_finish_time=1)
        
        # Serialize the position and velocity
        sim.serialize_particle_data(xyz=q[row])
        sim.serialize_particle_data(vxvyvz=v[row])
        
        # Save a snapshot on multiples of save_step or the first / last row
        if (i % save_step == 0) or (row in (0, M-1)):
            sim.simulationarchive_snapshot(filename=fname)

        # Save the orbital elements if applicable
        if save_elements:
            # Compute the orbital elements vs. the sun as primary
            primary = sim.particles['Sun']
            orbits = sim.calculate_orbits(primary=primary, jacobi_masses=False)
            # Save the elements on this date as an Nx6 array
            # This approach only traverses the (slow) Python list orbits one time
            elt_array = np.array([[orb.a, orb.e, orb.inc, orb.Omega, orb.omega, orb.f] \
                                  for orb in orbits])
            # Save the elements into the current row of the named orbital elements arrays
            # The LHS row mask starts at 1 b/c orbital elements are not computed for the first object (Sun)
            orb_a[row, 1:] = elt_array[:, 0]
            orb_e[row, 1:] = elt_array[:, 1]
            orb_inc[row, 1:] = elt_array[:, 2]
            orb_Omega[row, 1:] = elt_array[:, 3]
            orb_omega[row, 1:] = elt_array[:, 4]
            orb_f[row, 1:] = elt_array[:, 5]

    # Integrate the simulation forward in time
    idx_fwd: List[Tuple[int, float]] = list(enumerate(ts_fwd))
    if progbar:
        idx_fwd = tqdm_console(idx_fwd)
    for i, t in idx_fwd:
        # Row index for position data
        row: int = M_back + i
        # Process this row
        process_row(sim=sim_fwd, fname=fname_fwd, t=t, row=row)
        
    # Integrate the simulation backward in time
    idx_back: List[Tuple[int, float]] = list(enumerate(ts_back))
    if progbar:
        idx_back = tqdm_console(idx_back)
    for i, t in idx_back:
        # Row index for position data
        row: int = M_back - (i+1)
        # Process this row
        process_row(sim=sim_back, fname=fname_back, t=t, row=row)

    # Load the archives with the forward and backward snapshots
    sa_fwd: rebound.SimulationArchive = rebound.SimulationArchive(fname_fwd)
    sa_back: rebound.SimulationArchive = rebound.SimulationArchive(fname_back)
    
    # Filename for numpy arrays of position and velocity
    fname_np: str = fname_archive.replace('.bin', '.npz')

    # Save the epochs as a numpy array
    epochs_np: np.array = np.array(epochs)
    # Save the object names as a numpy array of strings
    object_names_np: np.array = np.array(object_names)

    # Save the object name hashes
    hashes: np.array = np.zeros(N, dtype=np.uint32)
    sim_epoch.serialize_particle_data(hash=hashes)

    # Combine the forward and backward archives in forward order from t0 to t1
    sims = chain(reversed(sa_back), sa_fwd)
    # Process each simulation snapshot in turn
    for i, sim in enumerate(sims):
        # Save a snapshot on multiples of save_step
        sim.simulationarchive_snapshot(fname_archive)        

    # Save the numpy arrays with the object hashes, position and velocity
    np.savez(fname_np, 
             q=q, v=v, elts=elts,
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
                 save_elements: bool = False,
                 progbar: bool = False) -> rebound.SimulationArchive:
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
                          epoch=epoch, dt0=dt0, dt1=dt1, 
                          time_step=time_step, save_step=save_step, 
                          save_elements=save_elements, progbar=progbar)
        # Load the new archive into memory
        sa = rebound.SimulationArchive(filename=fname_archive)
    return sa
    
# ********************************************************************************************************************* 
def load_sim_np(fname_np: str) ->Tuple[np.array, np.array, Dict[str, np.array]]:
    """Load numpy arrays for position, velocity, and catalog data from the named file"""
    # Load the numpy data file
    with np.load(fname_np, allow_pickle=True) as npz:
        # Extract position, velocity and hashes
        q = npz['q']
        v = npz['v']
        elts = npz['elts']
        ts = npz['ts']
        epochs_np = npz['epochs_np']
        epochs: List[datetime.datetime] = [nm for nm in epochs_np]
        hashes = npz['hashes']
        object_names_np = npz['object_names_np']
        object_names: List[str] = [nm for nm in object_names_np]

    # Wrap the catalog into a dictionary
    catalog = {
        'ts': ts,
        'epochs': epochs,
        'hashes': hashes,
        'object_names': object_names
        }

    # For some reason, np.save() squishes a namedtuple into an ND array.  Restore it to a named tuple
    elts = OrbitalElement(a=elts[0], e=elts[1], inc=elts[2], 
                          Omega=elts[3], omega=elts[4], f=elts[5])

    # Return the position, velocity, and catalog        
    return q, v, elts, catalog

# ********************************************************************************************************************* 
def sim_cfg_array(sim: rebound.Simulation, object_names: Optional[List[str]]=None) -> np.array:
    """Extract the Cartesian configuration of each body in the simulation"""
    # Allocate array of configurations
    num_objects: int = sim.N if object_names is None else len(object_names)
    cfgs: np.array = np.zeros(shape=(num_objects, 6))

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
def sim_elt_array(sim: rebound.Simulation, object_names=None) -> np.array:
    """Extract the orbital elements of each body in the simulation"""
    # Allocate array of elements
    num_objects: int = sim.N-1 if object_names is None else len(object_names)
    elts: np.array = np.zeros(shape=(num_objects, 9))
    
    # Iterate through particles AFTER the primary
    ps: List[rebound.Particles] = sim.particles
    primary: rebound.Particle = ps[0]
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
def report_sim_difference(sim0: rebound.Simulation, sim1: rebound.Simulation, 
                          object_names: List[str], verbose: bool=False) -> \
                          Tuple[np.array, np.array]:
    """Report the difference between two simulations on a summary basis"""
    # Extract configuration arrays for the two simulations
    cfg0: np.array = sim_cfg_array(sim0, object_names)
    cfg1: np.array = sim_cfg_array(sim1, object_names)
    
    # Convert both arrays to heliocentric coordinates
    cfg0 = cfg0 - cfg0[0:1,:]
    cfg1 = cfg1 - cfg1[0:1,:]

    # Displacement of each body to earth
    earth_idx: int = object_names.index('Earth')
    q0: np.array = cfg0[:, 0:3] - cfg0[earth_idx, 0:3]
    q1: np.array = cfg1[:, 0:3] - cfg1[earth_idx, 0:3]
    
    # Right Ascension and Declination
    r0, asc0, dec0 = cart_to_sph(q0)
    r1, asc1, dec1 = cart_to_sph(q1)

    # Error in asc and dec; convert from radians to arcseconds
    asc_err: np.array = np.degrees(np.abs(asc1-asc0)) * 3600
    dec_err: np.array = np.degrees(np.abs(dec1-dec0)) * 3600

    # Take differences
    cfg_diff: np.array = (cfg1 - cfg0)
    pos_diff: np.array = cfg_diff[:, 0:3]
    vel_diff: np.array = cfg_diff[:, 3:6]

    # Error in position and velocity in heliocentric coordinates; skip the sun
    pos_err: np.array = np.linalg.norm(pos_diff, axis=1)
    pos_err_den: np.array = np.linalg.norm(cfg0[:, 0:3], axis=1)
    pos_err_den[0] = 1.0
    pos_err_rel: np.array = pos_err / pos_err_den
    vel_err: np.array = np.linalg.norm(vel_diff, axis=1)
    vel_err_den: np.array = np.linalg.norm(cfg0[:, 3:6], axis=1)
    vel_err_den[0] = 1.0
    vel_err_rel: np.array = vel_err / vel_err_den

    if verbose:
        print(f'\nPosition difference - absolute & relative')
        print(f'(Angle errors in arcseconds, position in AU)')
        print(f'Body       : Phi     : Theta   : Pos AU  : Pos Rel : Vel Rel')
        object_names_short: List[str] = [nm.replace(' Barycenter', '') for nm in object_names]
        for i, nm in enumerate(object_names_short):
            print(f'{nm:10} : {asc_err[i]:5.2e}: {dec_err[i]:5.2e}: {pos_err[i]:5.2e}: '
                  f'{pos_err_rel[i]:5.2e}: {vel_err_rel[i]:5.2e}')
        print(f'Overall    : {rms(asc_err):5.2e}: {rms(dec_err):5.2e}: {rms(pos_err):5.2e}: '
              f'{rms(pos_err_rel):5.2e}: {rms(vel_err_rel):5.2e}')

    # Extract orbital element arrays from the two simulations
    elt0: np.array = sim_elt_array(sim0, object_names[1:])
    elt1: np.array = sim_elt_array(sim1, object_names[1:])

    # Take differences
    elt_diff: np.array = (elt1 - elt0)
    # Angle differences are mod two pi
    two_pi: float = 2.0 * np.pi
    elt_diff[:,2:] = (elt_diff[:,2:] +np.pi ) % two_pi - np.pi
    
    # Compute RMS difference by orbital element
    elt_rms: np.array = rms(elt_diff, axis=0)
    elt_err: np.array = np.abs(elt_diff)

    # Names of selected elements
    elt_names: List[str] = ['a', 'e', 'inc', 'Omega', 'omega', 'f', 'M', 'pomega', 'long']

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
    ang_err: np.array = rms(np.array([asc_err, dec_err]))
    
    # Return the RMS position error and angle errors
    return pos_err, ang_err
    
# ********************************************************************************************************************* 
def test_integration(sa: rebound.SimulationArchive, test_objects: List[str], 
                     sim_name: str, test_name: str, 
                     verbose: bool = False,
                     make_plot: bool = False) -> \
                     Tuple[np.array, np.array]:
    """Test the integration of the planets against Horizons data"""
    # Start time of simulation
    dt0: datetime = datetime(2000, 1, 1)
    
    # Dates to be tested
    test_years: List[int] = list(range(2000, 2041))
    test_dates: List[datetime] = [datetime(year, 1, 1) for year in test_years]
    verbose_dates: List[datetime] = [test_dates[-1]]
    
    # Errors on these dates
    ang_errs: List[np.array] = []
    pos_errs: List[np.array] = []
    
    # Test the dates
    dt_t: datetime
    for dt_t in test_dates:
        # The date to be tested as a time coordinate
        t: int = (dt_t - dt0).days
        # The reference simulation from Horizons
        sim0: rebound.Simulation = make_sim_horizons(object_names=test_objects, epoch=dt_t)
        # The test simulation from the simulation archive
        sim1: rebound.Simulation = sa.getSimulation(t=t, mode='exact')
        # Verbosity flag and screen print if applicable
        report_this_date: bool = (dt_t in verbose_dates) and verbose
        if report_this_date:
            print(f'\nDifference on {dt_t}:')
        # Run the test
        pos_err, ang_err = report_sim_difference(sim0=sim0, sim1=sim1, 
                                                 object_names=test_objects, verbose=report_this_date)
        # Save position and angle errors
        pos_errs.append(pos_err)
        ang_errs.append(ang_err)
    
    # Plot error summary
    pos_err_rms: np.array = np.array([rms(x) for x in pos_errs])
    ang_err_rms: np.array = np.array([rms(x) for x in ang_errs])
        
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
        fname: str = f'../figs/integration_test/sim_error_{sim_name}_{test_name}_pos.png'
        fig.savefig(fname=fname, bbox_inches='tight')

        # Error in the angle
        fig, ax = plt.subplots(figsize=[16,10])
        ax.set_title(f'Angle Error of {test_name_chart} in {sim_name_chart} Integration')
        ax.set_ylabel(f'RMS Angle Error vs. Earth in Arcseconds')
        ax.plot(test_years, ang_err_rms, marker='o', color='blue')
        ax.grid()
        fname: str = f'../figs/integration_test/sim_error_{sim_name}_{test_name}_angle.png'
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
