"""
Harvard IACS Masters Thesis
Numerically integrate trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import pandas as pd
import rebound
from datetime import datetime
from tqdm import tqdm as tqdm_console
import argparse
from typing import List, Tuple, Dict, Optional

# Local imports
from utils import print_header
from astro_utils import mjd_to_datetime
from rebound_utils import make_archive, load_sim_np
from rebound_utils import report_sim_difference, test_integration, sim_cfg_array, sim_elt_array
# from astro_utils import datetime_to_mjd
from horizons import make_sim_horizons
from planets import make_sim_planets, object_names_planets

# ********************************************************************************************************************* 
def load_data_impl() -> pd.DataFrame:
    """Load the asteroid data into a Pandas DataFrame"""
    # The source for this file is at https://ssd.jpl.nasa.gov/?sb_elem
    fname: str = '../jpl/orb_elements_asteroid.txt'

    # The field names in the JPL file and their column positions
    names: List[str] = ['Num', 'Name', 'Epoch', 'a', 'e', 'i', 'w', 'Node', 'M', 'H', 'G', 'Ref']
    colspec_tbl: Dict[str, Tuple[int, int]] = {
        'Num': (0,6), 
        'Name': (7, 25), 
        'Epoch': (25, 30), 
        'a': (31, 41), 
        'e': (42, 52), 
        'i': (54, 62), 
        'w': (63, 72),
        'Node': (73, 82),
        'M': (83, 94),
        'H': (95, 100),
        'G': (101, 105),
        'Ref': (106, 113),
    }
    
    # Other arguments for Pandas file import
    colspecs: List[Tuple[int, int]] = [colspec_tbl[nm] for nm in names]
    header: int = 0
    skiprows: List[int] = [1]
    dtype: Dict[str, int] = {
        'Num': int,
        'Name': str,
        'Epoch': float,
        'a': float,
        'e': float,
        'i': float,
        'w': float,
        'Node': float,
        'M': float,
        'H': float,
        'G': float,
        'Ref': str,
    }

    # Read the DataFrame
    df: pd.DataFrame = pd.read_fwf(fname, colspecs=colspecs, header=header, names=names, skiprows=skiprows, dtype=dtype)
    # Set the asteroid number field to be the index
    df.set_index(keys=['Num'], drop=False, inplace=True)
    return df

# ********************************************************************************************************************* 
def convert_data(df_in: pd.DataFrame, epoch_mjd: Optional[float]=None) -> pd.DataFrame:
    """Convert data from the JPL format to be friendly to rebound integrator and matching selected epoch"""
    # Apply the default value of epoch_mjd if it was not input
    if epoch_mjd is None:
        epoch_mjd = pd.Series.mode(df_in.Epoch)[0]

    # Create a mask with only the matching rows
    mask = (df_in.Epoch == epoch_mjd)
    
    # Initialize Dataframe with asteroid numbers
    df = pd.DataFrame(data=df_in.Num[mask])

    # Add fields one at a time
    df['Name'] = df_in.Name[mask]
    df['epoch_mjd'] = df_in.Epoch[mask]
    df['a'] = df_in.a[mask]
    df['e'] = df_in.e[mask]
    df['inc'] = np.radians(df_in.i[mask])
    df['Omega'] = np.radians(df_in.Node[mask])
    df['omega'] = np.radians(df_in.w[mask])
    df['M'] = np.radians(df_in.M[mask])
    df['H'] = df_in.H[mask]
    df['G'] = df_in.G[mask]
    df['Ref'] = df_in.Ref[mask]
    
    # Set the asteroid number field to be the index
    df.set_index(keys=['Num'], drop=False, inplace=True)

    # Return the newly assembled DataFrame
    return df

# ********************************************************************************************************************* 
def load_data() -> pd.DataFrame:
    """Load the asteroid data into a Pandas Dataframe"""
    # The name for the saved DataFrame
    fname: str = '../jpl/orb_elements_asteroid.h5'
    
    # Try to load from disk if available
    ast_elt: pd.DataFrame
    try:
        ast_elt = pd.read_hdf(fname, key='ast_elt')
    except:
        # Load data from JPL asteroids file
        df_in = load_data_impl()
        # Convert data to rebound format
        ast_elt = convert_data(df_in=df_in)
        # Add the calculated orbital elements
        ast_elt = ast_data_add_calc_elements(ast_elt)
        # Save it to h5
        ast_elt.to_hdf(fname, key='ast_elt', mode='w')
    
    return ast_elt

# ********************************************************************************************************************* 
def make_sim_asteroids(sim_base: rebound.Simulation, 
                       ast_elt: pd.DataFrame, 
                       n0: int, n1: int,
                       progbar: bool = False) -> Tuple[rebound.Simulation, List[str]]:
    """
    Create a simulation with the selected asteroids by their ID numbers.
    INPUTS:
    sim_base: the base simulation, with e.g. the sun, planets, and selected moons
    ast_elt: the DataFrame with asteroid orbital elements at the specified epoch
    n0: the first asteroid number to add, inclusive
    n1: the last asteroid number to add, exclusive
    """
    # Start with a copy of the base simulation
    sim = sim_base.copy()
    # Set the number of active particles to the base simulation
    # https://rebound.readthedocs.io/en/latest/ipython/Testparticles.html
    sim.N_active = sim_base.N

    # Add the specified asteroids one at a time
    mask = (n0 <= ast_elt.Num) & (ast_elt.Num < n1)
    nums = ast_elt.index[mask]
    iters = tqdm_console(nums) if progbar else nums
    for num in iters:
        # Unpack the orbital elements
        a = ast_elt.a[num]
        e = ast_elt.e[num]
        inc = ast_elt.inc[num]
        Omega = ast_elt.Omega[num]
        omega = ast_elt.omega[num]
        M = ast_elt.M[num]
        name = ast_elt.Name[num]
        # Set the primary to the sun (NOT the solar system barycenter!)
        primary = sim.particles['Sun']
        # Add the new asteroid
        sim.add(m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M, primary=primary)
        # Set the hash to the asteroid's name
        sim.particles[-1].hash = rebound.hash(name)

    # The corresponding list of asteroid names
    asteroid_names = [name for name in ast_elt.Name[nums]]

    # Return the new simulation including the asteroids
    return sim, asteroid_names

# ********************************************************************************************************************* 
def make_sim_from_elts(elts: Dict[str, np.array], epoch: float):
    """
    Create a simulation for planets and asteroids parameterized by their orbital elements
    INPUTS:
        elts: dictionary with keys 'a', 'e', etc. for 6 orbital elements.  values arrays of shape (N,)
        epoch: MJD as of which these orbital elements apply    
    """
    # Convert epoch to a datetime
    epoch_dt = mjd_to_datetime(epoch)
    # Base Rebound simulation of the planets and moons on this date
    sim = make_sim_planets(epoch=epoch_dt)
    # Set the number of active particles to the base simulation
    sim.N_active = sim.N

    # Unpack the orbital elements
    a = elts['a']
    e = elts['e']
    inc = elts['inc']
    Omega = elts['Omega']
    omega = elts['omega']
    f = elts['f']
    
    # Get the number of asteroids in the data set
    n: int = a.shape[0]
    
    # Add the specified asteroids one at a time
    for i in range(n):
        # Set the primary to the sun (NOT the solar system barycenter!)
        # Put this inside the loop b/c not guaranteed to remain constant as particles are added
        primary = sim.particles['Sun']
        # Add the new asteroid
        sim.add(m=0.0, a=a[i], e=e[i], inc=inc[i], Omega=Omega[i], omega=omega[i], f=f[i], primary=primary)
        # Set the hash to the asteroid's number in this batch
        sim.particles[-1].hash = rebound.hash(f'{i}')
        
    # Return the new simulation including the asteroids
    return sim

# ********************************************************************************************************************* 
def calc_ast_pos(elts: Dict[str, np.array], epoch: float, ts: np.array) -> np.array:
    """
    Calculate asteroid positions from the given elements on the fly
    INPUTS:
        elts: dictionary with keys 'a', 'e', etc. for 6 orbital elements.  values arrays of shape (N,)
        epoch: MJD as of which these orbital elements apply
        ts: array of MJDs as of which 
    Outputs:
        q_earth: position of earth at input times; shape (traj_size, 3,)
        q_ast: positions of asteroids at input times; shape (num_ast, traj_size, 3,)
    Note: this function has some overlap with make_archive_impl in rebount_utils.py
    It's different though because it's intended to generate arrays on the fly, not save them to disk.
    """
    # Build the simulation at the epoch
    sim_epoch = make_sim_from_elts(elts, epoch)
    
    # Create copies of the simulation to integrate forward and backward
    sim_fwd: rebound.Simulation = sim_epoch.copy()
    sim_back: rebound.Simulation = sim_epoch.copy()

    # Set the time counter on both simulation copies to the epoch time
    sim_fwd.t = epoch
    sim_back.t = epoch

    # Set the times for snapshots in both directions;
    idx: int = np.searchsorted(ts, epoch, side='left')
    ts_fwd: np.array = ts[idx:]
    ts_back: np.array = ts[:idx][::-1]

    # Number of snapshots
    M_back: int = len(ts_back)
    M_fwd: int = len(ts_fwd)
    M: int = M_back + M_fwd
    # Number of particles
    N = sim_epoch.N
    # Number of asteroids (smaller than total number of particles; not including sun and planets)
    N_ast: int = elts['a'].shape[0]
    # Number of heavy bodies; saved before first asteroid in output array
    N_heavy: int = N - N_ast

    # The sun is the primary and is always in column 0
    sun_idx = 0
    # Column index of Earth by searching for hash of earth in all hashes of the simulation
    hash_earth = rebound.hash('Earth')
    hashes = np.zeros(N, dtype=np.uint32)
    sim_epoch.serialize_particle_data(hash=hashes)
    earth_idx = np.argmax(hashes==hash_earth)
    
    # Initialize array for the positions of all particles in the simulation
    # Match the order required by rebound for efficient serialization!
    # This must be permuted at the end
    q: np.array = np.zeros(shape=(M, N, 3,), dtype=np.float64)
    
    # Subfunction: process one row of the loop    
    def process_row(sim: rebound.Simulation, t: float, row: int):
        # Integrate to the current time step with an exact finish time
        sim.integrate(t, exact_finish_time=1)
        
        # Serialize the positions of earth and all the bodies
        sim.serialize_particle_data(xyz=q[row])
    
    # process times in the forward direction
    for i, t in enumerate(ts_fwd):
        # Row index for position data
        row: int = M_back + i
        # Process this row
        process_row(sim=sim_fwd, t=t, row=row)
    
    # process times in the backward direction
    for i, t in enumerate(ts_back):
        # Row index for position data
        row: int = M_back - (i+1)
        # Process this row
        process_row(sim=sim_back, t=t, row=row)
    
    # Position of sun is the relevant column
    q_sun = q[:, sun_idx, 0:3]
    # Position of earth is the relevant column
    q_earth = q[:, earth_idx, 0:3]
    # The asteroid positions are in the right-hand slice of q_all
    # Transpose so resulting array has size (N_ast, traj_size, 3)
    q_ast = q[:, N_heavy:N, 0:3].transpose((1,0,2))

    return q_ast, q_sun, q_earth

# ********************************************************************************************************************* 
def make_sim_asteroids_horizons(asteroid_names: List[str], epoch: datetime) -> rebound.Simulation:
    """Create or load a simulation with the planets and the named asteroids"""

    # The objects: Sun, Earth and requested asteroids
    object_names: List[str] = ['Sun', 'Earth'] + asteroid_names

    # Build a simulation from Horizons data
    sim: rebound.Simulation = make_sim_horizons(object_names=object_names, epoch=epoch)

    return sim

# ********************************************************************************************************************* 
def ast_data_add_calc_elements(ast_elt) -> pd.DataFrame:
    """Add the true anomaly and other calculated orbital elements to the asteroid DataFrame"""
    
    # Number of asteroids
    N: int = len(ast_elt)

    # Initialize empty arrays for computed orbital elements
    f = np.zeros(N)
    P = np.zeros(N)
    mean_motion = np.zeros(N)
    long = np.zeros(N)
    theta = np.zeros(N)
    pomega = np.zeros(N)
    T_peri = np.zeros(N)

    # Get the epoch from the DataFrame
    epoch_mjd: float = ast_elt.epoch_mjd[1]
    epoch: datetime = mjd_to_datetime(epoch_mjd)
    
    # Rebound simulation of the planets and moons on this date
    sim_base = make_sim_planets(epoch=epoch)
        
    # Make a gigantic simulation with all these asteroids
    n0: int = np.min(ast_elt.Num)
    n1 = np.max(ast_elt.Num) + 1
    print(f'Making big simulation with all {n1} asteroids...')
    sim_ast, asteroid_names = make_sim_asteroids(sim_base=sim_base, ast_elt=ast_elt, n0=n0, n1=n1, progbar=True)
    
    # Calculate orbital elements for all particles; must specify primary = Sun!!!
    print(f'Computing orbital elements...')
    orbits = sim_ast.calculate_orbits(primary=sim_ast.particles[0])
    # Slice orbits so it skips the planets and only includes the asteroids
    orbits = orbits[sim_base.N-1:]
    
    # Iterate over all the asteroids in the simulation
    print(f'Copying additional orbital elements to DataFrame...')
    mask = (n0 <= ast_elt.Num) & (ast_elt.Num < n1)
    iters = list(enumerate(ast_elt.Num[mask]))
    for i, num in tqdm_console(iters):
        # Look up the orbit of asteroid i
        orb = orbits[i]
        # Unpack the additional (calculated) orbital elements
        f[i] = orb.f
        P[i] = orb.P
        mean_motion[i] = orb.n
        long[i] = orb.l
        theta[i] = orb.theta
        pomega[i] = orb.pomega
        T_peri[i] = orb.T

    # Save computed orbital elements to the DataFrame
    ast_elt['f'] = f
    ast_elt['P'] = P
    ast_elt['n'] = mean_motion
    ast_elt['long'] = long
    ast_elt['theta'] = theta
    ast_elt['pomega'] = pomega
    ast_elt['T_peri'] = T_peri

    # Return the updated DataFrame 
    return ast_elt

# ********************************************************************************************************************* 
def test_element_recovery(verbose: bool = False) -> bool:
    """Test recovery of initial orbital elements for selected asteroids"""
    # List of asteroids to test: first 25
    asteroid_names = [
        'Ceres', 'Pallas', 'Juno', 'Vesta', 'Astraea', 
        'Hebe', 'Iris', 'Flora', 'Metis', 'Hygiea', 
        'Parthenope', 'Victoria', 'Egeria', 'Irene', 'Eunomia', 
        'Psyche', 'Thetis', 'Melpomene', 'Fortuna', 'Massalia',
        'Lutetia', 'Kalliope', 'Thalia', 'Phocaea']

    # Load asteroid data as DataFrame
    ast_elt = load_data()
    
    # Get the epoch from the DataFrame
    epoch_mjd: float = ast_elt.epoch_mjd[1]
    epoch: datetime = mjd_to_datetime(epoch_mjd)
    
    # Rebound simulation of the planets and moons on this date
    sim_base = make_sim_planets(epoch=epoch)
        
    # Add selected asteroids
    sim_ast, asteroid_names_out = make_sim_asteroids(sim_base=sim_base, ast_elt=ast_elt, n0=1, n1=31)

    # Create the reference simulation
    sim_hrzn = make_sim_asteroids_horizons(asteroid_names=asteroid_names, epoch=epoch)

    # Report the difference
    object_names = ['Earth'] + asteroid_names
    pos_err, ang_err = \
    report_sim_difference(sim0=sim_hrzn, sim1=sim_ast, object_names=object_names, verbose=True)
    
    # Report details of one specific asteroid
    report_one_asteroid(sim=sim_ast, asteroid_name='Ceres', epoch=epoch, verbose=True)

    # Threshold for pass
    pos_tol: float = 1.0E-5
    ang_tol: float = 2.0    

    # Test result
    isOK: bool = all(pos_err < pos_tol) and (ang_err < ang_tol)
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'\n***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
def report_one_asteroid(sim: rebound.Simulation, asteroid_name: str, 
                        epoch: datetime, verbose: bool = False) -> Tuple[np.array, np.array]:
    """Test whether orbital elements of the named asteroid are recovered vs. Horizons"""
    # Create the reference simulation
    sim_hrzn: rebound.Simulation = make_sim_asteroids_horizons(asteroid_names=[asteroid_name], epoch=epoch)
    
    # Alias the reference simulation to sim1, the input to sim2
    sim1: rebound.Simulation = sim_hrzn
    sim2: rebound.Simulation = sim

    # Orbit of asteroid in simulation 1
    primary1: rebound.Particle = sim1.particles['Sun']
    p1: rebound.Particle = sim1.particles[asteroid_name]
    orb1: rebound.Orbit = p1.calculate_orbit(primary=primary1)
    
    # Orbit of asteroid in simulation 2
    primary2: rebound.Particle = sim2.particles['Sun']
    p2: rebound.Particle = sim2.particles[asteroid_name]
    orb2: rebound.Orbit = p2.calculate_orbit(primary=primary2)

    # Compute errors in cartesian coordinates
    q1: np.array = np.array([p1.x, p1.y, p1.z]) - np.array([primary1.x, primary1.y, primary1.z])
    q2: np.array = np.array([p2.x, p2.y, p2.z]) - np.array([primary2.x, primary2.y, primary2.z])
    q: np.array = np.linalg.norm(q2 - q1)
    v1: np.array = np.array([p1.vx, p1.vy, p1.vz]) - np.array([primary1.vx, primary1.vy, primary1.vz])
    v2: np.array = np.array([p2.vx, p2.vy, p2.vz]) - np.array([primary2.vx, primary2.vy, primary2.vz])
    v: np.array = np.linalg.norm(v2 - v1)

    # Compute errors in orbital elements
    a: np.array = np.abs(orb2.a - orb1.a)
    e: np.array = np.abs(orb2.e - orb1.e)
    inc: np.array = np.abs(orb2.inc - orb1.inc)
    Omega: np.array = np.abs(orb2.Omega - orb1.Omega)
    omega: np.array = np.abs(orb2.omega - orb1.omega)
    f: np.array = np.abs(orb2.f - orb1.f)
    
    # Report errors if requested
    if verbose:
        print(f'\nErrors in recovered configuration and orbital elements for {asteroid_name}:')
        print(f'q    : {q:5.3e}')
        print(f'v    : {v:5.3e}')
        print(f'a    : {a:5.3e}')
        print(f'e    : {e:5.3e}')
        print(f'inc  : {inc:5.3e}')
        print(f'Omega: {Omega:5.3e}')
        print(f'omega: {omega:5.3e}')
        print(f'f    : {f:5.3e}')
    
    # Return the errors in q and v
    return q, v

# ********************************************************************************************************************* 
def test_asteroid_sim(make_plot: bool = False, verbose: bool=False) -> bool:
    """Test the integration of the asteroids against Horizons"""
    # Load the simulation archive for the first 1000 asteroids
    n0: int = 0
    n1: int = 1000
    fname: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.bin'
    sa: rebound.SimulationArchive = rebound.SimulationArchive(fname)
    
    # List of objects to test: Earth and the first 25 asteroids
    test_objects: List[str] = [
        'Sun', 'Earth',
        'Ceres', 'Pallas', 'Juno', 'Vesta', 'Astraea', 
        'Hebe', 'Iris', 'Flora', 'Metis', 'Hygiea', 
        'Parthenope', 'Victoria', 'Egeria', 'Irene', 'Eunomia', 
        'Psyche', 'Thetis', 'Melpomene', 'Fortuna', 'Massalia',
        'Lutetia', 'Kalliope', 'Thalia', 'Phocaea'] 
    
    # Other args to test_integration
    sim_name: str = 'planets'
    
    # Test against the asteroid test set
    pos_err, ang_err = \
        test_integration(sa=sa, test_objects=test_objects, 
                         sim_name=sim_name, test_name='asteroids', 
                         make_plot=make_plot, verbose=verbose)
        
    # Threshold for pass
    pos_tol: float = 1.0E-5
    ang_tol: float = 2.0    

    # Test result
    isOK: bool = (max(pos_err) < pos_tol) and (max(ang_err) < ang_tol)
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'\n***** {msg} *****')
    return isOK
        
# ********************************************************************************************************************* 
def test_numpy(verbose: bool = False) -> bool:
    """Test the numpy output against the simulation archive"""
    # Start time of simulation
    dt0: datetime = datetime(2000, 1, 1)
    
    # Load the simulation archive for the first 1000 asteroids
    n0: int = 0
    n1: int = 1000
    fname_sa: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.bin'
    sa: rebound.SimulationArchive = rebound.SimulationArchive(fname_sa)
    
    # Name of the numpy archive
    fname_np: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.npz'
    
    # The full array of positions and velocities
    q, v, elts, catalog = load_sim_np(fname_np=fname_np)
    # The object names
    object_names = catalog['object_names']
    
    # Dates to be tested
    test_years: List[int] = list(range(2000, 2041))
    test_dates: List[datetime] = [datetime(year, mth, 1) for year in test_years for mth in [1]]
    # Errors on these date in q, v and orbital elements
    N_test: int = len(test_dates)
    q_errs = np.zeros(N_test)
    v_errs = np.zeros(N_test)
    elt_errs = np.zeros(N_test)
    
    # Header row
    if verbose:
        print(f'DATE      : q_err     : v_err     : elt_err')
    # Test the numpy arrays vs. sim archive on these dates
    for i, dt_t in enumerate(test_dates):
        # The date to be tested as a time coordinate
        t = (dt_t - dt0).days
        # The test simulation from the simulation archive
        sim = sa.getSimulation(t=t, mode='exact')
        # The position and velocity from the simulation
        cfg_sim = sim_cfg_array(sim=sim, object_names=object_names)
        q_sim, v_sim = cfg_sim[:, 0:3], cfg_sim[:, 3:6]
        # The orbital elements from the simulation
        elts_sim = sim_elt_array(sim=sim, object_names=object_names[1:])
        # Save just the first six columns and transpose to match shape of numpy
        elts_sim = elts_sim[:, 0:6]
        
        # The position, velocity and orbital elements from the numpy arrays
        q_np = q[t]
        v_np = v[t]
        # Extract the orbital elements from the numpy
        # Skip the first row; the sun has no orbital elements
        elts_np = np.array([elts.a[t, 1:], elts.e[t, 1:], elts.inc[t, 1:],
                            elts.Omega[t, 1:], elts.omega[t, 1:], elts.f[t, 1:]]).transpose()

        # The difference; should be zero
        q_err = np.linalg.norm(q_np - q_sim)
        v_err = np.linalg.norm(v_np - v_sim)
        elt_err = np.linalg.norm(elts_np - elts_sim)

        # Status
        if verbose:
            print(f'{dt_t.date()}: {q_err:5.3e} : {v_err:5.3e} : {elt_err:5.3e}')
        # Save to list
        q_errs[i] = q_err
        v_errs[i] = v_err
        elt_errs[i] = elt_err
    
    # Maximum errors
    q_err_max = np.max(q_errs)
    v_err_max = np.max(v_errs)
    elt_err_max = np.max(elt_err)

    if verbose:
        print(f'MAX ERROR : {q_err_max:5.3e} : {v_err_max:5.3e} : {elt_err_max:5.3e}')
    else:
        print(f'Max errors:')
        print(f'q_err   = {q_err_max:5.3e}')
        print(f'v_err   = {v_err_max:5.3e}')
        print(f'elt_err = {elt_err_max:5.3e}')

    # Threshold for pass
    q_tol: float = 1.0E-9
    v_tol: float = 1.0E-9
    elt_tol: float = 1.0E-9

    # Test result
    isOK: bool = (q_err_max < q_tol) and (v_err_max < v_tol) and (elt_err_max < elt_tol)
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'\n***** {msg} *****')
    return isOK
        
# ********************************************************************************************************************* 
def main():
    """Main routine for integrating the orbits of known asteroids"""
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Integrate the orbits of known asteroids from JPL ephemeris file.')
    parser.add_argument('n0', nargs='?', metavar='n0', type=int, default=0,
                        help='the first asteroid number to process')
    parser.add_argument('n_ast', nargs='?', metavar='B', type=int, default=1000,
                        help='the number of asteroids to process in this batch')
    parser.add_argument('--progress', default=False, action='store_true',
                        help='display progress bar')
    parser.add_argument('--test', default=False, action='store_true',
                        help='run in test mode')
    args = parser.parse_args()
    
    # If run in test mode, run tests without processing any asteroid trajectories
    if args.test:
        # Test  that initial orbital elements recovered from the JPL file
        print_header(f'Testing recovery of initial orbital elements with JPL text file vs. Horizons')
        test_element_recovery(verbose=True)

        # Test the integration vs. Horizons
        print_header(f'Testing asteroid integration vs. Horizons')
        test_asteroid_sim(verbose=True, make_plot=True)
        
        # Test numpy arrays
        print_header(f'Testing Numpy array vs. simulation archive:')
        test_numpy(verbose=True)
        
        # Quit early in test mode: don't want to do any integrations
        print()
        exit()

    # Unpack command line arguments
    n0: int = args.n0
    n1: int = n0 + args.n_ast
    progbar: bool = args.progress

    # Load asteroid data as DataFrame
    ast_elt: pd.DataFrame = load_data()

    # Get the epoch from the DataFrame
    epoch_mjd: float = ast_elt.epoch_mjd[1]
    epoch: datetime = mjd_to_datetime(epoch_mjd)

    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040,12,31)
    
    # Rebound simulation of the planets on this date
    integrator: str = 'ias15'
    steps_per_day: int = 16
    sim_base: rebound.Simulation = make_sim_planets(epoch=epoch, integrator=integrator, steps_per_day=steps_per_day)
        
    # Add selected asteroids
    sim: rebound.Simulation
    asteroid_names: List[str]
    sim, asteroid_names = make_sim_asteroids(sim_base=sim_base, ast_elt=ast_elt, n0=n0, n1=n1)
    
    # The list of object names corresponding to this simulation
    object_names: List[str] = object_names_planets + asteroid_names

    # Integrate the asteroids from dt0 to dt1 with a time step of 1 day
    fname: str = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.bin'
    time_step: int = 1
    save_step: int = 32
    save_elements: bool = True
    print(f'Processing asteroid trajectories for asteroid numbers {n0} to {n1}...')
    make_archive(fname_archive=fname, sim_epoch=sim, object_names=object_names,
                 epoch=epoch, dt0=dt0, dt1=dt1, 
                 time_step=time_step, save_step=save_step, 
                 save_elements=save_elements, progbar=progbar)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
