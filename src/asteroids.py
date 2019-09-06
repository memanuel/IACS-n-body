"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import numpy as np
import pandas as pd
import rebound
from jplephem.spk import SPK
from datetime import datetime
# from tqdm import tqdm as tqdm_console
import argparse
from typing import List, Dict

# Local imports
from astro_utils import mjd_to_horizons, datetime_to_horizons, datetime_to_mjd, mjd_to_datetime
from planets import make_sim_horizons, make_sim_planets, make_sim_moons, make_archive
from utils import rms

# Load asteroid kernel
kernel = SPK.open('../jpl/asteroids.bsp')


# ********************************************************************************************************************* 
def load_data():
    """Load the asteroid data into a Pandas DataFrame"""
    # The source for this file is at https://ssd.jpl.nasa.gov/?sb_elem
    fname = '../jpl/orb_elements_asteroid.txt'

    # The field names in the JPL file and their column positions
    names = ['Num', 'Name', 'Epoch', 'a', 'e', 'i', 'w', 'Node', 'M', 'H', 'G', 'Ref']
    colspec_tbl = {'Num': (0,6), 
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
                   'Ref': (106, 113)}
    
    # Other arguments for Pandas file import
    colspecs = [colspec_tbl[nm] for nm in names]
    header = 0
    skiprows = [1]
    dtype = {
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
    df = pd.read_fwf(fname, colspecs=colspecs, header=header, names=names, skiprows=skiprows, dtype=dtype)
    # Set the asteroid number field to be the index
    df.set_index(keys=['Num'], drop=False, inplace=True)
    return df

# ********************************************************************************************************************* 
def convert_data(df_in: pd.DataFrame, epoch_mjd: float=None):
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
def sim_clean_names(sim: rebound.Simulation):
    """Change names of particles to exclude Geocenter suffix"""
    planet_systems = ['Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    for nm in planet_systems:
        old_name = f'{nm} Geocenter'
        sim.particles[old_name].hash = rebound.hash(nm)

# ********************************************************************************************************************* 
def sa_clean_names(sa: rebound.SimulationArchive):
    """Change names of particles to exclude Geocenter suffix"""
    for sim in sa:
        sim_clean_names(sim)

# ********************************************************************************************************************* 
def make_sim_asteroids(sim_base: rebound.Simulation, ast_elt: pd.DataFrame, n0: int, n1: int):
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
    for num in nums:
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

    # Return the new simulation including the asteroids
    return sim

# ********************************************************************************************************************* 
def make_sim_asteroids_horizons(fname: str, asteroid_names: List[str], epoch_mjd: int):
    """Create or load a simulation with the named asteroids"""
        
    # If this file already exists, load and return it
    try:
        sim = rebound.Simulation(fname)
        # print(f'Found simulation archive {fname_archive}')
    except:
        # Initialize a new simulation
        object_names = ['Sun'] + asteroid_names       
        # The epoch as a horizon date string
        horizon_date = mjd_to_horizons(epoch_mjd)
        
        # Initialize simulation
        sim = make_sim_horizons(object_names=object_names, horizon_date=horizon_date)
        
        # Save a snapshot to the archive file
        sim.simulationarchive_snapshot(filename=fname)
    
    return sim

# ********************************************************************************************************************* 
def test_one_asteroid(sim: rebound.Simulation, asteroid_name: str, epoch_mjd: int, verbose: bool = False):
    """Test whether orbital elements of the named asteroid are recovered vs. Horizons"""
    # Convert epoch to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    # Filename for archive
    file_date = epoch_dt.strftime('%Y-%m-%d_%H-%M')
    fname: str = f'../data/asteroids/{asteroid_name}_{file_date}.bin'    

    # Create simulation using Horizons in automatic mode
    asteroid_names = [asteroid_name]
    sim1 = make_sim_asteroids_horizons(fname=fname, asteroid_names=asteroid_names, epoch_mjd=epoch_mjd)
    primary1 = sim1.particles['Sun']
    p1 = sim1.particles[asteroid_name]
    orb1 = p1.calculate_orbit(primary=primary1)
    
    # The simulation to be tested, which was assembled from the data file
    sim2 = sim
    primary2 = sim2.particles['Sun']
    p2 = sim2.particles[asteroid_name]
    orb2 = p2.calculate_orbit(primary=primary2)

    # Compute errors in cartesian coordinates
    q1 = np.array([p1.x, p1.y, p1.z]) - np.array([primary1.x, primary1.y, primary1.z])
    q2 = np.array([p2.x, p2.y, p2.z]) - np.array([primary2.x, primary2.y, primary2.z])
    q = np.linalg.norm(q2 - q1)
    v1 = np.array([p1.vx, p1.vy, p1.vz]) - np.array([primary1.vx, primary1.vy, primary1.vz])
    v2 = np.array([p2.vx, p2.vy, p2.vz]) - np.array([primary2.vx, primary2.vy, primary2.vz])
    v = np.linalg.norm(v2 - v1)

    # Compute errors in orbital elements
    a = np.abs(orb2.a - orb1.a)
    e = np.abs(orb2.e - orb1.e)
    inc = np.abs(orb2.inc - orb1.inc)
    Omega = np.abs(orb2.Omega - orb1.Omega)
    omega = np.abs(orb2.omega - orb1.omega)
    f = np.abs(orb2.f - orb1.f)
    
    # Report errors if requested
    if verbose:
        print(f'\nErrors in recovered orbital elements for {asteroid_name}:')
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
def test_element_recovery(epoch_mjd: float, verbose: bool = False):
    """Test recovery of initial orbital elements for selected asteroids"""
    # List of asteroids to test
    # Started with first 10
    # Juno has an ambiguous name
    # Hebe matches to Thebe
    # Metis matches, but the Horizons orbit has messed up elements with a=-3.025, e=1.245
    asteroid_names = ['Ceres', 'Pallas', 'Vesta', 'Astraea', 'Flora', 'Hygiea', 'Parthenope', 'Victoria',
                      'Egeria', 'Irene', 'Eunomia', 'Psyche', 'Thetis', 'Melpomene', 'Fortuna', 'Massalia',
                      'Lutetia', 'Kalliope', 'Thalia', 'Phocaea']

    # Load data from JPL asteroids file
    df_in = load_data()
    # Convert data to rebound format
    df = convert_data(df_in=df_in)
    
    # Convert to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    
    # Rebound simulation of the planets and moons on this date
    sim_base = make_sim_moons(epoch_dt)
    sim_clean_names(sim_base)
        
    # Add selected asteroids
    sim = make_sim_asteroids(sim_base=sim_base, df=df, n0=1, n1=31)

    # Assemble array of errors in q and v
    N = len(asteroid_names)
    q = np.zeros(N)
    v = np.zeros(N)
    for i, asteroid_name in enumerate(asteroid_names):
        # print(f'{asteroid_name}')
        qi, vi = test_one_asteroid(sim=sim, asteroid_name=asteroid_name, epoch_mjd=epoch_mjd, verbose=verbose)
        q[i], v[i] = qi, vi
    
    # Compute RMS error in position and velocity
    q_rms = rms(q)
    v_rms = rms(v)

    # Report results
    print(f'Testing recovery of orbital elements vs. Horizons for')
    print(', '.join(asteroid_names))
    print(f'RMS error over {N} asteroids:')
    print(f'q: {q_rms:5.3e}')
    print(f'v: {v_rms:5.3e}')

# ********************************************************************************************************************* 
def main():
    """Main routine for integrating the orbits of known asteroids"""
    # Process command line arguments
    parser = argparse.ArgumentParser(description='Integrate the orbits of known asteroids from JPL ephemeris file.')
    parser.add_argument('n0', metavar='n0', type=int, help='the first asteroid number to process')
    parser.add_argument('-n_ast', metavar='B', type=int, default=20000,
                        help='the number of asteroids to process in this batch')
    args = parser.parse_args()
    
    # Unpack command line arguments
    n0 = args.n0
    n1 = n0 + args.n_ast

    # Status
    print(f'Processing asteroids {n0} to {n1}...')

    # Load data from JPL asteroids file
    df_in = load_data()
    # Convert data to rebound format
    ast_elt = convert_data(df_in=df_in)

    # Reference epoch for asteroids file
    epoch_mjd: float = 58600.0
    # Convert to a datetime
    epoch_dt: datetime = mjd_to_datetime(epoch_mjd)
    
    # Start and end times of simulation
    dt0: datetime = datetime(2000, 1, 1)
    dt1: datetime = datetime(2040,12,31)
    
    # Rebound simulation of the planets and moons on this date
    # sim_base = make_sim_planets(epoch_dt)
    steps_per_day: int = 4
    sim_base = make_sim_moons(t=epoch_dt, steps_per_day=steps_per_day)
    sim_clean_names(sim_base)
        
    # Add selected asteroids
    sim = make_sim_asteroids(sim_base=sim_base, ast_elt=ast_elt, n0=n0, n1=n1)
    
    # Test whether initial elements match Horizons on selected asteroids
    # test_one_asteroid(sim=sim, asteroid_name='Ceres', epoch_mjd=epoch_mjd)
    # test_element_recovery(epoch_mjd=epoch_mjd, verbose=True)
    
    # Integrate the asteroids from dt0 to dt1 with a time step of 1 day
    fname = f'../data/asteroids/sim_asteroids_n_{n0:06}_{n1:06}.bin'
    time_step = 1
    make_archive(fname_archive=fname, sim_epoch=sim, epoch_dt=epoch_dt, dt0=dt0, dt1=dt1, time_step=time_step)

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
