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
from tqdm import tqdm as tqdm_console
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
    df = pd.read_fwf(fname, colspecs=colspecs, header=header, names=names, skiprows=skiprows, dtype=dtype)
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
def make_sim_asteroids(sim_base: rebound.Simulation, df: pd.DataFrame, n0: int, n1: int):
    """
    Create a simulation with the selected asteroids by their ID numbers.
    INPUTS:
    sim_base: the base simulation, with e.g. the sun, planets, and selected moons
    df: the DataFrame with asteroid orbital elements at the specified epoch
    n0: the first asteroid number to add, inclusive
    n1: the last asteroid number to add, exclusive
    """
    # Start with a copy of the base simulation
    sim = sim_base.copy()
    # The particles
    ps = sim.particles
    # Set the primary to the sun (NOT the solar system barycenter!)
    primary = ps['Sun']
    # Set the number of active particles to the base simulation
    # https://rebound.readthedocs.io/en/latest/ipython/Testparticles.html
    sim.N_active = sim_base.N
    # Add the specified asteroids one at a time
    for num in range(n0, n1):
        # Unpack the orbital elements
        a = df.a[num]
        e = df.e[num]
        inc = df.inc[num]
        Omega = df.Omega[num]
        omega = df.omega[num]
        M = df.M[num]
        name = df.Name[num]
        # Add the new asteroid
        sim.add(m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M, primary=primary)
        # Set the hash to the asteroid's name
        ps[-1].hash = rebound.hash(name)

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
def test_element_recovery(sim_base: rebound.Simulation, df:pd.DataFrame, asteroid_name: str, epoch_mjd: int):
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
    
    # Create simulation using data from file
    sim2 = make_sim_asteroids(sim_base=sim_base, df=df, n0=1, n1=2)
    primary2 = sim2.particles['Sun']
    p2 = sim2.particles[asteroid_name]
    orb2 = p2.calculate_orbit(primary=primary2)

    # Compute errors
    print(f'Errors in recovered orbital elements for {asteroid_name}:')
    print(f'a    : {np.abs(orb2.a - orb1.a):5.3e}')
    print(f'e    : {np.abs(orb2.e - orb1.e):5.3e}')
    print(f'inc  : {np.abs(orb2.inc - orb1.inc):5.3e}')
    print(f'Omega: {np.abs(orb2.Omega - orb1.Omega):5.3e}')
    print(f'omega: {np.abs(orb2.omega - orb1.omega):5.3e}')
    print(f'f    : {np.abs(orb2.f - orb1.f):5.3e}')

# ********************************************************************************************************************* 

# Load data from JPL asteroids file
df_in = load_data()
# Convert data to rebound format
df = convert_data(df_in=df_in)

# Reference epoch for asteroids file
epoch_mjd: float = 58600.0
# Convert to a datetime
epoch_dt: datetime = mjd_to_datetime(epoch_mjd)

# Start and end times of simulation
dt0: datetime = datetime(2000, 1, 1)
dt1: datetime = datetime(2040, 1, 1)

# Rebound simulation of the planets and moons on this date
sim_base = make_sim_moons(epoch_dt)
sim_clean_names(sim_base)
# ps = sim_base.particles
    
# Add selected asteroids
n0=1
n1=11
sim = make_sim_asteroids(sim_base=sim_base, df=df, n0=n0, n1=n1)
ps = sim.particles
primary = sim.particles['Sun']
p = sim.particles['Ceres']
orb = p.calculate_orbit(primary=primary)

# Test whether initial elements match Horizons
test_element_recovery(sim_base=sim_base, df=df, asteroid_name='Ceres', epoch_mjd=epoch_mjd)

# Integrate the planets and moons from dt0 to dt1
fname = '../data/planets/sim_moons_2000-2040_ias15_dt2.bin'
