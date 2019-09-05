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
from datetime import datetime
from tqdm import tqdm as tqdm_console
from typing import List, Dict

# Local imports
from astro_utils import mjd_to_horizons, datetime_to_horizons, datetime_to_mjd, mjd_to_datetime
from utils import rms

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
def make_sa_ceres(n_years: int, sample_freq:int ):
    """Create or load the sun-earth-jupiter system at start of J2000.0 frame"""
    
    # The name of the simulation archive
    fname_archive = '../data/asteroids/ss_ceres.bin'
    
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

df = load_data()
