"""
Harvard IACS Masters Thesis
Trajectories for Known Asteroids

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Magic command to run this file using remote kernel
# %run "/d/IACS/n-body/src/asteroid_traj.py"

import rebound
from sej_data import make_sim_horizons


# Create simulation using horizons
sim = rebound.Simulation()
object_names = ['Sun', 'Ceres']
horizon_date = '2000-01-01 12:00'
