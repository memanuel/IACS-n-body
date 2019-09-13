"""
Harvard IACS Masters Thesis
Utilities for NASA Horizons system

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import rebound
from datetime import datetime
import collections
import pickle
from typing import List, Dict, Set

# Local imports
from astro_utils import mjd_to_jd
from solar_system_objects import name_to_id, mass_tbl

# *************************************************************************************************
def datetime_to_horizons(t: datetime):
    """Convert a Python datetime to a datetime string understood by NASA Horizons"""
    return t.strftime('%Y-%m-%d %H:%M')

# *************************************************************************************************
def jd_to_horizons(jd: float):
    """Convert a Julian Day to a string understood by NASA Horizons"""
    return f'JD{jd:.8f}'

# *************************************************************************************************
def mjd_to_horizons(mjd: float):
    """Convert a Modified Julian Day to a string understood by NASA Horizons"""
    jd = mjd_to_jd(mjd)
    return jd_to_horizons(jd)

# ********************************************************************************************************************* 
# Convert from user friendly object names to Horizons names
# See https://ssd.jpl.nasa.gov/horizons.cgi#top for looking up IDs

# Initialize every name to map to itself
object_to_horizon_name = {name: name for name in name_to_id}

# Overrides to handle planet vs. barycenter naming ambiguity
overrides = {
    'Mercury Barycenter': '1',
    'Mercury': '199',
    'Venus Barycenter': '2',
    'Venus': '299',
    'Earth Barycenter': '3',
    'Earth': '399',
    'Moon': '301',
    'Mars Barycenter': '4',
    'Mars': '499',
    'Jupiter Barycenter': '5',
    'Jupiter': '599',
    'Saturn Barycenter': '6',
    'Saturn': '699',
    'Uranus Barycenter': '7',
    'Uranus': '799',
    'Neptune Barycenter': '8',
    'Neptune': '899',
    'Pluto Barycenter': '9',
    'Pluto': '999',
    # Asteroids with ambiguous names
    '52 Europa': 'NAME=Europa',
    # 'Juno': 'NAME=Juno',
    # 'Hebe': 'NAME=Hebe',
    # 'Iris': 'NAME=Iris',    
    'Sila': '79360',
    }

# Overrides for asteroids
for object_name, object_id in name_to_id.items():
    if (2000000 < object_id) and (object_id < 3000000) and object_name not in overrides:
        overrides[object_name] = f'NAME={object_name}'

# Apply the overrides
for object_name, horizon_name in overrides.items():
    object_to_horizon_name[object_name] = horizon_name

# ********************************************************************************************************************* 
horizon_entry = collections.namedtuple('horizon_entry', ['m', 'x', 'y', 'z', 'vx', 'vy', 'vz', 
                                       'object_name', 'object_id', 'horizon_name'])    

# ********************************************************************************************************************* 
def save_horizons_cache(hrzn):
    """Save the Horizons cache"""
    fname_cache = '../jpl/horizon_cache.pickle'    
    with open(fname_cache, 'wb') as fh:
        pickle.dump(hrzn, fh)

# ********************************************************************************************************************* 
def load_horizons_cache():
    """Load the Horizons cache"""
    fname_cache = '../jpl/horizon_cache.pickle'    
    with open(fname_cache, 'rb') as fh:
        hrzn = pickle.load(fh)
    return hrzn

# ********************************************************************************************************************* 
def purge_horizons_cache(object_name: str, epoch: datetime = None):
    """Purge all entries of the named object from the Horizons cache"""
    # Load the cache
    hrzn = load_horizons_cache()
    
    # The object_id of the object to purge
    object_id_purged = name_to_id[object_name]
    
    # The epochs to purge
    epoch_purged = epoch
    purge_all_epochs = epoch_purged is None
    
    # Iterate through entries to find which keys will be purged
    keys_to_purge = []
    for (epoch, object_id) in hrzn:
        if (object_id == object_id_purged) and (epoch == epoch_purged or purge_all_epochs):
            keys_to_purge.append((epoch, object_id))

    # Delete selected keys
    for key in keys_to_purge:
        del hrzn[key]

    # Save the revised cache and report
    save_horizons_cache(hrzn)
    num_deleted: int = len(keys_to_purge)
    print(f'Deleted {num_deleted} entries from horizons cache for {object_name}, epoch={epoch_purged}')

# ********************************************************************************************************************* 
def add_one_object_hrzn(sim: rebound.Simulation, object_name: str, epoch: datetime, hrzn: Dict):
    """Add one object to a simulation with data fromm horizons (cache or API)."""
    # Identifiers for this object
    object_id = name_to_id[object_name]
    key = (epoch, object_id)

    try:
        # Try to look up the object on the horizons cache
        p: horizon_entry = hrzn[key]
        sim.add(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz, hash=rebound.hash(object_name))
    except KeyError:
        # Search string for the horizon API
        horizon_name = object_to_horizon_name[object_name]
        # Convert epoch to a horizon date string
        horizon_date: str = datetime_to_horizons(epoch)
        print(f'Searching Horizons as of {horizon_date}')
        # Add the particle
        sim.add(horizon_name, date=horizon_date)
        # Set the mass and hash of this particle
        p: rebound.Particle = sim.particles[-1]
        # p.m = mass_tbl[object_name]
        p.m = mass_tbl.get(object_name, 0.0)
        p.hash = rebound.hash(object_name)
        # Create an entry for this particle on the cache
        entry: horizon_entry = horizon_entry(m=p.m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz, 
                                             object_name=object_name, object_id=object_id, horizon_name=horizon_name)
        hrzn[key] = entry
        
        # Save the revised cache
        save_horizons_cache(hrzn)

# ********************************************************************************************************************* 
def make_sim_horizons(object_names: List[str], epoch: datetime) -> rebound.Simulation:
    """Create a new rebound simulation with initial data from the NASA Horizons system"""
    # Create a simulation
    sim = rebound.Simulation()
    
    # Set units
    sim.units = ('day', 'AU', 'Msun')
    
    # Add objects one at a time
    for object_name in object_names:
        add_one_object_hrzn(sim=sim, object_name=object_name, epoch=epoch, hrzn=hrzn)
        
    # Move to center of mass
    sim.move_to_com()
    
    return sim

# ********************************************************************************************************************* 
def extend_sim_horizons(sim: rebound.Simulation, object_names: List[str], epoch: datetime) -> None:
    """Extend a rebound simulation with initial data from the NASA Horizons system"""
    # Generate list of missing object names
    hashes_present: Set[int] = set(p.hash.value for p in sim.particles)
    objects_missing: List[str] = [nm for nm in object_names if rebound.hash(nm).value not in hashes_present]
    
    # Add missing objects one at a time if not already present
    for object_name in objects_missing:
        add_one_object_hrzn(sim=sim, object_name=object_name, epoch=epoch, hrzn=hrzn)        
    
    # Move to center of mass
    sim.move_to_com()

# ********************************************************************************************************************* 
try:
    hrzn = load_horizons_cache()
    # print(f'Loaded Horizons cache with {len(hrzn)} entries.')
except:
    hrzn = dict()
    # print(f'Initialized empty Horizons cache.')
