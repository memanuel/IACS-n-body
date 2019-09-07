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
import glob
from typing import List, Dict

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
    'Pluto': '999'
    }
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
        # Add the particle
        sim.add(horizon_name, date=horizon_date)
        # Set the mass and hash of this particle
        p: rebound.Particle = sim.particles[-1]
        p.m = mass_tbl[object_name]
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
    # Add new objects one at a time if not already present
    for object_name in object_names:
        if object_name not in sim.particles:
            add_one_object_hrzn(sim=sim, object_name=object_name, epoch=epoch, hrzn=hrzn)        
    
    # Move to center of mass
    sim.move_to_com()

# ********************************************************************************************************************* 
def init_horizons_cache():
    """Initialize cache of Horizons entries"""
    
    # Create an empty dictionary; key is (object_id, datetime), value is (m, qx, qy, qz, vy, vy, vz)
    hrzn = dict()
    
    # The object names saved in the planet snapshots
    object_names_planets = \
        ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    # The object names saved in the moons snapshots
    object_names_moons = \
        ['Sun', 'Mercury', 'Venus', 
         'Earth Geocenter', 'Moon', 
         'Mars Geocenter', 'Phobos', 'Deimos',
         'Jupiter Geocenter', 'Io', 'Europa', 'Ganymede', 'Callisto',
         'Saturn Geocenter', 'Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Iapetus',
         'Uranus Geocenter', 'Ariel', 'Umbriel', 'Titania', 'Oberon', 'Miranda',
         'Neptune Geocenter', 'Triton',
         'Pluto', 'Charon',
         # These objects don't have mass on Horizons; mass added with table
         'Eris', 'Makemake', 'Haumea', '2007 OR10', 'Quaoar',
         'Ceres', 'Orcus', 'Hygiea', 'Varuna', 'Varda', 'Vesta', 'Pallas', '229762', '2002 UX25'
         ]
    
    # Correction from original object names
    object_name_correction = {nm: nm for nm in object_names_moons}
    corrections  = {
        # Corrections for planets shapshots
        'Earth': 'Earth Barycenter',
        'Mars': 'Mars Barycenter',
        'Jupiter' : 'Jupiter Barycenter',
        'Saturn': 'Saturn Barycenter',
        'Uranus': 'Uranus Barycenter',
        'Neptune': 'Neptune Barycenter',
        # Corrections for moons shapshots
        'Earth Geocenter': 'Earth',
        'Mars Geocenter': 'Mars',
        'Jupiter Geocenter' : 'Jupiter',
        'Saturn Geocenter': 'Saturn',
        'Uranus Geocenter': 'Uranus',
        'Neptune Geocenter': 'Neptune',
        }
    for name_old, name_new in corrections.items():
        object_name_correction[name_old] = name_new
    
    # Data from the planet snapshots
    fnames = glob.glob('../data/planets/v2/planets_*.bin')
    for fname in fnames:
        dt_str = fname.replace('../data/planets/v2/planets_', '')
        dt_str = dt_str.replace('.bin', '')
        date_str = dt_str[0:10]
        time_str = dt_str[11:16].replace('-', ':')
        if date_str == '2019-04-27':
            continue
        epoch = datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M')
        # Load the simulation as of this date
        sim = rebound.Simulation(fname)
        for object_name_orig in object_names_planets:
            # Look up the integer object_id for this body
            object_name = object_name_correction[object_name_orig]
            object_id = name_to_id[object_name]
            # The horizon_name of theis body
            horizon_name = object_to_horizon_name[object_name]
            # The dictionary key
            key = (epoch, object_id)
            p = sim.particles[object_name_orig]
            # The mass
            m = mass_tbl[object_name]
            # Save a horizon_entry of this particle's attributes
            hrzn[key] = horizon_entry(m=m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz,
                                      object_name=object_name, object_id=object_id, horizon_name=horizon_name)
    # Data from the moon snapshots
    fnames = glob.glob('../data/planets/v2/moons_*.bin')
    for fname in fnames:
        dt_str = fname.replace('../data/planets/v2/moons_', '')
        dt_str = dt_str.replace('.bin', '')
        date_str = dt_str[0:10]
        time_str = dt_str[11:16].replace('-', ':')
        if date_str == '2019-04-27':
            continue
        epoch = datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M')
        # Load the simulation as of this date
        sim = rebound.Simulation(fname)
        for object_name_orig in object_names_moons:
            # Look up the integer object_id for this body
            object_name = object_name_correction[object_name_orig]
            object_id = name_to_id[object_name]
            # The horizon_name of theis body
            horizon_name = object_to_horizon_name[object_name]
            # The dictionary key
            key = (epoch, object_id)
            p = sim.particles[object_name_orig]
            # The mass
            m = mass_tbl[object_name]
            # Save a horizon_entry of this particle's attributes
            hrzn[key] = horizon_entry(m=m, x=p.x, y=p.y, z=p.z, vx=p.vx, vy=p.vy, vz=p.vz,
                                      object_name=object_name, object_id=object_id, horizon_name=horizon_name)
    
    # Save the cache
    save_horizons_cache(hrzn)

# ********************************************************************************************************************* 
try:
    hrzn = load_horizons_cache()
    print(f'Loaded Horizons cache.')
except:
    init_horizons_cache()
    hrzn = load_horizons_cache()
    # hrzn  = dict()
    print(f'Initialized Horizons cache.')

#for epoch, object_id in hrzn.keys():
#    if object_id in (199, 299):
#        object_id_bary = (object_id-99) // 100
#        key = (epoch, object_id_bary)
#        if key not in hrzn:
#            hrzn[key] = hrzn[epoch, object_id]
#            hrzn[key].object_id = object_id_bary
#            hrzn[key].object_name = object_id_bary
