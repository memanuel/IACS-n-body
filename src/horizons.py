"""
Harvard IACS Masters Thesis
Utilities for NASA Horizons system

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import rebound
from datetime import datetime
import pickle
import glob
from typing import List, Dict

# Local imports
from astro_utils import mjd_to_jd

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
# Missing masses
# https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
mass_tbl = {
    'Eris': 16.7E21,
    'Makemake': 4.4E21,
    'Haumea': 4.006E21,
    '2007 OR10': 1.75E21,
    'Quaoar': 1.4E21,
    'Ceres': 0.939E21,
    'Orcus': 0.641E21,
    'Hygiea': 10.76E20,
    'Varuna': 3.7E20,
    'Varda': 2.664E20,
    'Vesta': 2.59E20,
    'Pallas': 2.11E20,
    '229762': 1.361E20,
    '2002 UX25': 1.25E20
    }

# Convert to solar masses
mass_sun = 1988550000.0E21
for nm in mass_tbl:
    mass_tbl[nm] /= mass_sun

# ********************************************************************************************************************* 
# Table mapping MSE object names to NASA NAIF IDs
# the offset for asteroid IDs in spice IDs is 2000000
ao: int = 2000000
name_to_id: Dict[str, int] = {
    # Sun
    'Solar System Barycenter': 0,
    'Sun': 10,
    # Mercury
    'Mercury Barycenter': 199,
    'Mercury': 199,
    # Venus
    'Venus Barycenter': 299,
    'Venus': 299,
    # Earth and the Moon
    'Earth Barycenter': 3,
    'Earth': 399,
    'Moon': 301,
    # Mars and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Mars
    'Mars Barycenter': 4,
    'Mars': 499,
    'Phobos': 401,
    'Deimos': 402,
    # Jupiter and its moons
    # https://en.wikipedia.org/wiki/Galilean_moons
    'Jupiter Barycenter': 5,
    'Jupiter': 599,
    'Io': 501,
    'Europa': 502,
    'Ganymede': 503,
    'Callisto': 504,
    # Saturn and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Saturn
    'Saturn Barycenter': 6,
    'Saturn': 699,
    'Mimas': 601,
    'Enceladus': 602,
    'Tethys': 603,
    'Dione': 604,
    'Rhea': 605,
    'Titan': 606,
    'Iapetus': 608,
    # Uranus and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Uranus
    'Uranus Barycenter': 7,
    'Uranus': 799,
    'Ariel': 701,
    'Umbriel': 702,
    'Titania': 703,
    'Oberon': 704,
    'Miranda': 705,
    # Neptune and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Neptune
    'Neptune Barycenter': 8,
    'Neptune': 899,
    'Triton': 801,
    # Pluto and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Pluto
    'Pluto Barycenter': 9,
    'Pluto': 999,
    'Charon': 901,
    # Miscellaneous heavy bodies
    'Eris': ao + 136199, 
    'Makemake': ao + 136472,
    'Haumea': ao + 136108, 
    '2007 OR10': ao + 225088, 
    'Quaoar': ao + 50000,
    'Ceres': ao + 1, 
    'Orcus': ao + 90482, 
    'Hygiea': ao + 10, 
    'Varuna': ao + 20000, 
    'Varda': ao + 174567, 
    'Vesta': ao + 4, 
    'Pallas': ao + 2, 
    '229762': ao + 229762, 
    '2002 UX25': ao + 55637,    
    }

# ********************************************************************************************************************* 
def object_to_horizon_names(object_names):
    """Convert from user friendly object names to Horizons names"""
    # See https://ssd.jpl.nasa.gov/horizons.cgi#top for looking up IDs
    # Initialize every name to map to itself
    object_name_to_hrzn : Dict['str', 'str'] = {nm: nm for nm in object_names}
    
    # Earth and the Moon
    object_name_to_hrzn['Earth Barycenter'] = '3'
    object_name_to_hrzn['Earth'] = '399'
    object_name_to_hrzn['Moon'] = '301'
    
    # Mars
    object_name_to_hrzn['Mars Barycenter'] = '4'
    object_name_to_hrzn['Mars'] = '499'

    # Jupiter
    object_name_to_hrzn['Jupiter Barycenter'] = '5'
    object_name_to_hrzn['Jupiter'] = '599'

    # Saturn
    object_name_to_hrzn['Saturn Barycenter'] = '6'
    object_name_to_hrzn['Saturn'] = '699'
    
    # Uranus
    object_name_to_hrzn['Uranus Barycenter'] = '7'
    object_name_to_hrzn['Uranus'] = '799'

    # Neptune and its moons
    object_name_to_hrzn['Neptune Barycenter'] = '8'
    object_name_to_hrzn['Neptune'] = '899'

    # Pluto and its moons
    object_name_to_hrzn['Pluto Barycenter'] = '9'
    object_name_to_hrzn['Pluto'] = '999'

    # List of horizon object names corresponding to these input object names
    horizon_names = [object_name_to_hrzn[nm] for nm in object_names]

    return horizon_names

# ********************************************************************************************************************* 
def make_sim_horizons(object_names: List[str], horizon_date: str):
    """Create a new rebound simulation with initial data from the NASA Horizons system"""
    # Create a simulation
    sim = rebound.Simulation()
    
    # Set units
    sim.units = ('day', 'AU', 'Msun')
    
    # Convert from user friendly object names to Horizons names
    horizon_names = object_to_horizon_names(object_names)

    # Add these objects from Horizons
    print(f'Searching Horizons as of {horizon_date}.')
    sim.add(horizon_names, date=horizon_date)
    
    # Add hashes for the object names
    for i, particle in enumerate(sim.particles):
        particle.hash = rebound.hash(object_names[i])
        
    # Move to center of mass
    sim.move_to_com()
    
    return sim

# ********************************************************************************************************************* 
def extend_sim_horizons(sim: rebound.Simulation, object_names: List[str], horizon_date: str):
    """Extend a rebound simulation with initial data from the NASA Horizons system"""
    # Number of particles
    N_orig = sim.N
    
    # Convert from user friendly object names to Horizons names
    horizon_names = object_to_horizon_names(object_names)

    # Add these objects from Horizons
    print(f'Searching Horizons as of {horizon_date}.')
    sim.add(horizon_names, date=horizon_date)
    
    # Add hashes for the object names
    for i, p in enumerate(sim.particles[N_orig:]):
        object_name = object_names[i]
        p.hash = rebound.hash(object_name)
        if p.m == 0.0:
            try:
                m = mass_tbl[object_name]
                p.m = m
                print(f'Added mass {m:5.3e} Msun for {object_name}.')
            except:
                print(f'Could not find mass for {object_name}.')

    # Move to center of mass
    sim.move_to_com()
    
    return sim

# ********************************************************************************************************************* 
def init_horizons_cache():
    """Initialize cache of Horizons entries"""
    
    # Create an empty dictionary; key is (object_id, datetime), value is (m, qx, qy, qz, vy, vy, vz)
    hrzn = dict()
    
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
    object_name_correction['Earth Geocenter'] = 'Earth'
    object_name_correction['Mars Geocenter'] = 'Mars'
    object_name_correction['Jupiter Geocenter'] = 'Jupiter'
    object_name_correction['Saturn Geocenter'] = 'Saturn'
    object_name_correction['Uranus Geocenter'] = 'Uranus'
    object_name_correction['Neptune Geocenter'] = 'Neptune'
    
    fnames = glob.glob('../data/planets/moons_*.bin')
    for fname in fnames:
        dt_str = fname.replace('../data/planets/moons_', '')
        dt_str = dt_str.replace('.bin', '')
        date_str = dt_str[0:10]
        time_str = dt_str[11:16].replace('-', ':')
        epoch = datetime.strptime(f'{date_str} {time_str}', '%Y-%m-%d %H:%M')
        # print(f'{date_str} {time_str} = {epoch}')
        # Load the simulation as of this date
        sim = rebound.Simulation(fname)
        for object_name in object_names_moons:
            # Look up the integer object_id for this body
            object_name_short = object_name_correction[object_name]
            object_id = name_to_id[object_name_short]
            # The dictionary key
            key = (epoch, object_id)
            # The particle
            # print(f'{object_name_short}')
            p = sim.particles[object_name]
            # Save a dictionary of this particle's attributes
            hrzn[key] = {
                'm': p.m,
                'x': p.x,
                'y': p.y,
                'z': p.z,
                'vx': p.vx,
                'vy': p.vy,
                'vz': p.vz,
                }
    
    # Save the cache to disk in the jpl directory
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

# init_horizons_cache()

hrzn = load_horizons_cache()

