"""
Harvard IACS Masters Thesis
Solar system objects

Michael S. Emanuel
Fri Aug 23 16:13:28 2019
"""

# Library imports
import rebound
from typing import Dict

# ********************************************************************************************************************* 
# Table of planets and their moons
moon_tbl = {
    'Mercury': tuple(),
    'Venus': tuple(),
    'Earth': ('Moon',),
    'Mars': ('Deimos', 'Phobos',),
    'Jupiter': ('Io', 'Europa', 'Ganymede', 'Callisto',),
    'Saturn': ('Mimas', 'Enceladus', 'Tethys', 'Dione', 'Rhea', 'Titan', 'Hyperion', 'Iapetus', 'Phoebe'),
    'Uranus': ('Ariel', 'Umbriel', 'Titania', 'Oberon',),
    'Neptune': ('Triton', 'Proteus',),
    'Pluto': ('Charon',)
    }

# ********************************************************************************************************************* 
# Table mapping MSE object names to NASA NAIF IDs
# the offset for asteroid IDs in spice IDs is 2000000
ao: int = 2000000
uo: int = 3000000
name_to_id: Dict[str, int] = {
    # Sun
    'Solar System Barycenter': 0,
    'Sun': 10,
    # Mercury
    'Mercury Barycenter': 1,
    'Mercury': 199,
    # Venus
    'Venus Barycenter': 2,
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
    'Hyperion': 607,
    'Iapetus': 608,
    'Phoebe': 609,
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
    'Proteus': 802,
    # Pluto and its moons
    # https://en.wikipedia.org/wiki/Moons_of_Pluto
    'Pluto Barycenter': 9,
    'Pluto': 999,
    'Charon': 901,
    # Miscellaneous heavy asteroids
    'Ceres': ao+1,
    'Pallas': ao+2,
    'Juno': ao+3,
    'Vesta': ao+4,
    'Astraea': ao+5,
    'Hebe': ao+6,
    'Iris': ao+7,
    'Flora': ao+8,
    'Metis': ao+9,    
    'Hygiea': ao+10,
    'Parthenope': ao+11,
    'Victoria': ao+12,
    'Egeria': ao+13,
    'Irene': ao+14,
    'Eunomia': ao+15,
    'Psyche': ao+16,
    'Thetis': ao+17,
    'Melpomene': ao+18,
    'Fortuna': ao+19,
    'Massalia': ao+20,
    'Lutetia': ao+21,
    'Kalliope': ao+22,
    'Thalia': ao+23,
    'Themes': ao+24,
    'Phocaea': ao+25,
    'Euphrosyne': ao+31,
    'Eugenia': ao+45,
    'Doris': ao+48,
    '52 Europa': ao+52,
    'Cybele': ao+65,
    'Sylvia': ao+87,
    'Thisbe': ao+88,
    'Camilla': ao+107,
    'Bamberga': ao+324,
    'Patientia': ao+451,
    'Davida': ao+511,
    'Hektor': ao+624,
    'Interamnia': ao+704,
    'Varuna': ao+20000,
    '1998 SM165': ao+26308,
    'Quaoar': ao+50000,
    '2002 UX25': ao+55637,
    'Ceto': ao+65489,
    'Sila': ao+79360,
    'Orcus': ao+90482,
    '2002 WC19': ao+119979,
    'Salacia': ao+120347,
    'Haumea': ao+136108,
    'Eris': ao+136199,
    'Makemake': ao+136472,
    '2004 UX10': ao+144897,
    'Varda': ao+174567,
    '2007 OR10': ao+225088,
    '229762': ao+229762,
    # Miscellaneous heavy non-numbered asteroids
    'Vanth': uo +  1,
    'Hiʻiaka': uo +  2,
    '2001 QC298': uo +  3,
    }

# ********************************************************************************************************************* 
# Dictionary mapping from ID to name
id_to_name: Dict[int, str] = {name_to_id[name]: name for name in name_to_id}

# Dictionary mapping from rebound hash to name
hash_to_name: Dict[int, str] = {rebound.hash(name_to_id[name]).value : name for name in name_to_id}
name_to_hash: Dict[str, int] = {name : rebound.hash(name_to_id[name]).value for name in name_to_id}

# ********************************************************************************************************************* 
# Mass of all solar system objects available on Wikipedia
# https://en.wikipedia.org/wiki/List_of_Solar_System_objects_by_size
mass_tbl = {
    'Sun':          1.000000000000E+00,
    # Inner planets
    'Mercury':      1.660506399135E-07,
    'Venus':        2.448266324709E-06,
    'Earth':        3.003997887908E-06,
    'Moon':         3.696160518971E-08,
    'Mars':         3.227728747077E-07,
    'Phobos':       5.431093007468E-15,
    'Deimos':       1.005757964346E-15,
    # Jupiter
    'Jupiter':      9.547660355535E-04,
    'Io':           4.490709310804E-08,
    'Europa':       2.413819114430E-08,
    'Ganymede':     7.452666515803E-08,
    'Callisto':     5.410977848181E-08,
    # Saturn
    'Saturn':       2.858665862060E-04,
    'Mimas':        1.885293304166E-11,
    'Enceladus':    5.431093007468E-11,
    'Tethys':       3.104271956954E-10,
    'Dione':        5.511553644615E-10,
    'Rhea':         1.164969450102E-09,
    'Titan':        6.763722310226E-08,
    'Hyperion':     2.826179879812E-12,
    'Iapetus':      9.926328229112E-10,
    'Phoebe':       4.169872520178E-12,
    # Uranus
    'Uranus':       4.366598778004E-05,
    'Ariel':        6.788866259335E-10,
    'Umbriel':      6.034547786075E-10,
    'Titania':      1.773151291142E-09,
    'Oberon':       1.515677252269E-09,
    'Miranda':      3.313972492520E-11,
    # Neptune
    'Neptune':      5.150989414397E-05,
    'Triton':       1.081189811672E-08,
    'Proteus':      2.212667521561E-11,
    # Pluto
    'Pluto':        6.590229061376E-09,
    'Charon':       7.643760529029E-10,
    # Asteroids
    'Ceres':        4.722033642604E-10,
    'Pallas':       1.061074652385E-10,
    'Juno':         1.372859621332E-11,
    'Vesta':        1.302456563828E-10,
    'Iris':         6.914586004878E-12,
    'Hygiea':       5.410977848181E-10,
    'Egeria':       4.435392622765E-12,
    'Eunomia':      1.579040004023E-11,
    'Psyche':       1.211938347037E-11,
    'Fortuna':      4.324759246687E-12,
    'Euphrosyne':   6.386563073596E-12,
    'Eugenia':      2.831208669634E-12,
    'Doris':        3.077619370898E-12,
    '52 Europa':    1.196851977572E-11,
    'Cybele':       6.839154157552E-12,
    'Sylvia':       7.432551356516E-12,
    'Thisbe':       7.694048427246E-12,
    'Camilla':      5.632244600337E-12,
    'Bamberga':     5.179653516381E-12,
    'Patientia':    5.481380905685E-12,
    'Davida':       1.699730959745E-11,
    'Hektor':       3.972743959166E-12,
    'Interamnia':   1.649443061527E-11,
    'Varuna':       1.860652234040E-10,
    '1998 SM165':   3.454778607528E-12,
    'Quaoar':       7.040305750421E-10,
    '2002 UX25':    6.285987277162E-11,
    'Ceto':         2.715546503734E-12,
    'Sila':         5.431093007468E-12,
    'Orcus':        3.223454275729E-10,
    '2002 WC19':    3.872168162732E-11,
    'Salacia':      2.202609941917E-10,
    'Haumea':       2.014533202585E-09,
    'Eris':         8.398079002288E-09,
    'Makemake':     2.212667521561E-09,
    '2004 UX10':    1.508636946519E-11,
    'Varda':        1.339669608509E-10,
    '2007 OR10':    8.800382188026E-10,
    '229762':       6.844182947374E-11,
    'Vanth':        2.011515928692E-11,
    'Hiʻiaka':      9.001533780896E-12,
    '2001 QC298':   5.974202308215E-12,
    }


# Add barycenter entries
for planet_name, moons in moon_tbl.items():
    mass_tbl[f'{planet_name} Barycenter'] = mass_tbl[planet_name] + sum(mass_tbl[mn] for mn in moon_tbl['Mercury'])
