"""
Harvard IACS Masters Thesis
Utilites

Michael S. Emanuel
Tue Jun  4 15:24:22 2019
"""

import numpy as np
import matplotlib as mpl
# import matplotlib.pyplot as plt
import pickle
import zlib

from typing import Dict, Callable, Optional

# Type aliases
funcType = Callable[[float], float]

# *************************************************************************************************
def range_inc(x: int, y: int = None, z: int = None) -> range:
    """Return a range inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + 1, z)
    elif z < 0:
        (start, stop, step) = (x, y - 1, z)
    return range(start, stop, step)


def arange_inc(x: float, y: float = None, z: float = None) -> np.ndarray:
    """Return a numpy arange inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + z, z)
    elif z < 0:
        (start, stop, step) = (x, y - z, z)
    return np.arange(start, stop, step)

# *************************************************************************************************
def plot_style() -> None:
    """Set plot style for the session."""
    # Set default font size to 20
    mpl.rcParams.update({'font.size': 20})

# *************************************************************************************************
def print_stars(newline: bool = False):
    """Print a row of 80 stars"""
    stars = '********************************************************************************'
    row = '\n' + stars if newline else stars
    print(row)

# *************************************************************************************************
def print_header(msg: str, newline=True):
    """Print a message wrapped in two layers of stars"""
    print_stars(newline=newline)
    print(msg)
    print_stars(newline=False)

# *************************************************************************************************
# Generic root mean square of numpy arrays
def rms(x: np.array, axis=None):
    return np.sqrt(np.mean(np.square(x), axis=axis))

# *************************************************************************************************
# Serialize generic Python variables using Pickle
def load_vartbl(fname: str) -> Dict:
    """Load a dictionary of variables from a pickled file"""
    try:
        with open(fname, 'rb') as fh:
            vartbl = pickle.load(fh)
    except:
        vartbl = dict()
    return vartbl

def save_vartbl(vartbl: Dict, fname: str) -> None:
    """Save a dictionary of variables to the given file with pickle"""
    with open(fname, 'wb') as fh:
        pickle.dump(vartbl, fh)

# *************************************************************************************************
def hash_id_crc32(attributes: Dict) -> int:
    """Create a hash ID from a dictionary using the CRC 32 alogrithm"""
    # Create a non-negative hash ID of an attributes dictionary
    attributes_bytes = bytes(str(attributes), 'utf-8')
    hash_id = zlib.crc32(attributes_bytes)
    return hash_id
    