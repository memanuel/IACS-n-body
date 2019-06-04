"""
Harvard IACS Masters Thesis
Utilites

Michael S. Emanuel
Tue Jun  4 15:24:22 2019
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# Type aliases
funcType = Callable[[float], float]


# *************************************************************************************************
def plot_style() -> None:
    """Set plot style for the session."""
    # Set up math plot library to use TeX
    # https://matplotlib.org/users/usetex.html
    # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rc('text', usetex=True)
    # Set default font size to 20
    mpl.rcParams.update({'font.size': 30})


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

