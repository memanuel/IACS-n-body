"""
Michael S. Emanuel
Tue Jun  4 15:24:22 2019
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import plot_style

# *************************************************************************************************
def make_ellipse(a, b):
    """
    Create an array of points corresponding to a 2D ellipse x^2/a^2 + y^2 / b^2 = 1
    https://en.wikipedia.org/wiki/Ellipse
    """
    # Set the "time" points t
    two_pi = 2.0 * np.pi
    t = np.linspace(0.0, 1.0, 361) * two_pi
    # The x and y coordinates
    x = a * np.cos(t)
    y = b * np.sin(t)
    # Assemble them into a Tx2 array
    q = np.stack([x, y]).T
    return q


# *************************************************************************************************
def plot_points(q, title):
    """Plot the points in q=[x, y]"""
    # Unpack x and y from q
    x = q[:, 0]
    y = q[:, 1]
    
    # Make plot
    fig, ax = plt.subplots(figsize=[12,12])
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(x, y, color = 'blue')
    return fig, ax


# *************************************************************************************************
plot_style()
# Ellipse shape
a = 1.050
b = 1.000
# Eccentricity
e = np.sqrt(1.0 - (b/a)**2)

q = make_ellipse(a, b)
title = f'Ellipse of Eccentricity {e:0.3f}'
plot_points(q, title)
