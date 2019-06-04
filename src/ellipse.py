"""
Michael S. Emanuel
Tue Jun  4 15:24:22 2019
"""

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import keras

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


def plot_points(q):
    """Plot the points in q= [x, y]"""
    fig, ax = plt.subplots()

a = 1.01
b = 1.0
q = make_ellipse(a, b)
