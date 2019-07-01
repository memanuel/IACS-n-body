"""
Harvard IACS Masters Thesis
Restricted Two Body Problem
Plot training data (trajectories)

Michael S. Emanuel
Tue Jun  18 15:29 2019
"""

# Library imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def plot_orbit_q(data):
    """Plot the orbit position in a training sample"""
    # Unpack data
    t = data['t']
    q = data['q']
    qx = q[:, 0]
    qy = q[:, 1]
    # Compute the distance r
    r = np.linalg.norm(q, axis=1)

    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Position')
    ax.set_xlabel('t - years')
    ax.set_ylabel('q - AU')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, qx, color='b', label='qx')
    ax.plot(t, qy, color='r', label='qy')
    ax.plot(t, r,  color='purple', label='r')
    ax.grid()
    ax.legend()
    
    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_v(data):
    """Plot the orbit velocity in a training sample"""
    # Unpack data
    t = data['t']
    v = data['v']
    vx = v[:, 0]
    vy = v[:, 1]
    spd = np.linalg.norm(v, axis=1)
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Velocity')
    ax.set_xlabel('t - years')
    ax.set_ylabel('v - AU / year')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, vx, color='b', label='vx')
    ax.plot(t, vy, color='r', label='vy')
    ax.plot(t, spd, color='purple', label='spd')
    ax.grid()
    ax.legend()

    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_a(data):
    """Plot the orbit acceleration in a training sample"""
    # Unpack data
    t = data['t']
    a = data['a']
    ax = a[:, 0]
    ay = a[:, 1]
    acc = np.linalg.norm(a, axis=1)
    
    # Plot the x and y coordinate
    # Name the axes object ax_ rather than ax to avoid a name collision with the x component of acceleration, ax
    fig, ax_ = plt.subplots(figsize=[16, 9])
    ax_.set_title('Orbit Acceleration')
    ax_.set_xlabel('t - years')
    ax_.set_ylabel('a - $AU / year^2$')
    ax_.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax_.plot(t, ax, color='b', label='ax')
    ax_.plot(t, ay, color='r', label='ay')
    ax_.plot(t, acc, color='purple', label='acc')
    ax_.grid()
    ax_.legend()

    return fig, ax_

# ********************************************************************************************************************* 
def plot_orbit_energy(data):
    """Plot the orbit energy in a training sample"""
    # Unpack data
    t = data['t']
    q = data['q']
    v = data['v']
    r = np.linalg.norm(q, axis=1)
    spd = np.linalg.norm(v, axis=1)
    
    # Compute the kinetic energy over m1
    T = 0.5 * spd * spd
    
    # Compute the potential energy over m1
    mu = (2.0 * np.pi)**2
    U = -mu / r
    
    # The total energy
    E = T + U
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Energy')
    ax.set_xlabel('t - years')
    ax.set_ylabel('Energy / m1 in $(au/year)^2$')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, T, color='b', label='T')
    ax.plot(t, U, color='r', label='U')
    ax.plot(t, E, color='purple', label='E')
    ax.grid()
    ax.legend()

    return fig, ax

