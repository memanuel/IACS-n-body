"""
Harvard IACS Masters Thesis
General Two Body Problem
Plot training data (trajectories)

Michael S. Emanuel
Mon Aug 05 11:14:00 2019
"""

# Library imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def plot_orbit_q(data, p_num=0):
    """Plot the orbit position in a training sample"""
    # Unpack data
    t = data['t']
    q1 = data['q1']
    q2 = data['q2']

    # Selected particle to plot
    if p_num == 1:
        # Plot particle 1
        q = q1
    elif p_num == 2:
        # Plot particle 2
        q = q2
    else:
        # Plot relative displacement
        q = q2 - q1
    
    # Components
    qx = q[:, 0]
    qy = q[:, 1]
    # Compute the distance r
    r = np.linalg.norm(q, axis=1)

    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    suffix = f'Body {p_num}' if p_num in (1, 2) else 'Relative'
    ax.set_title(f'Orbit Position - {suffix}')
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
def plot_orbit_v(data, p_num=0):
    """Plot the orbit velocity in a training sample"""
    # Unpack data
    t = data['t']
    v1 = data['v1']
    v2 = data['v2']

    # Selected particle to plot
    if p_num == 1:
        # Plot particle 1
        v = v1
    elif p_num == 2:
        # Plot particle 2
        v = v2
    else:
        # Plot relative displacement
        v = v2 - v1

    # Components and speed
    vx = v[:, 0]
    vy = v[:, 1]
    spd = np.linalg.norm(v, axis=1)
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    suffix = f'Body {p_num}' if p_num in (1, 2) else 'Relative'
    ax.set_title(f'Orbit Velocity - {suffix}')
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
def plot_orbit_a(data, p_num=0):
    """Plot the orbit acceleration in a training sample"""
    # Unpack data
    t = data['t']
    a1 = data['a1']
    a2 = data['a2']

    # Selected particle to plot
    if p_num == 1:
        # Plot particle 1
        a = a1
    elif p_num == 2:
        # Plot particle 2
        a = a2
    else:
        # Plot relative displacement
        a = a2 - a1

    # Components and magnitude
    ax = a[:, 0]
    ay = a[:, 1]
    acc = np.linalg.norm(a, axis=1)
    
    # Plot the x and y coordinate
    # Name the axes object ax_ rather than ax to avoid a name collision with the x component of acceleration, ax
    fig, ax_ = plt.subplots(figsize=[16, 9])
    suffix = f'Body {p_num}' if p_num in (1, 2) else 'Relative'
    ax_.set_title(f'Orbit Acceleration - {suffix}')
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
    T = data['T']
    U = data['U']
    H = data['H']
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title('Orbit Energy')
    ax.set_xlabel('t - years')
    ax.set_ylabel('Energy in $M_{sun}(au/year)^2$')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, T, color='b', label='T')
    ax.plot(t, U, color='r', label='U')
    ax.plot(t, H, color='purple', label='H')
    ax.grid()
    ax.legend()

    return fig, ax

