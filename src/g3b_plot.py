"""
Harvard IACS Masters Thesis
General Three Body Problem
Plot training data (trajectories)

Michael S. Emanuel
Thu Aug 08 11:24:00 2019
"""

# Library imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def plot_orbit_q(data, p_num: int, is_rel: bool = True, t_max=None):
    """Plot the orbit position in a training sample"""
    # Unpack data
    t = data['t']
    q = data['q']

    # Indivividual particles
    q0 = q[:,0]
    q1 = q[:,1]
    q2 = q[:,2]
    
    # Are we plotting absolute coordinates?
    is_abs = not is_rel
    
    # Determine t_max of not specified
    if t_max is None:
        t_max = np.max(t)

    # Selected particle to plot
    if p_num == 1:
        # Plot particle 1
        q_plot = q1 if is_abs else q1 - q0
    elif p_num == 2:
        # Plot particle 2
        q_plot = q2 if is_abs else q2 - q0
    elif p_num == 0:
        # Plot particle 0 (usually not very interesting...)
        q_plot = q0
    else:
        # Plot relative displacement
        raise ValueError('p_num must be an integer in [0, 1, 2].')

    # Components
    qx = q_plot[:, 0]
    qy = q_plot[:, 1]
    qz = q_plot[:, 2]
    # Compute the distance r
    r = np.linalg.norm(q_plot, axis=1)

    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    suffix = f'Body {p_num} ' + '(absolute)' if is_abs else '(relative to primary)'
    ax.set_title(f'Orbit Position - {suffix}')
    ax.set_xlabel('t - years')
    ax.set_ylabel('q - AU')
    ax.set_xlim([0.0, t_max])
    ax.set_xticks(np.arange(0.0, t_max, 1.0))
    ax.plot(t, qx, color='b', label='qx')
    ax.plot(t, qy, color='r', label='qy')
    ax.plot(t, qz, color='g', label='qz')
    ax.plot(t, r,  color='purple', label='r')
    ax.grid()
    ax.legend()
    
    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_v(data, p_num: int, is_rel: bool = True):
    """Plot the orbit velocity in a training sample"""
    # Unpack data
    t = data['t']
    v = data['v']

    # Indivividual particles
    v0 = v[:,0]
    v1 = v[:,1]
    v2 = v[:,2]
    
    # Are we plotting absolute coordinates?
    is_abs = not is_rel

    # Selected particle to plot
    if p_num == 1:
        # Plot particle 1
        v_plot = v1 if is_abs else v1 - v0
    elif p_num == 2:
        # Plot particle 2
        v_plot = v2 if is_abs else v2 - v0
    elif p_num == 0:
        # Plot particle 0 (usually not very interesting...)
        v_plot = v0
    else:
        # Plot relative displacement
        raise ValueError('p_num must be an integer in [0, 1, 2].')

    # Components
    vx = v_plot[:, 0]
    vy = v_plot[:, 1]
    vz = v_plot[:, 2]
    # Compute the speed
    spd = np.linalg.norm(v_plot, axis=1)

    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    suffix = f'Body {p_num} ' + '(absolute)' if is_abs else '(relative to primary)'
    ax.set_title(f'Orbit Velocity - {suffix}')
    ax.set_xlabel('t - years')
    ax.set_ylabel('v - AU/yr')
    ax.set_xticks(np.arange(0.0, np.max(t)+0.25, 0.25))
    ax.plot(t, vx, color='b', label='vx')
    ax.plot(t, vy, color='r', label='vy')
    ax.plot(t, vz, color='g', label='vz')
    ax.plot(t, spd,  color='purple', label='speed')
    ax.grid()
    ax.legend()
    
    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_a(data, p_num: int, is_rel: bool = True):
    """Plot the orbit acceleration in a training sample"""
    # Unpack data
    t = data['t']
    a = data['a']

    # Indivividual particles
    a0 = a[:,0]
    a1 = a[:,1]
    a2 = a[:,2]
    
    # Are we plotting absolute coordinates?
    is_abs = not is_rel

    # Selected particle to plot
    if p_num == 1:
        # Plot particle 1
        a_plot = a1 if is_abs else a1 - a0
    elif p_num == 2:
        # Plot particle 2
        a_plot = a2 if is_abs else a2 - a0
    elif p_num == 0:
        # Plot particle 0 (usually not very interesting...)
        a_plot = a0
    else:
        # Plot relative displacement
        raise ValueError('p_num must be an integer in [0, 1, 2].')

    # Components
    ax = a_plot[:, 0]
    ay = a_plot[:, 1]
    az = a_plot[:, 2]
    # Compute the speed
    acc = np.linalg.norm(a_plot, axis=1)


    
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
    ax_.plot(t, az, color='g', label='az')
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
    ax.set_xlim([0, 100])
    ax.set_xticks(np.arange(0, 101, 5))
    ax.plot(t, T, color='b', label='T')
    ax.plot(t, U, color='r', label='U')
    ax.plot(t, H, color='purple', label='H')
    ax.grid()
    ax.legend()

    return fig, ax

# ********************************************************************************************************************* 
def plot_orbit_element(data, element):
    """Plot the selected orbital element in a training sample"""
    # Unpack data
    key = f'orb_{element}'
    t = data['t']
    x = data[key]
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    # Plot the x and y coordinate
    fig, ax = plt.subplots(figsize=[16, 9])
    ax.set_title(f'Orbital Element ${element}$ Over Time')
    ax.set_xlabel('t - years')
    ax.set_ylabel(f'Orbital Element ${element}$')
    ax.set_xlim([0, 100])
    ax.set_xticks(np.arange(0, 101, 5))
    ax.plot(t, x1, color='b', label=f'{element}1')
    ax.plot(t, x2, color='r', label=f'{element}2')
    ax.grid()
    ax.legend()

    return fig, ax
