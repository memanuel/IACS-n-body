"""
Harvard IACS Masters Thesis
Polar coordinate transformations

Michael S. Emanuel
Thu Jun 20 12:47:41 2019
"""

# Library imports
import tensorflow as tf
import numpy as np

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
def make_data_sin(n):
    """Make data arrays for mapping between theta and y = sin(theta)"""
    # Array of angles theta
    theta = np.linspace(-np.pi/2.0, np.pi/2.0, n+1)
    
    # The sin of these angles
    y = np.sin(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_data_cos(n):
    """Make data arrays for mapping between theta and x = cos(theta)"""
    # Array of angles theta
    theta = np.linspace(0.0, np.pi0, n+1)
    
    # The cos of these angles
    x = np.cos(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'x': x}
    
    return data

# ********************************************************************************************************************* 
def make_data_circle(n):
    """Make data arrays for mapping between theta and (x,y) on unit circle"""
    # Array of angles theta
    theta = np.linspace(-np.pi, np.pi0, n+1)
    
    # The cos and sin of these angles
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'x': x,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_data_polar(n_r, n_theta, r_min=0.5, r_max=32.0):
    """
    Make data arrays for mapping between (r, theta) and (x,y)
    Draw samples on a grid in polar coordinate space (r, theta) and transform to (x, y)
    """
    # Increment n_theta and n_r
    n_theta += 1
    n_r += 1

    # Array of radius r to sample
    r_ = np.linspace(r_min, r_max, n_r)
    # Array of angle theta to sample
    theta_ = np.linspace(-np.pi, np.pi, n_theta)
    
    # Repeat r_ to generate r
    r = np.repeat(r_, repeats=n_theta)
    # Tile theta_ to generate theta
    theta = np.tile(theta_, reps=n_r)

    # Compute x and y from r and theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Wrap data dict
    data = {'r': r,
            'theta': theta,
            'x': x,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_data_cart(n_a, a):
    """
    Make data arrays for mapping between (r, theta) and (x,y)
    Draw samples on a square grid and transform to angles
    """
    # Increment n_a
    n_a += 1

    # Array of x and y to sample
    x_ = np.linspace(-a, a, n_a)
    y_ = np.linspace(-a, a, n_a)

    # Repeat x_ to generate x
    x = np.repeat(x_, repeats=n_a)
    # Tile theta_ to generate theta
    y = np.tile(y_, reps=n_a)

    # Compute r and theta from x and y
    r = np.sqrt(x*x + y*y)
    theta = np.arctan2(y, x)

    # Wrap data dict
    data = {'r': r,
            'theta': theta,
            'x': x,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_dataset_sin(n, batch_size=None):
    """Make datasets for mapping between y and sin(theta)"""
    # Default batch_size is n
    if batch_size is None:
        batch_size = n+1

    # Make data dictionary
    data = make_data_sin(n)
    
    # Unpack arrays
    y = data['y']
    theta = data['theta']

    # Create DataSet object for polar to cartesian: y = sin(theta)
    ds_p2c = tf.data.Dataset.from_tensor_slices((theta, y))
    
    # Create DataSet object for cartesian to polar: theta = arcsin(y)
    ds_c2p = tf.data.Dataset.from_tensor_slices((y, theta))
    
    # Create DataSet object for polar to polar autoencoder
    ds_p2p = tf.data.Dataset.from_tensor_slices((theta, theta))

    # Create DataSet object for cartesian to cartesian autoencoder
    d2_c2c = tf.data.Dataset.from_tensor_slices((y, y))

    # Set shuffle buffer size
    # buffer_size = batch_size

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.batch(batch_size)
    ds_c2p = ds_c2p.batch(batch_size)
    ds_p2p = ds_p2p.batch(batch_size)
    ds_c2c = d2_c2c.batch(batch_size)
    
    return ds_p2c, ds_c2p, ds_p2p, ds_c2c

# ********************************************************************************************************************* 
def make_dataset_cos(n, batch_size=None):
    """Make datasets for mapping between x and cos(theta)"""
    # Default batch_size is n
    if batch_size is None:
        batch_size = n+1

    # Make data dictionary
    data = make_data_cos(n)
    
    # Unpack arrays
    x = data['x']
    theta = data['theta']

    # Create DataSet object for polar to cartesian: x = cos(theta)
    ds_p2c = tf.data.Dataset.from_tensor_slices((theta, x))
    
    # Create DataSet object for cartesian to polar: theta = arccos(x)
    ds_c2p = tf.data.Dataset.from_tensor_slices((x, theta))
    
    # Create DataSet object for polar to polara utoencoder
    ds_p2p = tf.data.Dataset.from_tensor_slices((theta, theta))

    # Create DataSet object for cartesian to cartesian autoencoder
    d2_c2c = tf.data.Dataset.from_tensor_slices((x, x))

    # Set shuffle buffer size
    # buffer_size = batch_size

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.batch(batch_size)
    ds_c2p = ds_c2p.batch(batch_size)
    ds_p2p = ds_p2p.batch(batch_size)
    ds_c2c = d2_c2c.batch(batch_size)
    
    return ds_p2c, ds_c2p, ds_p2p, ds_c2c

# ********************************************************************************************************************* 
def make_dataset_circle(n, batch_size=None):
    """Make datasets for mapping between x and cos(theta)"""
    # Default batch_size is n
    if batch_size is None:
        batch_size = n+1

    # Make data dictionary
    data = make_data_circle(n)
    
    # Unpack arrays
    x = data['x']
    y = data['y']
    theta = data['theta']
    
    # Wrap the two cartesian inputs into a dict
    cart = {'x': x,
            'y': y}

    # Wrap the polar inputs into a dict
    polar = {'theta': theta,}

    # Create DataSet object for polar to cartesian
    ds_p2c = tf.data.Dataset.from_tensor_slices((polar, cart))
    
    # Create DataSet object for cartesian to polar
    ds_c2p = tf.data.Dataset.from_tensor_slices((cart, polar))
    
    # Create DataSet object for polar to polar autoencoder
    ds_p2p = tf.data.Dataset.from_tensor_slices((polar, polar))

    # Create DataSet object for cartesian to cartesian autoencoder
    d2_c2c = tf.data.Dataset.from_tensor_slices((cart, cart))

    # Set shuffle buffer size
    # buffer_size = batch_size

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.batch(batch_size)
    ds_c2p = ds_c2p.batch(batch_size)
    ds_p2p = ds_p2p.batch(batch_size)
    ds_c2c = d2_c2c.batch(batch_size)
    
    return ds_p2c, ds_c2p, ds_p2p, ds_c2c

# ********************************************************************************************************************* 
def make_model_sin_math():
    """Mathematical models transforming between y and sin(theta)"""
    # Input layers
    theta = keras.Input(shape=(1,), name='theta')
    y = keras.Input(shape=(1,), name='y')
    
    # Layers to compute sin and arcsin
    sin = keras.layers.Activation(tf.math.sin, name='sin_theta')
    arcsin = keras.layers.Activation(tf.math.asin, name='arcsin_y')
    
    # Compute sin(theta) and arcsin(y)
    sin_theta = sin(theta)
    arcsin_y = arcsin(y)
    
    # Compute the recovered values of y and theta for auto-encoder
    theta_rec = arcsin(sin_theta)
    y_rec = sin(arcsin_y)
    
    # Models for p2c and c2p
    model_p2c = keras.Model(inputs=theta, outputs=sin_theta)
    model_c2p = keras.Model(inputs=y, outputs=arcsin_y)
    
    # Models for autoencoders p2p and c2c
    model_p2p = keras.Model(inputs=theta, outputs=theta_rec)
    model_c2c = keras.Model(inputs=y, outputs=y_rec)
    
    return model_p2c, model_c2p, model_p2p, model_c2c

# ********************************************************************************************************************* 
def make_model_sin(hidden_sizes):
    """Neural net model of y = sin(theta)"""
    # Input layer
    theta = keras.Input(shape=(1,), name='theta')

    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # Dense feature layers

    # First hidden layer
    phi_1 = keras.layers.Dense(units=hidden_sizes[0], activation='tanh', name='phi_1')(theta)

    # Second hidden layer if applicable
    phi_2 = None
    if num_layers > 1:
        phi_2 = keras.layers.Dense(units=hidden_sizes[1], activation='tanh', name='phi_2')(phi_1)

    # Lookup table for last feature layer
    phi_tbl = {1: phi_1,
               2: phi_2}

    # The last feature layer
    phi_n = phi_tbl[num_layers]

    # Output layer
    y = keras.layers.Dense(units=1, name='y')(phi_n)

    # Wrap into a model
    model = keras.Model(inputs=theta, outputs=y, name='model_sin') 
    return model