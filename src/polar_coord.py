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
    """Make data arrays for mapping between y and sin(theta)"""
    # Array of angles theta
    theta = np.linspace(-np.pi/2.0, np.pi/2.0, n+1)
    
    # The sin of these angles
    y = np.sin(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_dataset_sin(n, batch_size=None):
    """Make datasets for mapping between y and sin(theta)"""
    # Default batch_size is n
    if batch_size is None:
        batch_size = n+1

    # Make data dictionaries
    data = make_data_sin(n)
    
    # Unpack arrays
    y = data['y']
    theta = data['theta']

    # Create DataSet object for polar to cartesian: y = sin(theta)
    ds_p2c = tf.data.Dataset.from_tensor_slices((theta, y))
    
    # Create DataSet object for cartesian to polar: theta = arcsin(y)
    ds_c2p = tf.data.Dataset.from_tensor_slices((y, theta))
    
    # Create DataSet object for polar to polarautoencoder
    ds_p2p = tf.data.Dataset.from_tensor_slices((theta, theta))

    # Create DataSet object for cartesian to cartesian autoencoder
    d2_c2c = tf.data.Dataset.from_tensor_slices((y, y))

    # Set shuffle buffer size
    # buffer_size = batch_size * 256

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
