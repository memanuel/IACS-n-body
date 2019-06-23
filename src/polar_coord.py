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

# Local imports
from utils import EpochLoss, TimeHistory

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

    # Set shuffle buffer size equal to size of data set
    buffer_size = n+1

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.shuffle(buffer_size).batch(batch_size)
    ds_c2p = ds_c2p.shuffle(buffer_size).batch(batch_size)
    ds_p2p = ds_p2p.shuffle(buffer_size).batch(batch_size)
    ds_c2c = d2_c2c.shuffle(buffer_size).batch(batch_size)
    
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
    # buffer_size = n+1

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
def compile_and_fit(model, ds, epochs, loss, optimizer, metrics, save_freq):
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_name = model.name
    filepath=f'../models/polar_{model_name}.h5'

    # Create callbacks
    interval = epochs // 20
    cb_log = EpochLoss(interval=interval)
    cb_time = TimeHistory()
    cb_ckp = keras.callbacks.ModelCheckpoint(
            filepath=filepath, 
            save_freq=save_freq,
            save_best_only=True,
            save_weights_only=True,
            monitor='loss',
            verbose=0)    
    cb_early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=1.0E-8,
            patience=1000,
            verbose=1,
            restore_best_weights=True)    
    callbacks = [cb_log, cb_time, cb_ckp, cb_early_stop]
    
    # Fit the model
    hist = model.fit(ds, epochs=epochs, callbacks=callbacks, verbose=0)
    # Add the times to the history
    hist.history['time'] = cb_time.times
    
    # Restore the model to the best weights
    model.load_weights(filepath)

    return hist

# ********************************************************************************************************************* 
def make_model_odd(func_name, input_name, output_name, hidden_sizes):
    """
    Neural net model of odd functions, e.g. y = sin(theta)
    Example call: model_sin_16_16 = make_model_odd('sin', [16, 16])
    """
    # Input layer
    x = keras.Input(shape=(1,), name=input_name)

    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # Feature augmentation; odd powers up to 7
    x3 = keras.layers.Lambda(lambda x: tf.pow(x, 3), name='x3')(x)
    x5 = keras.layers.Lambda(lambda x: tf.pow(x, 5), name='x5')(x)
    x7 = keras.layers.Lambda(lambda x: tf.pow(x, 7), name='x7')(x)
    
    # Augmented feature layer
    phi_0 = keras.layers.concatenate(inputs=[x, x3, x5, x7], name='phi_0')
    
    # Dense feature layers

    # First hidden layer
    phi_1 = keras.layers.Dense(units=hidden_sizes[0], activation='tanh', name='phi_1')(phi_0)
    phi_n = phi_1

    # Second hidden layer if applicable
    if num_layers > 1:
        phi_2 = keras.layers.Dense(units=hidden_sizes[1], activation='tanh', name='phi_2')(phi_1)
        phi_n = phi_2

    # Output layer
    y = keras.layers.Dense(units=1, name=output_name)(phi_n)

    # Wrap into a model
    model_name = f'model_{func_name}_' + str(hidden_sizes)
    model = keras.Model(inputs=x, outputs=y, name=model_name) 
    return model

# ********************************************************************************************************************* 
def make_model_even(func_name, input_name, output_name, hidden_sizes):
    """
    Neural net model of even functions, e.g. y = cos(theta)
    Example call: model_cos_16_16 = make_model_even('cos', [16, 16])
    """
    # Input layer
    x = keras.Input(shape=(1,), name=input_name)

    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # Feature augmentation; even powers up to 8
    x2 = keras.layers.Lambda(lambda x: tf.pow(x, 2), name='x2')(x)
    x4 = keras.layers.Lambda(lambda x: tf.pow(x, 4), name='x4')(x)
    x6 = keras.layers.Lambda(lambda x: tf.pow(x, 6), name='x6')(x)
    x8 = keras.layers.Lambda(lambda x: tf.pow(x, 8), name='x8')(x)
    
    # Augmented feature layer
    phi_0 = keras.layers.concatenate(inputs=[x2, x4, x6, x8], name='phi_0')

    # Dense feature layers

    # First hidden layer
    phi_1 = keras.layers.Dense(units=hidden_sizes[0], activation='tanh', name='phi_1')(phi_0)
    phi_n = phi_1

    # Second hidden layer if applicable
    if num_layers > 1:
        phi_2 = keras.layers.Dense(units=hidden_sizes[1], activation='tanh', name='phi_2')(phi_1)
        phi_n = phi_2

    # Output layer
    y = keras.layers.Dense(units=1, name=output_name)(phi_n)

    # Wrap into a model
    model_name = f'model_{func_name}_' + str(hidden_sizes)
    model = keras.Model(inputs=x, outputs=y, name=model_name) 
    return model

# ********************************************************************************************************************* 
def make_model_autoencoder(model_p2c, model_c2p):
    """Generate two autoencoder models"""
    # autoencoder from p to p
    theta_in = keras.Input(shape=(1,), name='theta')
    y = model_p2c(theta_in)
    theta_out = model_c2p(y)
    model_p2p = keras.Model(theta_in, theta_out)
    
    # autoencoder from c to c
    y_in = keras.Input(shape=(1,), name='y')
    theta = model_c2p(y_in)
    y_out = model_p2c(theta)
    model_c2c = keras.Model(y_in, y_out)

    return model_p2p, model_c2c