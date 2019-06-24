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
    theta = np.linspace(-np.pi/2.0, np.pi/2.0, n+1, dtype=np.float32)
    
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
    theta = np.linspace(0.0, np.pi, n+1, dtype=np.float32)
    
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
    theta = np.linspace(-np.pi, np.pi0, n+1, dtype=np.float32)
    
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
    r_ = np.linspace(r_min, r_max, n_r, dtype=np.float32)
    # Array of angle theta to sample
    theta_ = np.linspace(-np.pi, np.pi, n_theta, dtype=np.float32)
    
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
    x_ = np.linspace(-a, a, n_a, dtype=np.float32)
    y_ = np.linspace(-a, a, n_a, dtype=np.float32)

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

    # Default batch_size is all the data
    if batch_size is None:
        batch_size = theta.shape[0]

    # Set shuffle buffer size equal to size of data set
    buffer_size = theta.shape[0]

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.shuffle(buffer_size).batch(batch_size)
    ds_c2p = ds_c2p.shuffle(buffer_size).batch(batch_size)
    ds_p2p = ds_p2p.shuffle(buffer_size).batch(batch_size)
    ds_c2c = d2_c2c.shuffle(buffer_size).batch(batch_size)
    
    return ds_p2c, ds_c2p, ds_p2p, ds_c2c

# ********************************************************************************************************************* 
def make_dataset_cos(n, batch_size=None):
    """Make datasets for mapping between x and cos(theta)"""
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

    # Default batch_size is all the data
    if batch_size is None:
        batch_size = theta.shape[0]

    # Set shuffle buffer size equal to size of data set
    buffer_size = theta.shape[0]

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.shuffle(buffer_size).batch(batch_size)
    ds_c2p = ds_c2p.shuffle(buffer_size).batch(batch_size)
    ds_p2p = ds_p2p.shuffle(buffer_size).batch(batch_size)
    ds_c2c = d2_c2c.shuffle(buffer_size).batch(batch_size)
    
    return ds_p2c, ds_c2p, ds_p2p, ds_c2c

# ********************************************************************************************************************* 
def make_dataset_circle(n, batch_size=None):
    """Make datasets for mapping between x and cos(theta)"""
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

    # Default batch_size is all the data
    if batch_size is None:
        batch_size = theta.shape[0]

    # Set shuffle buffer size
    buffer_size = theta.shape[0]

    # Shuffle and batch data sets
    ds_p2c = ds_p2c.shuffle(buffer_size).batch(batch_size)
    ds_c2p = ds_c2p.shuffle(buffer_size).batch(batch_size)
    ds_p2p = ds_p2p.shuffle(buffer_size).batch(batch_size)
    ds_c2c = d2_c2c.shuffle(buffer_size).batch(batch_size)
       
    return ds_p2c, ds_c2p, ds_p2p, ds_c2c

# ********************************************************************************************************************* 
def make_model_sin_math():
    """Mathematical model transforming between y and sin(theta)"""
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
def make_model_cos_math():
    """Mathematical model transforming between x and cos(theta)"""
    # Input layers
    theta = keras.Input(shape=(1,), name='theta')
    x = keras.Input(shape=(1,), name='x')
    
    # Layers to compute cos and arccos
    cos = keras.layers.Activation(tf.math.cos, name='cos_theta')
    arccos = keras.layers.Activation(tf.math.acos, name='arccos_x')
    
    # Compute sin(theta) and arcsin(y)
    cos_theta = cos(theta)
    arccos_x = arccos(x)
    
    # Compute the recovered values of y and theta for auto-encoder
    theta_rec = arccos(cos_theta)
    x_rec = cos(arccos_x)
    
    # Models for p2c and c2p
    model_p2c = keras.Model(inputs=theta, outputs=cos_theta)
    model_c2p = keras.Model(inputs=x, outputs=arccos_x)
    
    # Models for autoencoders p2p and c2c
    model_p2p = keras.Model(inputs=theta, outputs=theta_rec)
    model_c2c = keras.Model(inputs=x, outputs=x_rec)
    
    return model_p2c, model_c2p, model_p2p, model_c2c

# ********************************************************************************************************************* 
def make_model_circle_math():
    """Mathematical model transforming between theta and (x, y) on unit circle"""
    # Input layers
    theta_in = keras.Input(shape=(1,), name='theta')
    x_in = keras.Input(shape=(1,), name='x')
    y_in = keras.Input(shape=(1,), name='y')

    # Layers to compute sin and cos
    cos = keras.layers.Activation(tf.math.cos, name='cos_theta')
    sin = keras.layers.Activation(tf.math.sin, name='sin_theta')

    # Layer to compute atan2
    atan2 = keras.layers.Lambda(lambda x, y: tf.math.atan2(y, x), name='atan2')
    
    # Compute sin(theta) and cos(theta)
    x_out = cos(theta_in)
    y_out = sin(theta_in)
    
    # Compute atan2(y, x)
    theta_out = atan2([x_in, y_in])

    # Compute the recovered values of theta, x, y for auto-encoder
    theta_rec = atan2([x_out, y_out])
    x_rec = cos(theta_out)
    y_rec = sin(theta_out)
    
    # Models for p2c and c2p
    model_p2c = keras.Model(inputs=theta_in, outputs=[x_out, y_out])
    model_c2p = keras.Model(inputs=[x_in, y_in], outputs=theta_out)

    # Models for autoencoders p2p and c2c
    model_p2p = keras.Model(inputs=theta_in, outputs=theta_rec)
    model_c2c = keras.Model(inputs=[x_in, x_out], outputs=[x_rec, y_rec])
    
    return model_p2c, model_c2p, model_p2p, model_c2c

# ********************************************************************************************************************* 
def compile_and_fit(model, ds, epochs, loss, optimizer, metrics, save_freq):
    # Compile the model
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model_name = model.name
    filepath=f'../models/polar/{model_name}.h5'

    # Create callbacks
    interval = epochs // 20
    patience = epochs // 10
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
            patience=patience,
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
def make_model_pow(func_name, input_name, output_name, powers, hidden_sizes, skip_layers):
    """
    Neural net model of functions using powers of x as features
    INPUTS:
        func_name: name of the function being fit, e.g. 'cos'
        input_name: name of the input layer, e.g. 'theta'
        output_name: name of the output layer, e.g. 'x'
        powers: list of integer powers of the input in feature augmentation
        hidden_sizes: sizes of up to 3 hidden layers
        skip_layers: whether to include skip layers (copy of previous features)
    Example call: 
        model_cos_16_16 = make_model_even(
            func_name='cos',
            input_name='theta',
            output_name='x',
            powers=[2,4,6,8],
            hidden_sizes=[16, 16])
    """
    # Input layer
    x = keras.Input(shape=(1,), name=input_name)

    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # Feature augmentation; the selected powers
    xps = []
    for p in powers:
        xp = keras.layers.Lambda(lambda x: tf.pow(x, p) / tf.exp(tf.math.lgamma(p+1.0)), name=f'x{p}')(x)
        xps.append(xp)
    
    # Augmented feature layer
    phi_0 = keras.layers.concatenate(inputs=xps, name='phi_0')
    phi_n = phi_0

    # Dense feature layers
    
    # First hidden layer if applicable
    if num_layers > 0:
        phi_1 = keras.layers.Dense(units=hidden_sizes[0], activation='tanh', name='phi_1')(phi_0)
        if skip_layers:
            phi_1 = keras.layers.concatenate(inputs=[phi_0, phi_1], name='phi_1_aug')
        phi_n = phi_1

    # Second hidden layer if applicable
    if num_layers > 1:
        phi_2 = keras.layers.Dense(units=hidden_sizes[1], activation='tanh', name='phi_2')(phi_1)
        if skip_layers:
            phi_2 = keras.layers.concatenate(inputs=[phi_1, phi_2], name='phi_2_aug')
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