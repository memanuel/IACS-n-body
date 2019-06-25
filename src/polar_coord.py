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
    # Half-interval width
    dt = np.pi / (2*n)
    # Array of angles theta
    theta = np.linspace(-np.pi/2.0+dt, np.pi/2.0-dt, n, dtype=np.float32)

    # The sin of these angles
    y = np.sin(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_data_cos(n):
    """Make data arrays for mapping between theta and x = cos(theta)"""
    # Half-interval width
    dt = np.pi / (2*n)
    # Array of angles theta
    theta = np.linspace(0.0+dt, np.pi-dt, n, dtype=np.float32)
    
    # The cos of these angles
    x = np.cos(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'x': x}
    
    return data

# ********************************************************************************************************************* 
def make_data_tan(n):
    """Make data arrays for mapping between theta and x = cos(theta)"""
    # Half-interval width
    dt = np.pi / (2*n)
    # Array of angles theta
    theta = np.linspace(-np.pi/2.0+dt, np.pi/2.0-dt, n, dtype=np.float32)
    
    # The tan of these angles
    z = np.tan(theta)
    
    # Wrap data dict
    data = {'theta': theta,
            'z': z}
    
    return data

# ********************************************************************************************************************* 
def make_data_circle(n):
    """Make data arrays for mapping between theta and (x,y) on unit circle"""
    # Half-interval width
    dx = np.pi / n
    # Array of angles theta
    theta = np.linspace(-np.pi+dx, np.pi-dx, n, dtype=np.float32)
    
    # The cos and sin of these angles
    x = np.cos(theta)
    y = np.sin(theta)

    # Wrap data dict
    data = {'theta': theta,
            'x': x,
            'y': y}
    
    return data

# ********************************************************************************************************************* 
def make_data_polar_radial(n_r, n_theta, r_min=0.5, r_max=32.0):
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
def make_data_polar_grid(n_a, a):
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
    theta = data['theta']
    x = data['x']

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
def make_dataset_tan(n, batch_size=None):
    """Make datasets for mapping between x and cos(theta)"""
    # Make data dictionary
    data = make_data_tan(n)
    
    # Unpack arrays
    theta = data['theta']
    z = data['z']

    # Create DataSet object for polar to cartesian: z = tan(theta)
    ds_p2c = tf.data.Dataset.from_tensor_slices((theta, z))
    
    # Create DataSet object for cartesian to polar: theta = arctan(z)
    ds_c2p = tf.data.Dataset.from_tensor_slices((z, theta))
    
    # Create DataSet object for polar to polar autoencoder
    ds_p2p = tf.data.Dataset.from_tensor_slices((theta, theta))

    # Create DataSet object for cartesian to cartesian autoencoder
    d2_c2c = tf.data.Dataset.from_tensor_slices((z, z))

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
    theta = data['theta']
    x = data['x']
    y = data['y']
    
    # Wrap the polar inputs into a dict
    polar = {'theta': theta,}
    polar_in = {'theta_in': theta,}
    polar_out = {'theta_out': theta,}

    # Wrap the two cartesian inputs into a dict
    cart = {'x': x,
            'y': y}
    cart_in = {'x_in': x,
               'y_in': y}
    cart_out = {'x_out': x,
                'y_out': y}

    # Create DataSet object for polar to cartesian
    ds_p2c = tf.data.Dataset.from_tensor_slices((polar, cart))
    
    # Create DataSet object for cartesian to polar
    ds_c2p = tf.data.Dataset.from_tensor_slices((cart, polar))
    
    # Create DataSet object for polar to polar autoencoder
    ds_p2p = tf.data.Dataset.from_tensor_slices((polar_in, polar_out))

    # Create DataSet object for cartesian to cartesian autoencoder
    d2_c2c = tf.data.Dataset.from_tensor_slices((cart_in, cart_out))

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
def make_models_sin_math():
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
def make_models_cos_math():
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
def make_models_tan_math():
    """Mathematical model transforming between z and tan(theta)"""
    # Input layers
    theta = keras.Input(shape=(1,), name='theta')
    z = keras.Input(shape=(1,), name='z')
    
    # Layers to compute cos and arccos
    tan = keras.layers.Activation(tf.math.tan, name='tan_theta')
    arctan = keras.layers.Activation(tf.math.atan, name='arctan_z')
    
    # Compute sin(theta) and arcsin(y)
    tan_theta = tan(theta)
    arctan_z = arctan(z)
    
    # Compute the recovered values of y and theta for auto-encoder
    theta_rec = arctan(tan_theta)
    z_rec = tan(arctan_z)
    
    # Models for p2c and c2p
    model_p2c = keras.Model(inputs=theta, outputs=tan_theta)
    model_c2p = keras.Model(inputs=z, outputs=arctan_z)
    
    # Models for autoencoders p2p and c2c
    model_p2p = keras.Model(inputs=theta, outputs=theta_rec)
    model_c2c = keras.Model(inputs=z, outputs=z_rec)
    
    return model_p2c, model_c2p, model_p2p, model_c2c

# ********************************************************************************************************************* 
def make_model_circle_math_p2c():
    """Mathematical model transforming between theta and (x, y) on unit circle"""
    # Input layer
    theta = keras.Input(shape=(1,), name='theta')

    # Compute sin(theta) and cos(theta)
    x = keras.layers.Activation(tf.math.cos, name='x')(theta)
    y = keras.layers.Activation(tf.math.sin, name='y')(theta)
    
    # Model for polar to cartesian
    model_p2c = keras.Model(inputs=theta, outputs=[x, y], name='math_p2c')

    return model_p2c

def make_model_circle_math_c2p():
    """Mathematical model transforming between theta and (x, y) on unit circle"""
    # Input layers
    x = keras.Input(shape=(1,), name='x')
    y = keras.Input(shape=(1,), name='y')

    # Compute atan2(y, x)
    atan_func = lambda q: tf.math.atan2(q[1], q[0])
    theta = keras.layers.Lambda(atan_func, name='theta')([x, y])

    # Model cartesian to polar
    model_c2p = keras.Model(inputs=[x, y], outputs=theta, name='math_c2p')

    return model_c2p

def make_model_circle_math_c2c():
    """Mathematical model transforming between theta and (x, y) on unit circle"""
    # Input layers
    x_in = keras.Input(shape=(1,), name='x_in')
    y_in = keras.Input(shape=(1,), name='y_in')

    # Models for p2c and c2p
    model_p2c = make_model_circle_math_p2c()
    model_c2p = make_model_circle_math_c2p()

    # Theta from (x_in, y_in)
    theta = model_c2p([x_in, y_in])
    # (x_out, y_out) from theta
    [x_out, y_out] = model_p2c(theta)

    # Rename output layers using identity layers; otherwise can't pass dataset with named variables
    x_out = keras.layers.Activation(tf.identity, name='x_out')(x_out)
    y_out = keras.layers.Activation(tf.identity, name='y_out')(y_out)

    # Model cartesian to cartesian
    model_c2c = keras.Model(inputs=[x_in, y_in], outputs=[x_out, y_out], name='math_c2c')

    return model_c2c

def make_model_circle_math_p2p():
    """Mathematical model transforming between theta and (x, y) on unit circle"""
    # Input layer
    theta_in = keras.Input(shape=(1,), name='theta_in')

    # Models for p2c and c2p
    model_p2c = make_model_circle_math_p2c()
    model_c2p = make_model_circle_math_c2p()

    # (x, y) from theta_in
    [x, y] = model_p2c(theta_in)
    # theta_out from (x, y)
    theta_out = model_c2p([x, y])

    # Rename output layer using identity layers; otherwise can't pass dataset with named variables
    theta_out = keras.layers.Activation(tf.identity, name='theta_out')(theta_out)

    # Models polar to polar
    model_p2p = keras.Model(inputs=theta_in, outputs=theta_out, name='math_p2p')
    
    return model_p2p

# ********************************************************************************************************************* 
def make_models_circle_math():
    """Mathematical model transforming between theta and (x, y) on unit circle"""
    model_p2c = make_model_circle_math_p2c()
    model_c2p = make_model_circle_math_c2p()
    model_c2c = make_model_circle_math_c2c()
    model_p2p = make_model_circle_math_p2p()
    
    return model_p2c, model_c2p, model_p2p, model_c2c

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
def make_model_arctan(hidden_sizes, skip_layers):
    """
    Neural net model of arctan function
    """
    # Input layer
    z = keras.Input(shape=(1,), name='z')
    
    # Compute transform z / (1+z^2)
    z_over_1pz2 = keras.layers.Lambda(lambda z : z / (1.0 + z * z), name='z_over_1pz2')(z)
    
    # Compute features in Euler power series expansion of arctan
    # https://en.wikipedia.org/wiki/Inverse_trigonometric_functions (search for Euler)
    f1 = z_over_1pz2
    a0 = f1
    
    f2 = keras.layers.Lambda(lambda f: tf.pow(f, 2), name='f2')(z_over_1pz2)
    a1 = keras.layers.multiply(inputs=[f2, z], name='z3_over_1pz2_2')
    
    f3 = keras.layers.Lambda(lambda f: tf.pow(f, 3), name='f3')(z_over_1pz2)
    z2 = keras.layers.Lambda(lambda z: tf.pow(z, 2), name='z2')(z)
    a2 = keras.layers.multiply(inputs=[f3, z2], name='z5_over_1pz2_3')

    f4 = keras.layers.Lambda(lambda f: tf.pow(f, 4), name='f4')(z_over_1pz2)
    z3 = keras.layers.Lambda(lambda z: tf.pow(z, 3), name='z3')(z)
    a3 = keras.layers.multiply(inputs=[f4, z3], name='z7_over_1pz2_3')

    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # Augmented feature layer - transforms of z^(2n+1) / (1+z^2)^(n+1)
    phi_0 = keras.layers.concatenate([a0, a1, a2, a3], name='phi_0')
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
    # y = keras.layers.Dense(units=1, kernel_initializer='zeros', name='theta')(phi_n)
    y = keras.layers.Dense(units=1, name='theta')(phi_n)

    # Wrap into a model
    model_name = f'model_arctan_' + str(hidden_sizes)
    model = keras.Model(inputs=z, outputs=y, name=model_name) 
    return model

# ********************************************************************************************************************* 
def make_model_circle_p2c(powers, hidden_sizes, skip_layers):
    """
    Neural net model from theta to (x, y)
    INPUTS:
        powers: list of integer powers of the input in feature augmentation
        hidden_sizes: sizes of up to 2 hidden layers
        skip_layers: whether to include skip layers (copy of previous features)
    """
    # Input layer
    theta_in = keras.Input(shape=(1,), name='theta_in')

    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # Feature augmentation; the selected powers
    theta_ps = []
    for p in powers:
        theta_p = keras.layers.Lambda(lambda z: tf.pow(z, p) / tf.exp(tf.math.lgamma(p+1.0)), name=f'theta_{p}')(theta_in)
        theta_ps.append(theta_p)
    
    # Augmented feature layer
    phi_0 = keras.layers.concatenate(inputs=theta_ps, name='phi_0')
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

    # Output layers
    x_out = keras.layers.Dense(units=1, name='x_out')(phi_n)
    y_out = keras.layers.Dense(units=1, name='y_out')(phi_n)

    # Wrap into a model
    model_name = f'model_circle_p2c_' + str(hidden_sizes)
    model = keras.Model(inputs=theta_in, outputs=[x_out, y_out], name=model_name) 
    return model