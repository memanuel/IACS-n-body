"""
Harvard IACS Masters Thesis
Tensorflow Utilites

Michael S. Emanuel
Wed Jun 26 20:24:42 2019
"""

import tensorflow as tf
keras = tf.keras
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

# *************************************************************************************************
def gpu_grow_memory():
    pass
    """Set TensorFlow to grow memory of GPUs rather than grabbing it all at once."""
    # Get available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Set all selected GPUs to 
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ********************************************************************************************************************* 
def plot_loss_hist(hist,  model_name, key='loss', baseline=None):
    """Plot loss vs. wall time"""
    # Extract loss and wall time arrays
    loss = hist[key]
    time = hist['time']
    
    # Plot loss vs. wall time
    fig, ax = plt.subplots(figsize=[16,9])
    ax.set_title(f'Loss vs. Wall Time for {model_name}')
    ax.set_xlabel('Wall Time (Seconds)')
    ax.set_ylabel(f'{key}')
    ax.plot(time, loss, color='blue', label=key)
    if baseline is not None:
        ax.axhline(baseline, color='red', label='baseline')
    ax.set_yscale('log')
    ax.grid()
    ax.legend()

    return fig, ax

# *************************************************************************************************
# Custom Callbacks
# *************************************************************************************************

# *************************************************************************************************
# https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit 
class TimeHistory(keras.callbacks.Callback):
    """Save the wall time after every epoch"""
    def on_train_begin(self, logs={}):
        self.times = []
        self.train_time_start = time.time()

    # def on_epoch_begin(self, batch, logs={}):
    #    self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.train_time_start)

# *************************************************************************************************
class EpochLoss(tf.keras.callbacks.Callback):
    """Log the loss every N epochs"""
    def __init__(self, interval=10, newline=False):
        super(EpochLoss, self).__init__()
        self.interval = interval
        self.train_time_start = time.time()
        self.log_prefix = '\n' if newline else ''

    def log_to_screen(self, epoch, logs):
        loss = logs['loss']
        elapsed = time.time() - self.train_time_start
        elapsed_str = str(datetime.timedelta(seconds=np.round(elapsed)))
        print(f'{self.log_prefix}Epoch {epoch:04}; loss {loss:5.2e}; elapsed {elapsed_str}') 

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch+1
        if (epoch % self.interval == 0) or (epoch == 1):
            self.log_to_screen(epoch, logs)
            
# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class Identity(keras.layers.Activation):
    """Identity layer for labeling outputs in models"""
    def __init__(self, **kwargs):
        super(Identity, self).__init__(activation = tf.identity, **kwargs)
        
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
class Divide(keras.layers.Layer):
    """Division"""
    def call(self, inputs):
        # Unpack inputs
        x, y = inputs
        
        # Return quotient x / y
        return tf.divide(x, y)
    
    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
def make_features_pow(x, powers, input_name, output_name):
    """
    Make features with powers of an input feature
    INPUTS:
        x: the original feature
        powers: list of integer powers, e.g. [1,3,5,7]        
        input_name: the name of the input feature, e.g. 'x' or 'theta'
        output_name: the name of the output feature layer, e.g. 'phi_0'
    """
    # List with layers x**p / p!
    xps = []
    # Iterate over the specified powers
    for p in powers:
        xp = keras.layers.Lambda(lambda x: tf.pow(x, p) / tf.exp(tf.math.lgamma(p+1.0)), name=f'{input_name}_{p}')(x)
        xps.append(xp)
    
    # Augmented feature layer
    return keras.layers.concatenate(inputs=xps, name=output_name)

# ********************************************************************************************************************* 
# Functional Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_pow(func_name, input_name, output_name, powers, hidden_sizes, skip_layers):
    """
    Neural net model of functions using powers of x as features
    INPUTS:
        func_name: name of the function being fit, e.g. 'cos'
        input_name: name of the input layer, e.g. 'theta'
        output_name: name of the output layer, e.g. 'x'
        powers: list of integer powers of the input in feature augmentation
        hidden_sizes: sizes of up to 2 hidden layers
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

    # Augmented feature layer - selected powers of the input
    phi_0 = make_features_pow(x=x, powers=powers, input_name=input_name, output_name='phi_0')
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
