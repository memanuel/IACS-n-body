"""
Harvard IACS Masters Thesis
Utilites

Michael S. Emanuel
Tue Jun  4 15:24:22 2019
"""

import numpy as np
import tensorflow as tf
keras = tf.keras
import matplotlib as mpl
plt = mpl.pyplot
import pickle
import time
import datetime

from typing import Tuple, Dict, Callable

# Type aliases
funcType = Callable[[float], float]

# *************************************************************************************************
def plot_style() -> None:
    """Set plot style for the session."""
    # Set default font size to 20
    mpl.rcParams.update({'font.size': 20})

# *************************************************************************************************
def range_inc(x: int, y: int = None, z: int = None) -> range:
    """Return a range inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + 1, z)
    elif z < 0:
        (start, stop, step) = (x, y - 1, z)
    return range(start, stop, step)


def arange_inc(x: float, y: float = None, z: float = None) -> np.ndarray:
    """Return a numpy arange inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + z, z)
    elif z < 0:
        (start, stop, step) = (x, y - z, z)
    return np.arange(start, stop, step)

# *************************************************************************************************
# Serialize generic Python variables using Pickle
def load_vartbl(fname: str) -> Dict:
    """Load a dictionary of variables from a pickled file"""
    try:
        with open(fname, 'rb') as fh:
            vartbl = pickle.load(fh)
    except:
        vartbl = dict()
    return vartbl


def save_vartbl(vartbl: Dict, fname: str) -> None:
    """Save a dictionary of variables to the given file with pickle"""
    with open(fname, 'wb') as fh:
        pickle.dump(vartbl, fh)

# *************************************************************************************************
def gpu_grow_memory():
	"""Set TensorFlow to grow memory of GPUs rather than grabbing it all at once."""
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
        
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
    def __init__(self, interval=10):
        super(EpochLoss, self).__init__()
        self.interval = interval
        self.train_time_start = time.time()

    def log_to_screen(self, epoch, logs):
        loss = logs['loss']
        elapsed = time.time() - self.train_time_start
        elapsed_str = str(datetime.timedelta(seconds=np.round(elapsed)))
        print(f'Epoch {epoch:04}; loss {loss:5.2e}; elapsed {elapsed_str}') 
        
    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch+1
        if (epoch % self.interval == 0) or (epoch == 1):
            self.log_to_screen(epoch, logs)
            
# ********************************************************************************************************************* 
def plot_loss_hist(hist,  model_name):
    """Plot loss vs. wall time"""
    # Extract loss and wall time arrays
    loss = hist['loss']
    time = hist['time']
    
    # Plot loss vs. wall time
    fig, ax = plt.subplots(figsize=[16,9])
    ax.set_title(f'Loss vs. Wall Time for {model_name}')
    ax.set_xlabel('Wall Time (Seconds)')
    ax.set_ylabel('Loss')
    ax.plot(time, loss, color='blue')
    ax.set_yscale('log')    
    ax.grid()

    return fig, ax

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
    # List with layers x**p
    xps = []
    # Iterate over the specified powers
    for p in powers:
        xp = keras.layers.Lambda(lambda x: tf.pow(x, p) / tf.exp(tf.math.lgamma(p+1.0)), name=f'{input_name}_{p}')(x)
        xps.append(xp)
    
    # Augmented feature layer
    return keras.layers.concatenate(inputs=xps, name=output_name)

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
