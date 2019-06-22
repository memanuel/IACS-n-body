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
        if epoch % self.interval == 0:
            self.log_to_screen(epoch, logs)
            
# ********************************************************************************************************************* 
def plot_loss_hist(hist,  model_name):
    """Plot loss vs. wall time"""
    # Extract loss and wall time arrays
    loss = hist.history['loss']
    time = hist.history['time']
    
    # Plot loss vs. wall time
    fig, ax = plt.subplots(figsize=[16,9])
    ax.set_title(f'Loss vs. Wall Time for {model_name}')
    ax.set_xlabel('Wall Time (Seconds)')
    ax.set_ylabel('Loss')
    ax.plot(time, loss, color='blue')
    ax.set_yscale('log')    
    ax.grid()

    return fig, ax