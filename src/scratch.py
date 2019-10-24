"""
Michael S. Emanuel
Thu Oct 24 14:28:11 2019
"""

# Library imports
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np

# Local imports
from asteroid_integrate import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, random_direction
# from observation_data import make_synthetic_obs_tensors
from asteroid_data import orbital_element_batch
from asteroid_model import AsteroidPosition, make_model_ast_dir, make_model_ast_pos
from tf_utils import Identity

# Aliases
keras = tf.keras

# Constants
space_dims = 3

# ********************************************************************************************************************* 
class OrbitalElements(keras.layers.Layer):
    """Custom layer to maintain state of orbital elements."""
    def __init__(self, elts_np: dict, batch_size: int, **kwargs):
        super(OrbitalElements, self).__init__(**kwargs)
        elts_np = orbital_element_batch(1)

        self.a = tf.Variable(initial_value=elts_np['a'], trainable=True, name='a')
        self.e = tf.Variable(initial_value=elts_np['e'], trainable=True, name='e')
        self.inc = tf.Variable(initial_value=elts_np['inc'], trainable=True, name='inc')
        self.Omega = tf.Variable(initial_value=elts_np['Omega'], trainable=True, name='Omega')
        self.omega = tf.Variable(initial_value=elts_np['omega'], trainable=True, name='omega')
        self.f = tf.Variable(initial_value=elts_np['f'], trainable=True, name='f')

        # The epoch is not trainable
        self.epoch = tf.Variable(initial_value=elts_np['epoch'], trainable=False, name='epoch')

    def call(self, inputs):
        """Predict directions with the current orbital elements"""
        return self.a, self.e, self.inc, self.Omega, self.omega, self.f, self.epoch

# ********************************************************************************************************************* 
class Linear(keras.layers.Layer):

    def __init__(self, units=1, input_dim=1, **kwargs):
        super(Linear, self).__init__(**kwargs)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype='float32'),trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# *********************************************************************************************************************
# minimal model with custom layer        
x = keras.Input(shape=(1,), name='x')
y = Linear(units=1, input_dim=1, name='linear')(x)
model1 = keras.Model(inputs=x, outputs=y)

# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# *********************************************************************************************************************
# Dataset of observations: synthetic data on first 1000 asteroids
n0: int = 1
# n1: int = 1000
n1: int = 10
ds, ts, row_len = make_synthetic_obs_dataset(n0=n0, n1=n1)
# Get example batch
batch_in, batch_out = list(ds.take(1))[0]
# Contents of this batch
t = batch_in['t']
u_obs = batch_in['u_obs']
row_len = batch_in['row_len']

# Get trajectory size
batch_size: int = t.shape[0]
traj_size: int = ts.shape[0]
max_obs: int = u_obs.shape[1]
space_dims: int = 3

# Example batch of orbital elements
elts_np = orbital_element_batch(1)

# *********************************************************************************************************************
# minimal model with OrbitalElements layer        
traj_size: int = ts.shape[0]
space_dims: int = 3

# Inputs
t = keras.Input(shape=(1,), name='t')
u_obs = keras.Input(shape=(max_obs, space_dims), name='u_obs')

# Direction model
model_ast_dir = make_model_ast_dir(ts=ts, batch_size=batch_size)

# Orbital elements to try
elts_np = orbital_element_batch(1)

# Change to elements
# da = keras.layers.Dense(units=1, use_bias=False, kernel_initializer='ones', name='a')(1.0)

# Orbital elements layer
elts_layer = OrbitalElements(elts_np=elts_np, batch_size=batch_size, name='elts')
# Unpack orbital elements
a, e, inc, Omega, omega, f, epoch = elts_layer(t)

# Compute asteroid directions
# u_pred = model_ast_dir.predict_on_batch(elts.inputs_ast_dir)

# Wrap inputs and outputs
inputs = (t, u_obs,)
outputs = (a,)

# Create model with functional API    
model = keras.Model(inputs=inputs, outputs=outputs)


