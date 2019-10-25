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

# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# ********************************************************************************************************************* 
class AsteroidSearchModel(keras.Model):
    def __init__(self, ts: np.array, row_len, 
                 batch_size: int = 64, R_deg: float=10.0, **kwargs):
        super(AsteroidSearchModel, self).__init__(**kwargs)

        # Save input arguments
        self.ts = ts
        self.batch_size = batch_size
        self.R_deg = R_deg
        self.row_len = row_len
        
        # The trajectory size
        self.traj_size = ts.shape[0]
        # Total number of observations; cast to tf.float32 type for compatibility with score
        self.num_obs = tf.cast(tf.reduce_sum(row_len), tf.float32)
        
        # Load asteroid names and orbital elements
        self.ast_elt = load_data_asteroids()
        # Placeholder for orbital elements
        elt_placeholder = np.zeros(shape=(batch_size,), dtype=np.float32)

        # Create trainable variables for the 6 orbital elements
        self.a = tf.Variable(initial_value=elt_placeholder, trainable=True, name='a')
        self.e = tf.Variable(initial_value=elt_placeholder, trainable=True, name='e')
        self.inc = tf.Variable(initial_value=elt_placeholder, trainable=True, name='inc')
        self.Omega = tf.Variable(initial_value=elt_placeholder, trainable=True, name='Omega')
        self.omega = tf.Variable(initial_value=elt_placeholder, trainable=True, name='omega')
        self.f = tf.Variable(initial_value=elt_placeholder, trainable=True, name='f')
        
        # Create a non-trainable variable for the epoch
        self.epoch = tf.Variable(initial_value=elt_placeholder, trainable=False, name='epoch')
        
        # Initialize the orbital elements to the first 64 asteroids
        self.set_orbital_elements_asteroids(1)
        
        # Dictionary wrapping inputs for the asteroid direction model
        self.inputs_ast_dir = {
            'a': self.a,
            'e': self.e,
            'inc': self.inc,
            'Omega': self.Omega,
            'omega': self.omega,
            'f': self.f,
            'epoch': self.epoch,        
        }                

        # Create trainable variable for resolution factor
        R_np = np.deg2rad(self.R_deg) * np.ones(shape=batch_size, dtype=np.float32)
        self.R = tf.Variable(initial_value=R_np, trainable=True, name='R')
        
        # Create a layer to compute directions from orbital elements at these times
        self.model_ast_dir = make_model_ast_dir(ts=ts, batch_size=batch_size)
        
        # Create variable to store the predicted directions
        u_pred_shape = shape=(batch_size, traj_size, space_dims)
        self.u_pred = tf.Variable(initial_value=self.model_ast_dir.predict(self.inputs_ast_dir), trainable=False, name='u_pred')
        
        # Create layer to score trajectories
        # self.traj_score = TrajectoryScore(batch_size=self.batch_size, name='traj_score')
        
        # Variable to save the raw and t scores
        # self.score = tf.Variable(initial_value=elt_placeholder, trainable=False, name='score')
        # self.t_score = tf.Variable(initial_value=elt_placeholder, trainable=False, name='t_score')
        
        # Create layer to accumulate losses
        # self.traj_loss= TrajectoryLoss()

    def set_orbital_elements_asteroids(self, n0: int):
        """Set a batch of orbital elements with data for a block of asteroids starting at n0"""
        # Get start and end index location of this asteroid number
        i0: int = ast_elt.index.get_loc(n0)
        i1: int = i0 + self.batch_size
        # Assign the orbital elements and epoch
        self.a.assign(ast_elt.a[i0:i1])
        self.e.assign(ast_elt.e[i0:i1])
        self.inc.assign(ast_elt.inc[i0:i1])
        self.Omega.assign(ast_elt.Omega[i0:i1])
        self.omega.assign(ast_elt.omega[i0:i1])
        self.f.assign(ast_elt.f[i0:i1])
        self.epoch.assign(ast_elt.epoch_mjd[i0:i1])

    def set_R(self, R_deg: float):
        """Update the resolution factor"""
        R_np = np.deg2rad(R_deg) * np.ones(shape=self.batch_size, dtype=np.float32)
        self.R.assign(R_np)

    def predict_directions(self):
        """
        Compute orbits and implied directions for one batch of orbital element parameters.
        """
        # Predict asteroid positions and directions from earth;
        # update u_pred in place with the results
        u_pred = self.model_ast_dir.predict_on_batch(self.inputs_ast_dir)
        # self.u_pred.assign(u_pred)
        
        # Predict asteroid directions from earth with current orbital elements
        # return self.model_ast_dir.predict_on_batch(self.inputs_ast_dir)
        # return self.model_ast_dir.predict_on_batch((self.a, self.e, self.inc, self.Omega, self.omega, self.f, self.epoch))

    def call(self, inputs):
        """
        Compute orbits and implied directions for one batch of orbital element parameters.
        Score the predicted directions with the current resolution settings.
        """
        # Unpack the inputs
        # t, u_obs, row_len = inputs
        t = inputs['t']
        u_obs = inputs['u_obs']
        row_len = inputs['row_len']
                
        # Debug problem in predict_directions()
        self.predict_directions()
        
        # model_ast_dir = make_model_ast_dir(ts=self.ts, batch_size=self.batch_size)
        # inputs_ast_dir = self.inputs_ast_dir  
        # u_pred = model_ast_dir.predict_directions(inputs_ast_dir)
        
        # **** original calculation
        
        # Predicted asteroid positions
        # self.predict_directions()

        # Difference in between observed and predicted directions
        # z = DirectionDifference(name='direction_difference')(u_obs, self.u_pred)

        # raw and t score
        # score, t_score = self.traj_score(z, self.R, self.num_obs)
        # self.score.assign(score)
        # self.t_score.assign(t_score)        
        
        # add negative t_score to the loss
        # self.add_loss(self.traj_loss, inputs=self.t_score)
    
        # Return
        # return t_score
        return t

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

