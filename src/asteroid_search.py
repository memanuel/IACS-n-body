"""
Harvard IACS Masters Thesis
asteroid_search.py: Search for orbital elements of asteroids given observational data.

Michael S. Emanuel
Thu Oct 17 15:24:10 2019
"""

# Library imports
import tensorflow as tf
# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import numpy as np
from datetime import datetime

# Local imports
from asteroid_integrate import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, random_direction
# from observation_data import make_synthetic_obs_tensors
from asteroid_data import get_earth_pos, orbital_element_batch
from asteroid_model import AsteroidDirection
from asteroid_model import AsteroidPosition, DirectionUnitVector
# from asteroid_model make_model_ast_dir, make_model_ast_pos
from search_score_functions import score_mean, score_var, score_mean_2d, score_var_2d
# score_mean_2d_approx, score_var_2d_approx
from tf_utils import Identity

# Aliases
keras = tf.keras

# Constants
space_dims = 3

# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()
# Range for resolution parameter
log_R_min_ = np.log(np.deg2rad(1.0/3600))
log_R_max_ = np.log(np.deg2rad(10.0))


# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class DirectionDifference(keras.layers.Layer):
    """Compute the difference in direction between observed and predicted directions"""
    def __init__(self, batch_size: int, traj_size: int, max_obs: int, **kwargs):
        super(DirectionDifference, self).__init__(**kwargs)
        # Save sizes
        self.batch_size = batch_size
        self.traj_size = traj_size
        self.max_obs = max_obs
    
    def call(self, u_obs, u_pred, idx):
        """
        INPUTS:
            u_obs: observed directions, PADDED to a regular tensor; shape (traj_size, max_obs, 3,)
            u_pred: predicted directions; shape (batch_size, traj_size, 3,)
        """
        # Get sizes
        # batch_size: int
        # traj_size: int
        # batch_size, traj_size = u_pred.shape[0:2]
        # max_obs: int = u_obs.shape[-1]
        batch_size = self.batch_size
        traj_size = self.traj_size
        max_obs = self.max_obs

        # Slice of the full trajectory
        i0 = idx[0]
        i1 = idx[-1] + 1
        u_pred_slice = u_pred[:,i0:i1]
        # Manually set the shapes to work around documented bug on slices losing shape info
        u_slice_shape = (batch_size, traj_size, 3)
        u_pred_slice.set_shape(u_slice_shape)

        # Debug
        # print(f'u_obs.shape = {u_obs.shape}')
        # print(f'u_pred.shape = {u_pred.shape}')
        # print(f'u_pred_slice.shape = {u_pred_slice.shape}')
        # print(f'batch_size={batch_size}, traj_size={traj_size}, max_obs={max_obs}.')
        # print(f'i0={i0}, i1={i1}.')

        # The observations; broadcast to shape (1, traj_size, max_obs, 3)
        y = tf.broadcast_to(u_obs, (1, traj_size, max_obs, space_dims))
        # print(f'y.shape = {y.shape}')
        # The predicted directions; reshape to (batch_size, traj_size, 1, 3)
        x = tf.reshape(u_pred_slice, (batch_size, traj_size, 1, space_dims))
        # print(f'x.shape = {x.shape}')
        
        # The difference in directions; size (batch_size, traj_size, max_obs, 3)
        z = tf.subtract(y, x, name='z')
        # print(f'z.shape = {z.shape}')

        return z

# ********************************************************************************************************************* 
class TrajectoryScore(keras.layers.Layer):
    """Score candidate trajectories"""
    def __init__(self, batch_size: int, **kwargs):
        super(TrajectoryScore, self).__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, z: tf.Tensor, R: tf.Tensor, num_obs: float):
        """
        Score candidate trajectories in current batch based on how well they match observations
        INPUTS:
            z: difference in direction between u_pred and u_obs
            R: resolution factor in radians for score function
            num_obs: total number of observations (real, not padded!)
        """
        # The scaling coefficient for scores; score = exp(-1/2 A epsilon^2)
        A = 1.0 / R**2
        
        # The coefficient that multiplies epsilon^2
        B = tf.reshape(-0.5 * A, (self.batch_size, 1, 1,))
        # print(f'B.shape = {B.shape}')
        
        # Argument to the exponential
        arg = tf.multiply(B, tf.linalg.norm(z, axis=-1))
        # print(f'arg.shape = {arg.shape}')
        
        # The score function
        score = K.sum(tf.exp(arg), axis=(1,2))
        # print(f'score.shape = {score.shape}')
        
        # The expected score
        wt_3d = 0.99
        wt_2d = 0.01
        mu_per_obs = wt_3d*score_mean(A) + wt_2d * score_mean_2d(A)
        mu = tf.multiply(num_obs, mu_per_obs, name='mu')
        # print(f'mu.shape = {mu.shape}')
        
        # The expected variance; use the 2D (plane) estimate
        var_per_obs = wt_3d *score_var(A) + wt_2d * score_var_2d(A)
        sigma2 = tf.multiply(num_obs, var_per_obs, name='sigma2')
        sigma = tf.sqrt(sigma2, name='sigma')
        # print(f'sigma.shape = {sigma.shape}')
        
        # The t-score for each sample
        # This is a modified t-score with an extra penalty for overly large R
        t_score = (score - mu) / sigma
        # print(f't_score.shape = {t_score.shape}')

        # Return both the raw and t scores
        return score, t_score, mu, sigma

# ********************************************************************************************************************* 
class SearchCandidates(keras.layers.Layer):
    """Custom layer to maintain state of candidate orbital elements and resolutions."""

    def __init__(self, elts_np: dict, batch_size: int, R_deg: float, **kwargs):
        super(SearchCandidates, self).__init__(**kwargs)
        self.a = tf.Variable(initial_value=elts_np['a'], trainable=True, name='a')
        self.e = tf.Variable(initial_value=elts_np['e'], trainable=True, name='e')
        self.inc = tf.Variable(initial_value=elts_np['inc'], trainable=True, name='inc')
        self.Omega = tf.Variable(initial_value=elts_np['Omega'], trainable=True, name='Omega')
        self.omega = tf.Variable(initial_value=elts_np['omega'], trainable=True, name='omega')
        self.f = tf.Variable(initial_value=elts_np['f'], trainable=True, name='f')

        # The epoch is not trainable
        self.epoch = tf.Variable(initial_value=elts_np['epoch'], trainable=False, name='epoch')
        
        # The control of the log of the resolution factor between R_min and R_max
        R_init= np.deg2rad(R_deg) * np.ones_like(elts_np['a'])
        log_R_init  = np.log(R_init)
        self.log_R = tf.Variable(initial_value=log_R_init, trainable=True, name='log_R')

    def call(self, inputs):
        """Return the current settings"""
        print(f'type(inputs)={type(inputs)}.')
        return self.a, self.e, self.inc, self.Omega, self.omega, self.f, self.epoch, self.log_R

# ********************************************************************************************************************* 
# Functional API model
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_asteroid_search(ts: tf.Tensor,
                               elts_np: dict,
                               max_obs: int,
                               num_obs: float,
                               elt_batch_size: int=64, 
                               time_batch_size: int=None,
                               R_deg: float = 5.0):
    """Make functional API model for scoring elements"""

    # The full trajectory size
    traj_size: int = ts.shape[0]
    # Default for time_batch_size is full trajectory size
    if time_batch_size is None:
        time_batch_size = traj_size

    # Inputs
    t = keras.Input(shape=(), batch_size=time_batch_size, dtype=tf.float32, name='t' )
    idx = keras.Input(shape=(), batch_size=time_batch_size, dtype=tf.int32, name='idx')
    row_len = keras.Input(shape=(), batch_size=time_batch_size, dtype=tf.int32, name='row_len')
    u_obs = keras.Input(shape=(max_obs, space_dims), batch_size=time_batch_size, dtype=tf.float32, name='u_obs')
    
    # Output times are a constant
    ts = keras.backend.constant(ts, name='ts')

    # Set of trainable weights with candidate
    a, e, inc, Omega, omega, f, epoch, log_R = \
        SearchCandidates(elts_np=elts_np, batch_size=elt_batch_size, R_deg=R_deg, name='candidates')(idx)

    # Alias the orbital elements; 6 are trainable, epoch is fixed
    a = Identity(name='a')(a)
    e = Identity(name='e')(e)
    inc = Identity(name='inc')(inc)
    Omega = Identity(name='Omega')(Omega)
    omega = Identity(name='omega')(omega)
    f = Identity(name='f')(f)
    epoch = Identity(name='epoch')(epoch)

    # Transform the resolution output
    log_R = Identity(name='log_R')(log_R)
    # Clip log_R in allowed range
    log_R_min = tf.constant(log_R_min_, dtype=tf.float32, name='log_R_min')
    log_R_max = tf.constant(log_R_max_, dtype=tf.float32, name='log_R_max')
    log_R_clip = tf.clip_by_value(log_R, log_R_min, log_R_max, name='log_R_clip')
    # The resolution from log_R, with clipping
    R = keras.layers.Activation(activation=tf.exp, name='R')(log_R_clip)

    # The orbital elements; stack to shape (elt_batch_size, 7)
    elts = tf.stack(values=[a, e, inc, Omega, omega, f, epoch], axis=1, name='elts')

    # The predicted direction
    u_pred = AsteroidDirection(ts=ts, batch_size=elt_batch_size, name='u_pred')(a, e, inc, Omega, omega, f, epoch)

    # Difference in direction between u_obs and u_pred
    z = DirectionDifference(batch_size=elt_batch_size, 
                            traj_size=time_batch_size, 
                            max_obs=max_obs, 
                            name='z')(u_obs, u_pred, idx)
    
    # Raw score and t_score
    score, t_score, mu, sigma = TrajectoryScore(batch_size=elt_batch_size)(z, R, num_obs)
    # Stack the scores
    scores = tf.stack(values=[score, t_score, mu, sigma], axis=1, name='scores')

    # Wrap inputs and outputs
    inputs = (t, idx, row_len, u_obs)
    outputs = (elts, R, u_pred, z, scores)

    # Create model with functional API
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Add custom loss; negative of the total t_score
    model.add_loss(-tf.reduce_sum(t_score))
       
    return model

# ********************************************************************************************************************* 
def perturb_elts(elts, sigma_a=0.05, sigma_e=0.10, sigma_f=np.deg2rad(5.0), mask=None):
    """Apply perturbations to orbital elements"""

    # Copy the elements
    elts_new = elts.copy()

    # Default for mask is all elements
    if mask is None:
        mask = np.ones_like(elts['a'], dtype=bool)

    # Number of elements to perturb
    num_shift = np.sum(mask)
    
    # Apply shift log(a)
    log_a = np.log(elts['a'])
    log_a[mask] += np.random.normal(scale=sigma_a, size=num_shift)
    elts_new['a'] = np.exp(log_a)
    
    # Apply shift to log(e)
    log_e = np.log(elts['e'])
    log_e[mask] += np.random.normal(scale=sigma_e, size=num_shift)
    elts_new['e'] = np.exp(log_e)
    
    # Apply shift directly to true anomaly f
    f = elts['f']
    f[mask] += np.random.normal(scale=sigma_f, size=num_shift)
    elts_new['f'] = f
    
    return elts_new

# ********************************************************************************************************************* 
def main():
    pass

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()

# ********************************************************************************************************************* 
def report_model(model, R0, mask_good):
    """Report summary of model on good and b"""
    mask_bad = ~mask_good
    
    loss = model.evaluate(ds)
    pred = model.predict_on_batch(ds)
    elts, R, u_pred, z, scores = pred
    # raw_score = scores[:,0]
    t_score = scores[:,1]
    # mu = scores[:,2]
    # sigma = scores[:,3]
    
    # Compare original and revised orbital elements
    R0 = np.ones(elt_batch_size) * np.deg2rad(R_deg)
#    d_elt = elts - elts0
#    d_a = d_elt[:,0]
#    d_e = d_elt[:,1]
#    d_inc = d_elt[:,2]
#    d_Omega = d_elt[:,3]
#    d_omega = d_elt[:,4]
#    d_f = d_elt[:,5]
    d_R = R - R0
    
    # Mean t_score on good and bad masks
    mean_good = np.mean(t_score[mask_good])
    std_good = np.std(t_score[mask_good])
    mean_bad = np.mean(t_score[mask_bad])
    std_bad = np.std(t_score[mask_bad])
    
    dR_good = np.mean(d_R[mask_good])
    dR_bad = np.mean(d_R[mask_bad])
    
    print(f'Processed first {n1} asteroids; seeded with first 64. Perturbed 33-65.')
    print(f'Mean & std t_score by Category:')
    print(f'Good: {mean_good:8.2f} +/- {std_good:8.2f}')
    print(f'Bad:  {mean_bad:8.2f} +/- {std_bad:8.2f}')
    print(f'Change in resolution R By Category:')
    print(f'Good: {dR_good:+8.6f}')
    print(f'Bad:  {dR_bad:+8.6f}')
    print(f'Loss = {loss:8.0f}')

# ********************************************************************************************************************* 
# Dataset of observations: synthetic data on first 1000 asteroids

# Build the dataset
n0: int = 1
n1: int = 100
dt0: datetime = datetime(2000,1,1)
dt1: datetime = datetime(2019,1,1)
time_batch_size = 1024
ds, ts, row_len = make_synthetic_obs_dataset(n0=n0, n1=n1, dt0=dt0, dt1=dt1, batch_size=time_batch_size)

# def run_training():
# Get example batch
batch_in, batch_out = list(ds.take(1))[0]
# Contents of this batch
t = batch_in['t']
idx = batch_in['idx']
row_len = batch_in['row_len']
u_obs = batch_in['u_obs']

# Get trajectory size and max_obs
traj_size: int = ts.shape[0]
max_obs: int = u_obs.shape[1]
num_obs: float = np.sum(row_len, dtype=np.float32)

# Build functional model for asteroid score
R_deg: float = 10.0
elts_np = orbital_element_batch(1)
elt_batch_size = 64

# Mask where data expected vs not
mask_good = np.arange(64) < 32
mask_bad = ~mask_good
# Perturb second half of orbital elements
elts_np2 = perturb_elts(elts_np, mask=mask_bad)

model = make_model_asteroid_search(ts=ts,
                                   elts_np=elts_np2,
                                   max_obs=max_obs,
                                   num_obs=num_obs,
                                   elt_batch_size=elt_batch_size, 
                                   time_batch_size=time_batch_size,
                                   R_deg = R_deg)

model.compile(optimizer='Adam', learning_rate=0.05)
loss0 = model.evaluate(ds)
pred0 = model.predict(ds)
elts0, R0, u_pred0, z0, scores0 = pred0
t_score0 = scores0[:,1]

hist = model.fit(ds, epochs=10)
pred = model.predict(ds)
report_model(model, R0, mask_good)
pred = model.predict(ds)
elts, R, u_pred, z, scores = pred
dR = R - R0