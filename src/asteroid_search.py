"""
Harvard IACS Masters Thesis
asteroid_search.py: Search for orbital elements of asteroids given observational data.

Michael S. Emanuel
Thu Oct 17 15:24:10 2019
"""

# Library imports
import tensorflow as tf
from tensorflow.python.keras import backend as K
import tensorflow_probability as tfp
import numpy as np
from datetime import datetime
import logging
from typing import Dict

# Local imports
from asteroid_integrate import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, random_direction
# from observation_data import make_synthetic_obs_tensors
from asteroid_data import orbital_element_batch
from asteroid_model import AsteroidDirection
from asteroid_integrate import calc_ast_pos
# from asteroid_model make_model_ast_dir, make_model_ast_pos
from search_score_functions import score_mean, score_var, score_mean_2d, score_var_2d
# score_mean_2d_approx, score_var_2d_approx
from tf_utils import Identity

# Aliases
keras = tf.keras
tfd = tfp.distributions

# ********************************************************************************************************************* 
# Turn off all logging; only solution found to eliminate crushing volume of unresolvable autograph warnings
logging.getLogger('tensorflow').disabled = True

# Constants
space_dims = 3

# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# Range for a
log_a_min_ = np.log(0.10) 
log_a_max_ = np.log(100.0) 

# Range for e
log_e_min_ = np.log(1.0E-6)
log_e_max_ = np.log(0.999999)

# Range for resolution parameter R
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
        raw_score = K.sum(tf.exp(arg), axis=(1,2))
        # print(f'score.shape = {score.shape}')
        
        # The expected score
        wt_3d = 1.00
        wt_2d = 0.00
        mu_per_obs = wt_3d*score_mean(A) + wt_2d*score_mean_2d(A)
        mu = tf.multiply(num_obs, mu_per_obs, name='mu')
        # print(f'mu.shape = {mu.shape}')
        
        # The expected variance; use the 2D (plane) estimate
        var_per_obs = wt_3d*score_var(A) + wt_2d*score_var_2d(A)
        sigma2 = tf.multiply(num_obs, var_per_obs, name='sigma2')
        sigma = tf.sqrt(sigma2, name='sigma')
        # print(f'sigma.shape = {sigma.shape}')
        
        # Effective number of observations
        eff_obs = raw_score - mu
        
        # The t-score for each sample
        t_score = eff_obs / sigma
        # print(f't_score.shape = {t_score.shape}')
        
        # Assemble the objective function to be maximized
        scale_factor = 1.0 / self.batch_size
        mu_factor = 1.0
        # sigma_power = 0.0
        # quality_factor = 0.80*tfd.Normal(loc=0.0, scale=1.0).cdf(t_score) + 0.20
        cdf = tf.nn.sigmoid(t_score)
        quality_factor = 0.80*cdf + 0.20
        # objective = scale_factor * (raw_score - mu_factor * mu) * tf.math.pow(sigma, -sigma_power)
        objective = scale_factor * quality_factor * (raw_score - mu_factor * mu)

        # Return both the raw and t scores
        return raw_score, t_score, mu, sigma, objective

# ********************************************************************************************************************* 
class OrbitalElements(keras.layers.Layer):
    """Custom layer to maintain state of candidate orbital elements and resolutions."""

    def __init__(self, elts_np: dict, batch_size: int, R_deg: float, **kwargs):
        super(OrbitalElements, self).__init__(**kwargs)
        self.log_a = tf.Variable(initial_value=np.log(elts_np['a']), trainable=True, 
                                 constraint=lambda t: tf.clip_by_value(t, log_a_min_, log_a_max_), name='log_a')
        self.log_e = tf.Variable(initial_value=np.log(elts_np['e']), trainable=True, 
                                 constraint=lambda t: tf.clip_by_value(t, log_e_min_, log_e_max_), name='log_e')
        self.inc = tf.Variable(initial_value=elts_np['inc'], trainable=False, name='inc')
        self.Omega = tf.Variable(initial_value=elts_np['Omega'], trainable=False, name='Omega')
        self.omega = tf.Variable(initial_value=elts_np['omega'], trainable=False, name='omega')
        self.f = tf.Variable(initial_value=elts_np['f'], trainable=True, name='f')

        # The epoch is not trainable
        self.epoch = tf.Variable(initial_value=elts_np['epoch'], trainable=False, name='epoch')
        
        # The control of the log of the resolution factor between R_min and R_max
        R_init= np.deg2rad(R_deg) * np.ones_like(elts_np['a'])
        log_R_init  = np.log(R_init)
        self.log_R = tf.Variable(initial_value=log_R_init, trainable=False, 
                                 constraint=lambda t: tf.clip_by_value(t, log_R_min_, log_R_max_), name='log_R')
        
        # Actual values of a, e, R - for inspection
        self.a = tf.exp(self.log_a)
        self.e = tf.exp(self.log_e)
        self.R = tf.exp(self.log_R)

    def call(self, inputs):
        """Return the current orbital elements and resolution"""
        # print(f'type(inputs)={type(inputs)}.')
        # return self.log_a, self.log_e, self.inc, self.Omega, self.omega, self.f, self.epoch, self.log_R
        # Transform a, e, and R from log to linear
        a = tf.exp(self.log_a)
        e = tf.exp(self.log_e)
        R = tf.exp(self.log_R)
        return a, e, self.inc, self.Omega, self.omega, self.f, self.epoch, R

# ********************************************************************************************************************* 
# Functional API model
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_asteroid_search(ts: tf.Tensor,
                               elts_np: Dict,
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
    elements_layer = OrbitalElements(elts_np=elts_np, batch_size=elt_batch_size, R_deg=R_deg, name='candidates')
    a, e, inc, Omega, omega, f, epoch, R = elements_layer(idx)
    
    # Alias the orbital elements; a, e, inc, Omega, omega, and f are trainable; epoch is fixed
    a = Identity(name='a')(a)
    e = Identity(name='e')(e)
    inc = Identity(name='inc')(inc)
    Omega = Identity(name='Omega')(Omega)
    omega = Identity(name='omega')(omega)
    f = Identity(name='f')(f)
    epoch = Identity(name='epoch')(epoch)

    # Alias the resolution output
    R = Identity(name='R')(R)

    # The orbital elements; stack to shape (elt_batch_size, 7)
    elts = tf.stack(values=[a, e, inc, Omega, omega, f, epoch], axis=1, name='elts')

    # The predicted direction
    direction_layer = AsteroidDirection(ts=ts, batch_size=elt_batch_size, name='u_pred')

    # Compute numerical orbits for calibration
    epoch0 = elts_np['epoch'][0]
    print(f'Numerically integrating orbits for calibration...')
    q_ast, q_sun, q_earth = calc_ast_pos(elts=elts_np, epoch=epoch0, ts=ts)

    # Calibrate the direction prediction layer
    direction_layer.calibrate(elts=elts_np, q_ast=q_ast, q_sun=q_sun)
    # Tensor of predicted directions
    u_pred = direction_layer(a, e, inc, Omega, omega, f, epoch)

    # Difference in direction between u_obs and u_pred
    z = DirectionDifference(batch_size=elt_batch_size, 
                            traj_size=time_batch_size, 
                            max_obs=max_obs, 
                            name='z')(u_obs, u_pred, idx)
    
    # Raw score and t_score
    score_layer = TrajectoryScore(batch_size=elt_batch_size)
    raw_score, t_score, mu, sigma, objective = score_layer(z, R, num_obs)
    # Stack the scores
    scores = tf.stack(values=[raw_score, t_score, mu, sigma, objective], axis=1, name='scores')

    # Wrap inputs and outputs
    inputs = (t, idx, row_len, u_obs)
    outputs = (elts, R, u_pred, z, scores)

    # Create model with functional API
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Bind the custom layers to model
    model.elements = elements_layer
    model.direction = direction_layer
    model.score = score_layer
    
    # Add custom loss; negative of the total objective function
    model.add_loss(-tf.reduce_sum(objective))

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
def report_model(model, R0, mask_good, steps, elts_true):
    """Report summary of model on good and b"""
    mask_bad = ~mask_good

    loss = model.evaluate(ds, steps=steps)
    pred = model.predict_on_batch(ds)
    elts, R, u_pred, z, scores = pred
    raw_score = scores[:,0]
    t_score = scores[:,1]
    mu = scores[:,2]
    # sigma = scores[:,3]
    eff_obs = raw_score - mu
    
    # Change in resolution
    R0 = np.ones(elt_batch_size) * np.deg2rad(R_deg)
    d_R = R - R0

    # Error in orbital elements
    elt_err = np.abs(elts - elts_true)
    # Mean element error on good and bad masks
    elt_err_g = elt_err[mask_good]
    elt_err_b = elt_err[mask_bad]
    mean_err_g = np.mean(elt_err_g[0:6], axis=0)
    mean_err_b = np.mean(elt_err_b[0:6], axis=0)

    # Mean & std t_score on good and bad masks
    t_mean_g = np.mean(t_score[mask_good])
    t_std_g = np.std(t_score[mask_good])
    t_mean_b = np.mean(t_score[mask_bad])
    t_std_b = np.std(t_score[mask_bad])
    
    # Mean & std Effective observations
    obs_mean_g = np.mean(eff_obs[mask_good])
    obs_std_g = np.std(eff_obs[mask_good])
    obs_mean_b = np.mean(eff_obs[mask_bad])
    obs_std_b = np.std(eff_obs[mask_bad])

    # Change in resolution    
    dR_good = np.mean(d_R[mask_good])
    dR_bad = np.mean(d_R[mask_bad])
        
    print(f'\nLoss = {loss:8.0f}')
    print(f'\nError in orbital elements:')
    print(f'Good: {mean_err_g[0]:5.2e},  {mean_err_g[1]:5.2e}, {mean_err_g[2]:5.2e}, '
          f'{mean_err_g[0]:5.2e},  {mean_err_g[1]:5.2e}, {mean_err_g[2]:5.2e}, ')
    print(f'Bad : {mean_err_b[0]:5.2e},  {mean_err_b[1]:5.2e}, {mean_err_b[2]:5.2e}, '
          f'{mean_err_b[0]:5.2e},  {mean_err_b[1]:5.2e}, {mean_err_b[2]:5.2e}, ')

    print(f'\nMean & std Effective Observations by Category:')
    print(f'Good: {obs_mean_g:8.2f} +/- {obs_std_g:8.2f}')
    print(f'Bad:  {obs_mean_b:8.2f} +/- {obs_std_b:8.2f}')

    print(f'\nMean & std t_score by Category:')
    print(f'Good: {t_mean_g:8.2f} +/- {t_std_g:8.2f}')
    print(f'Bad:  {t_mean_b:8.2f} +/- {t_std_b:8.2f}')
    
    print(f'\nChange in resolution R By Category:')
    print(f'Good: {dR_good:+8.6f}')
    print(f'Bad:  {dR_bad:+8.6f}')
    return mean_err_g, mean_err_b

# ********************************************************************************************************************* 
# Dataset of observations: synthetic data on first 1000 asteroids

# Build the dataset
n0: int = 1
n1: int = 100
dt0: datetime = datetime(2000,1,1)
dt1: datetime = datetime(2019,1,1)
time_batch_size = 1024
traj_size = 14976
steps = int(np.ceil(traj_size / time_batch_size))
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
R_deg: float = 2.0
elts_np = orbital_element_batch(1)
elt_batch_size = 64
# The correct orbital elements as an array
elts_true = np.array([elts_np['a'], elts_np['e'], elts_np['inc'], elts_np['Omega'], 
                      elts_np['omega'], elts_np['f'], elts_np['epoch']]).transpose()

# Mask where data expected vs not
mask_good = np.arange(64) < 32
mask_bad = ~mask_good
# Perturb second half of orbital elements
elts_np2 = perturb_elts(elts_np, mask=mask_bad)
# Initialize model with perturbed orbital elements
model = make_model_asteroid_search(ts=ts,
                                   elts_np=elts_np2,
                                   max_obs=max_obs,
                                   num_obs=num_obs,
                                   elt_batch_size=elt_batch_size, 
                                   time_batch_size=time_batch_size,
                                   R_deg = R_deg)
# Use Adam optimizer with gradient clipping
# opt = keras.optimizers.Adam(learning_rate=0.0005, clipvalue=1.0)
opt = keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0)
# opt = keras.optimizers.Adadelta(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=opt)

# Report losses before training
print(f'Processed first {n1} asteroids; seeded with first 64. Perturbed 33-65.')
print(f'\nModel before training:')
# loss0 = model.evaluate(ds, steps=steps)
pred0 = model.predict_on_batch(ds)
elts0, R0, u_pred0, z0, scores0 = pred0
mean_err0_g, mean_err0_b = report_model(model, R0, mask_good, steps, elts_true)
raw_score0 = scores0[:,0]
t_score0 = scores0[:,1]
mu0 = scores0[:, 2]
sigma0 = scores0[:,3]

# Get gradients
with tf.GradientTape(persistent=True) as gt:
    pred = model.predict_on_batch(ds)
    # pred = model.predict_on_batch(ds.take(traj_size))
    elts, R, u_pred, z, scores = pred
    raw_score = scores[:, 0]
    t_score = scores[:, 1]
    mu = scores[:, 2]
    objective = scores[:, 4]
    loss = tf.reduce_sum(objective)
dL_da = gt.gradient(loss, model.elements.log_a)
dL_de = gt.gradient(loss, model.elements.log_e)
dL_dinc = gt.gradient(loss, model.elements.inc)
dL_dOmega = gt.gradient(loss, model.elements.Omega)
dL_domega = gt.gradient(loss, model.elements.omega)
dL_df = gt.gradient(loss, model.elements.f)
dL_dR = gt.gradient(loss, model.elements.log_R)
del gt

# Train model
step_multiplier = 4
steps_per_epoch = steps*step_multiplier
hist = model.fit(ds, epochs=10, steps_per_epoch=steps*step_multiplier)
pred = model.predict_on_batch(ds)
elts, R, u_pred, z, scores = pred
raw_score = scores[:,0]
t_score = scores[:,1]
mu = scores[:, 2]
sigma = scores[:,3]

# Report results
print(f'\nModel after training:')
mean_err_g, mean_err_b = report_model(model, R0, mask_good, steps, elts_true)

# Changes in element errors and R
d_err_g = mean_err_g - mean_err0_g
d_err_b = mean_err_b - mean_err0_b

print(f'Change in Orbital Element error by Category:')
print(f'd_err_g: {d_err_g[0]:+5.2e},  {d_err_g[1]:+5.2e}, {d_err_g[2]:+5.2e}, '
      f'{d_err_g[0]:+5.2e},  {d_err_g[1]:+5.2e}, {d_err_g[2]:+5.2e}, ')
print(f'd_err_b: {d_err_b[0]:+5.2e},  {d_err_b[1]:+5.2e}, {d_err_b[2]:+5.2e}, '
      f'{d_err_b[0]:+5.2e},  {d_err_b[1]:+5.2e}, {d_err_b[2]:+5.2e}, ')

err_a = np.abs(model.elements.a - elts_np['a'])
err_e = np.abs(model.elements.a - elts_np['e'])