"""
Harvard IACS Masters Thesis
asteroid_search.py: Search for orbital elements of asteroids given observational data.

Michael S. Emanuel
Thu Oct 17 15:24:10 2019
"""

# Library imports
import tensorflow as tf
from tensorflow.python.keras import backend as K
import numpy as np
from datetime import datetime
import logging
from typing import Dict

# Local imports
from asteroid_integrate import load_data as load_data_asteroids
from observation_data import make_synthetic_obs_dataset, random_direction
from asteroid_data import orbital_element_batch
from asteroid_model import AsteroidDirection
from asteroid_integrate import calc_ast_pos
from search_score_functions import score_mean, score_var, score_mean_2d, score_var_2d
from tf_utils import Identity

# Aliases
keras = tf.keras

# ********************************************************************************************************************* 
# Turn off all logging; only solution found to eliminate crushing volume of unresolvable autograph warnings
logging.getLogger('tensorflow').disabled = True

# Constants
space_dims = 3

# Load asteroid names and orbital elements
ast_elt = load_data_asteroids()

# Range for a
a_min_: float = 0.125
a_max_: float = 64.0
# log_a_min_: float = np.log(a_min_) 
# log_a_max_: float = np.log(a_max_)

# Range for e
e_min_: float = 2.0**-20
e_max_: float = 1.0 - e_min_

# Range for resolution parameter R
R_min_ = np.deg2rad(1.0/3600)
R_max_ = np.deg2rad(10.0)
log_R_min_ = np.log(R_min_)
log_R_max_ = np.log(R_max_)

# ********************************************************************************************************************* 
# Custom Layers
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class OrbitalElements(keras.layers.Layer):
    """Custom layer to maintain state of candidate orbital elements and resolutions."""

    def __init__(self, elts_np: dict, batch_size: int, R_deg: float, **kwargs):
        super(OrbitalElements, self).__init__(**kwargs)
        
        # Configuration for serialization
        self.cfg = {
            'elts_np': elts_np,
            'batch_size': batch_size,
            'R_deg': R_deg,
        }
        
        self.batch_size = batch_size
        self.elts_np = elts_np
        self.R_deg = R_deg
        
        # Control over a_, in range 0.0 to 1.0
        self.a_min = tf.constant(a_min_, dtype=tf.float32)
        self.log_a_range = tf.constant(tf.math.log(a_max_) - tf.math.log(a_min_), dtype=tf.float32)
        self.a_ = tf.Variable(initial_value=self.inverse_a(elts_np['a']), trainable=True, 
                              constraint=lambda t: tf.clip_by_value(t, 0.0, 1.0), name='a_')
        
        # Control over e_, in range e_min to e_max
        self.e_min = tf.constant(e_min_, dtype=tf.float32)
        self.e_range = tf.constant(e_max_ - e_min_, dtype=tf.float32)
        self.e_ = tf.Variable(initial_value=self.inverse_e(elts_np['e']), trainable=False, 
                              constraint=lambda t: tf.clip_by_value(t, 0.0, 1.0), name='e_')
        
        # Control over inc_, in range -pi/2 to pi/2
        self.inc_min = tf.constant(np.pi/2*(1-2**-20), dtype=tf.float32)
        self.inc_range = tf.constant(2*self.inc_min, dtype=tf.float32)
        self.inc_ = tf.Variable(initial_value=self.inverse_inc(elts_np['inc']), trainable=False, 
                                constraint=lambda t: tf.clip_by_value(t, 0.0, 1.0), name='inc_')
        
        # Scale factor for unconstrained angles is 2*pi
        self.two_pi = tf.constant(2*np.pi, dtype=tf.float32)
        
        self.Omega_ = tf.Variable(initial_value=self.inverse_angle(elts_np['Omega']), trainable=False, name='Omega_')
        self.omega_ = tf.Variable(initial_value=self.inverse_angle(elts_np['omega']), trainable=False, name='omega_')
        self.f_ = tf.Variable(initial_value=self.inverse_angle(elts_np['f']), trainable=False, name='f_')

        # The epoch is not trainable
        self.epoch = tf.Variable(initial_value=elts_np['epoch'], trainable=False, name='epoch')
        
        # Control of the resolution factor R_, in range 0.0 to 1.0
        R_init = np.deg2rad(R_deg) * np.ones_like(elts_np['a'])
        # log_R_init  = np.log(R_init)
        self.R_min = tf.constant(R_min_, dtype=tf.float32)
        self.log_R_range = tf.constant(log_R_max_ - log_R_min_, dtype=tf.float32)
        self.R_ = tf.Variable(initial_value=self.inverse_R(R_init), trainable=False, 
                              constraint=lambda t: tf.clip_by_value(t, 0.0, 1.0), name='a_')
        
    def get_a(self):
        """Transformed value of a"""
        return self.a_min * tf.exp(self.a_ * self.log_a_range)

    def inverse_a(self, a):
        """Inverse transform value of a"""
        return tf.math.log(a / self.a_min) / self.log_a_range

    def get_e(self):
        """Transformed value of e"""
        return self.e_min + self.e_ * self.e_range

    def inverse_e(self, e):
        """Inverse transform value of e"""
        return (e - self.e_min) / self.e_range

    def get_inc(self):
        """Transformed value of inc"""
        return self.inc_min + self.inc_ * self.inc_range

    def inverse_inc(self, inc):
        """Inverse transform value of inc"""
        return (inc - self.inc_min) / self.inc_range

    def get_angle(self, angle_):
        """Forward transform of an unconstrained angle variable (Omega, omega, f)"""
        return self.two_pi * angle_

    def inverse_angle(self, angle):
        """Forward transform of an unconstrained angle variable (Omega, omega, f)"""
        return angle / self.two_pi

    def get_R(self):
        """Transformed value of R"""
        return self.R_min * tf.exp(self.R_ * self.log_R_range)

    def inverse_R(self, R):
        """Inverse transform value of R"""
        return tf.math.log(R / self.R_min) / self.log_R_range

    def call(self, inputs):
        """Return the current orbital elements and resolution"""
        # print(f'type(inputs)={type(inputs)}.')
        # Transform a, e, and R from log to linear
        a = self.get_a()
        e = self.get_e()
        inc = self.get_inc()
        Omega = self.get_angle(self.Omega_)
        omega = self.get_angle(self.omega_)
        f = self.get_angle(self.f_)
        R = self.get_R()
        return a, e, inc, Omega, omega, f, self.epoch, R

    def get_config(self):
        return self.cfg

# ********************************************************************************************************************* 
class DirectionDifference(keras.layers.Layer):
    """Compute the difference in direction between observed and predicted directions"""
    def __init__(self, batch_size: int, traj_size: int, max_obs: int, **kwargs):
        super(DirectionDifference, self).__init__(**kwargs)

        # Configuration for serialization
        self.cfg = {
            'batch_size': batch_size,
            'traj_size': traj_size,
            'max_obs': max_obs,
        }

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
    
    def get_config(self):
        return self.cfg

# ********************************************************************************************************************* 
class TrajectoryScore(keras.layers.Layer):
    """Score candidate trajectories"""
    def __init__(self, batch_size: int, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """
        INPUTS:
            batch_size: this is element_batch_size, the number of orbital elements per batch
            alpha: multiplicative factor on mu in objective function
            beta: multiplicative factor on sigma2 in objective function
        """
        super(TrajectoryScore, self).__init__(**kwargs)

        # Configuration for seralization
        self.cfg = {
            'batch_size': batch_size,
            'alpha': alpha,
            'beta': beta,
        }

        # Save sizes and parameters
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

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
        # print(f'raw_score.shape = {raw_score.shape}')
        
        # The expected score
        wt_3d = 0.95
        wt_2d = 0.05
        mu_per_obs = wt_3d*score_mean(A) + wt_2d*score_mean_2d(A)
        mu = tf.multiply(num_obs, mu_per_obs, name='mu')
        
        # The expected variance; use the 2D (plane) estimate
        var_per_obs = wt_3d*score_var(A) + wt_2d*score_var_2d(A)
        sigma2 = tf.multiply(num_obs, var_per_obs, name='sigma2')
        # sigma = tf.sqrt(sigma2, name='sigma')
        
        # Effective number of observations
        # eff_obs = raw_score - mu
        
        # The t-score for each sample
        # t_score = (raw_score - mu) / sigma
        
        # Assemble the objective function to be maximized
        # scale_factor = 1.0 / self.batch_size
        scale_factor = 1.0
        # sigma_power = 0.0
        # cdf = tf.nn.sigmoid(t_score)
        # quality_factor = 0.80*cdf + 0.20
        # objective = scale_factor * (raw_score - mu_factor * mu) * tf.math.pow(sigma, -sigma_power)
        objective = scale_factor * (raw_score - self.alpha * mu - self.beta * sigma2)

        # Add the loss function
        self.add_loss(-tf.reduce_sum(objective))

        # Return both the raw and t scores
        return raw_score, mu, sigma2, objective

    def get_config(self):
        return self.cfg

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
                               R_deg: float = 5.0,
                               q_cal = None):
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
    if q_cal is None:
        print(f'Numerically integrating orbits for calibration...')
        epoch0 = elts_np['epoch'][0]
        q_ast, q_sun, q_earth = calc_ast_pos(elts=elts_np, epoch=epoch0, ts=ts)
    else:
        q_ast, q_sun, q_earth = q_cal

    # Calibrate the direction prediction layer
    direction_layer.calibrate(elts=elts_np, q_ast=q_ast, q_sun=q_sun)
    # Tensor of predicted directions
    u_pred = direction_layer(a, e, inc, Omega, omega, f, epoch)

    # Difference in direction between u_obs and u_pred
    z = DirectionDifference(batch_size=elt_batch_size, 
                            traj_size=time_batch_size, 
                            max_obs=max_obs, 
                            name='z')(u_obs, u_pred, idx)
    
    # Calculate score compoments
    score_layer = TrajectoryScore(batch_size=elt_batch_size, alpha=1.0, beta=0.0)
    raw_score,  mu, sigma2, objective = score_layer(z, R, num_obs)
    # Stack the scores
    scores = tf.stack(values=[raw_score, mu, sigma2, objective], axis=1, name='scores')

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
    # model.add_loss(-tf.reduce_sum(objective))

    return model

# ********************************************************************************************************************* 
def perturb_elts(elts, sigma_a=0.05, sigma_e=0.10, sigma_f_deg=5.0, mask=None):
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
    sigma_f = np.deg2rad(sigma_f_deg)
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
def report_model_attribute(att: np.array, mask_good: np.array, att_name: str):
    """Report mean and stdev of a model attribute on good and bad masks"""
    mask_bad = ~mask_good
    # Attribute on masks
    att_g = att[mask_good]
    att_b = att[mask_bad]
    mean_g = np.mean(att_g, axis=0)
    mean_b = np.mean(att_b, axis=0)
    mean_a = np.mean(att, axis=0)
    std_g = np.std(att_g, axis=0)
    std_b = np.std(att_b, axis=0)
    std_a = np.std(att, axis=0)

    print(f'\nMean & std {att_name} by Category:')
    print(f'Good: {mean_g:8.2f} +/- {std_g:8.2f}')
    print(f'Bad:  {mean_b:8.2f} +/- {std_b:8.2f}')
    print(f'All:  {mean_a:8.2f} +/- {std_a:8.2f}')
    

# ********************************************************************************************************************* 
def report_model(model, R_deg, mask_good, batch_size, steps, elts_true, display=True):
    """Report summary of model on good and b"""

    mask_bad = ~mask_good
    loss_mean = model.evaluate(ds, steps=steps)
    loss_total = steps * loss_mean
    obj_per_elt = -loss_total / batch_size

    # Get scores on the whole data set
    pred = model.predict(ds, steps=steps)
    elts, R, u_pred, z, scores_all = pred

    # Consolidate results to batch_size rows
    num_rows = elts.shape[0]
    score_cols = scores_all.shape[1]
    row_idx = np.arange(num_rows, dtype=np.int32) % batch_size
    elts = elts[0:batch_size]
    R = R[0:batch_size]
    u_pred = u_pred[0:batch_size]

    # Consolidate the scores; create 2 extra columns for sigma and t_score
    scores = np.zeros((batch_size, score_cols+2))
    for batch_idx in range(batch_size):
        mask = (row_idx == batch_idx)
        scores[batch_idx, 0:score_cols] = scores_all[mask].sum(axis=0)

    # Unpock scores
    raw_score = scores[:,0]
    mu = scores[:,1]
    sigma2 = scores[:,2]
    objective = scores[:,3]

    # Compute derived scores after aggregation
    sigma = np.sqrt(sigma2)
    eff_obs = raw_score - mu
    t_score = eff_obs / sigma
    
    # Pack sigma and t_score at the end of scores
    scores[:, 4] = sigma
    scores[:, 5] = t_score
    
    # Change in resolution
    R0 = tf.ones_like(R) * np.deg2rad(R_deg)
    d_R = R - R0

    # Error in orbital elements
    elt_err = np.abs(elts - elts_true)
    # Mean element error on good and bad masks
    elt_err_g = elt_err[mask_good]
    elt_err_b = elt_err[mask_bad]
    mean_err_g = np.mean(elt_err_g[0:6], axis=0)
    mean_err_b = np.mean(elt_err_b[0:6], axis=0)

    if display:
        # Report errors in orbital elements
        # print(f'\nMean Loss per batch =  {loss_mean:8.0f}')
        print(f'\nMean Objective per elt = {obj_per_elt:8.0f}')
        print(f'\nError in orbital elements:')
        print(f'Good: {mean_err_g[0]:5.2e},  {mean_err_g[1]:5.2e}, {mean_err_g[2]:5.2e}, '
              f'{mean_err_g[0]:5.2e},  {mean_err_g[1]:5.2e}, {mean_err_g[2]:5.2e}, ')
        print(f'Bad : {mean_err_b[0]:5.2e},  {mean_err_b[1]:5.2e}, {mean_err_b[2]:5.2e}, '
              f'{mean_err_b[0]:5.2e},  {mean_err_b[1]:5.2e}, {mean_err_b[2]:5.2e}, ')
    
        # Report effective observations, mu, sigma, and t_score    
        # report_model_attribute(raw_score, mask_good, 'Raw Score')
        # report_model_attribute(mu, mask_good, 'Mu')
        report_model_attribute(eff_obs, mask_good, 'Effective Observations')
        # report_model_attribute(sigma, mask_good, 'Sigma')
        report_model_attribute(t_score, mask_good, 't_score')
        report_model_attribute(objective, mask_good, 'Objective Function')
        # report_model_attribute(d_R, mask_good, 'Change in resolution R')

    return scores, elt_err

# ********************************************************************************************************************* 
def report_training_progress(d_scores, d_elt_err, d_R):
    """Report progress while model trained"""
    # Unpack change in scores
    d_raw_score = d_scores[:,0]
    d_mu = d_scores[:, 1]
    # d_sigma2 = d_scores[:,2]
    d_objective = d_scores[:, 3]
    # d_sigma = d_scores[:, 4]
    d_t_score = d_scores[:,5]

    # Calculations
    d_eff_obs = d_raw_score - d_mu

    # report_model_attribute(d_raw_score, mask_good, 'Change in Raw Score')
    # report_model_attribute(d_eff_obs, mask_good, 'Change in Effective Observations')
    # report_model_attribute(d_t_score, mask_good, 'Change in t_score')
    report_model_attribute(d_objective, mask_good, 'Change in Objective Function')
    # report_model_attribute(d_R, mask_good, 'Change in resolution R')

    # Changes in element errors and R
    d_err_g = np.mean(d_elt_err[mask_good], axis=0)
    d_err_b = np.mean(d_elt_err[mask_bad], axis=0)
    
    #    print(f'\nChange in Orbital Element error by Category:')
    #    print(f'd_err_g: {d_err_g[0]:+5.2e},  {d_err_g[1]:+5.2e}, {d_err_g[2]:+5.2e}, '
    #          f'{d_err_g[3]:+5.2e},  {d_err_g[4]:+5.2e}, {d_err_g[5]:+5.2e}, ')
    #    print(f'd_err_b: {d_err_b[0]:+5.2e},  {d_err_b[1]:+5.2e}, {d_err_b[2]:+5.2e}, '
    #          f'{d_err_b[4]:+5.2e},  {d_err_b[5]:+5.2e}, {d_err_b[5]:+5.2e}, ')

# ********************************************************************************************************************* 
# Dataset of observations: synthetic data on first 1000 asteroids

# Build the dataset
n0: int = 1
n1: int = 100
dt0: datetime = datetime(2000,1,1)
dt1: datetime = datetime(2019,1,1)
time_batch_size = 128
ds, ts, row_len = make_synthetic_obs_dataset(n0=n0, n1=n1, dt0=dt0, dt1=dt1, batch_size=time_batch_size)
# Trajectory size and steps per batch
traj_size = ts.shape[0]
steps = int(np.ceil(traj_size / time_batch_size))

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
R_deg: float = 1.0
elts_np = orbital_element_batch(1)
epoch = elts_np['epoch'][0]
elt_batch_size = 64
# The correct orbital elements as an array
elts_true = np.array([elts_np['a'], elts_np['e'], elts_np['inc'], elts_np['Omega'], 
                      elts_np['omega'], elts_np['f'], elts_np['epoch']]).transpose()

# Mask where data expected vs not
mask_good = np.arange(64) < 32
mask_bad = ~mask_good
# Perturb second half of orbital elements
elts_np2 = perturb_elts(elts_np, sigma_a=0.01, sigma_e=0.0, sigma_f_deg=0.0, mask=mask_bad)

# Orbits for calibration
if 'q_cal' not in globals():
    q_cal = calc_ast_pos(elts=elts_np, epoch=epoch, ts=ts)

# Initialize model with perturbed orbital elements
model = make_model_asteroid_search(ts=ts,
                                   elts_np=elts_np2,
                                   max_obs=max_obs,
                                   num_obs=num_obs,
                                   elt_batch_size=elt_batch_size, 
                                   time_batch_size=time_batch_size,
                                   R_deg = R_deg,
                                   q_cal=q_cal)
# Use Adam optimizer with gradient clipping
# opt = keras.optimizers.Adam(learning_rate=1.0e-6, epsilon=1.0, amsgrad=True, clipvalue=1.0, )
# opt = keras.optimizers.SGD(learning_rate=1.0e-9, clipvalue=1.0E-9)
# opt = keras.optimizers.Adadelta(learning_rate=1.0e-4, rho=0.999, epsilon=1.0E-10, clipnorm=1.0)
# opt = keras.optimizers.Adadelta(learning_rate=1.0e-4, rho=0.999, epsilon=1.0E-10, clipnorm=1.0)
opt = keras.optimizers.Adamax(learning_rate=1.0e-6, beta_1=0.999, beta_2=0.999, 
                              epsilon=1.0E-7, clipnorm=1.0)
opt = keras.optimizers.Adam(learning_rate=1.0e-7, beta_1=0.900, beta_2=0.999,
                            epsilon=1.0E-7, amsgrad=False, clipnorm=1.0)
model.compile(optimizer=opt)

# Clone model before training
model0 = make_model_asteroid_search(ts=ts,
                                    elts_np=elts_np2,
                                    max_obs=max_obs,
                                    num_obs=num_obs,
                                    elt_batch_size=elt_batch_size, 
                                    time_batch_size=time_batch_size,
                                    R_deg = R_deg,
                                    q_cal=q_cal)
model0.compile(optimizer=opt)

# Set display for report_model
display = True

# Report losses before training
print(f'Processed first {n1} asteroids; seeded with first 64. Perturbed 33-65.')
print(f'\nModel before training:')
# loss0 = model.evaluate(ds, steps=steps)
pred0 = model.predict_on_batch(ds)
elts0, R0, u_pred0, z0, _ = pred0
scores0, elt_err0 = report_model(model=model, R_deg=R_deg, mask_good=mask_good, 
                                 batch_size=elt_batch_size, steps=steps, elts_true=elts_true, display=display)

# Get gradients on entire data set
with tf.GradientTape(persistent=True) as gt:
    gt.watch([model.elements.e_, model.elements.inc_, model.elements.R_])
    pred = model.predict_on_batch(ds.take(traj_size))
    # pred = model.predict_on_batch(ds.take(traj_size))
    elts, R, u_pred, z, scores = pred
    raw_score = scores[:, 0]
    mu = scores[:, 1]
    sigma2 = scores[:, 2]
    objective = scores[:, 3]
    loss = tf.reduce_sum(-objective)
dL_da = gt.gradient(loss, model.elements.a_) / steps
dL_de = gt.gradient(loss, model.elements.e_) / steps
dL_dinc = gt.gradient(loss, model.elements.inc_) / steps
# dL_dOmega = gt.gradient(loss, model.elements.Omega) / steps
# dL_domega = gt.gradient(loss, model.elements.omega) / steps
# dL_df = gt.gradient(loss, model.elements.f) / steps
dL_dR = gt.gradient(loss, model.elements.R_) / steps
del gt

# Train model
step_multiplier = 1
steps_per_epoch = steps*step_multiplier
steps_per_epoch = 1
epochs = 1
hist = model.fit(ds, epochs=epochs, steps_per_epoch=steps_per_epoch)
pred = model.predict_on_batch(ds)
elts, R, u_pred, z, scores = pred
elt_err = np.abs(elts - elts_true)

# Report results
print(f'\nModel after training:')
scores, elt_err = report_model(model=model, R_deg=R_deg, mask_good=mask_good, 
                               batch_size=elt_batch_size, steps=steps, elts_true=elts_true, display=display)
# Unpack scores after training
raw_score = scores[:,0]
mu = scores[:,1]
sigma2 = scores[:,2]
objective = scores[:,3]
sigma = scores[:, 4]
t_score = scores[:, 5]
eff_obs = raw_score - mu

## Change in scores
d_scores = scores - scores0
d_elt_err = elt_err - elt_err0
d_R = R - R0
# d_objective = d_scores[:,3]

# Report training progress: scores, orbital element errors, and resolution
print(f'\nProgress during training:')
report_training_progress(d_scores, d_elt_err, d_R)

# Change in elements
orb0_a = elts_np['a']
orb1_a = model.elements.get_a()
da = orb1_a - orb0_a

# err_a = np.abs(model.elements.get_a() - elts_np['a'])

orb0_a_ = model0.elements.a_
orb1_a_ = model.elements.a_
d_a_ = orb1_a_ - orb0_a_
