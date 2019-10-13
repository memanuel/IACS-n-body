"""
Harvard IACS Masters Thesis
Solar Test: Predict the movement of a test particle in the solar system
using the Kepler approximation with the sun as a fixed central attractor.

Michael S. Emanuel
Sun Oct 13 11:56:50 2019
"""

# Library imports
import tensorflow as tf
# import rebound

# Local imports
from tf_utils import Identity
from orbital_element import make_model_elt_to_pos, MeanToTrueAnomaly
from asteroid_data import make_dataset_ast_pos, make_dataset_ast_dir, get_earth_pos

# Aliases
keras = tf.keras

# The gravitational constant in ('day', 'AU', 'Msun') coordinates
# sim = rebound.Simulation()
# sim.units = ('day', 'AU', 'Msun')
# G_ = sim.G
G_ = 0.00029591220828559104

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_model_ast_pos(traj_size:int =14976, batch_size:int =64) -> keras.Model:
    """
    Compute orbit positions for asteroids in the solar system from
    the initial orbital elements with the Kepler model.
    Factory function that returns a functional model.
    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
    Outputs of the model are the position of the asteroid relative to the sun.
    INPUTS;
        traj_size: size of the trajectory; defulat 14976 is 40 years sampled daily
        batch_size: defaults to 64
    """
    # Adjust batch size for the number of GPUs
    # batch_size=batch_size*num_gpus
    
    # Inputs: 6 orbital elements; epoch; ts (output times as MJD)
    a = keras.Input(shape=(), batch_size=batch_size, name='a')
    e = keras.Input(shape=(), batch_size=batch_size, name='e')
    inc = keras.Input(shape=(), batch_size=batch_size, name='inc')
    Omega = keras.Input(shape=(), batch_size=batch_size, name='Omega')
    omega = keras.Input(shape=(), batch_size=batch_size, name='omega')
    f = keras.Input(shape=(), batch_size=batch_size, name='f')
    epoch = keras.Input(shape=(), batch_size=batch_size, name='epoch')
    ts = keras.Input(shape=(traj_size,), batch_size=batch_size, name='ts')

    # Wrap these up into one tuple of inputs for the model
    inputs = (a, e, inc, Omega, omega, f, epoch, ts)
    
    # The gravitational field strength
    mu = tf.constant(G_)
    
    # Compute eccentric anomaly E from f and e
    # https://en.wikipedia.org/wiki/Eccentric_anomaly
    cos_f = tf.cos(f)
    denom = 1.0 + e * cos_f
    cos_E = (e + cos_f) / denom
    sin_f = tf.sin(f)
    sqrt_one_m_e2 = tf.sqrt(1.0 - tf.square(e))
    sin_E = (sqrt_one_m_e2 * sin_f) / denom
    E = tf.atan2(y=sin_E, x=cos_E)
    
    # Compute mean anomaly M from E using Kepler's Equation
    # https://en.wikipedia.org/wiki/Mean_anomaly
    M = E - e * tf.sin(E)
    
    # Compute mean motion N from mu and a
    a3 = a * a * a
    N = tf.sqrt(mu / a3)
    
    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(ts)
    # Reshape epoch to (batch_size, traj_size, 1)
    epoch_vec = keras.layers.RepeatVector(n=traj_size, name='epoch_vec')(keras.backend.reshape(epoch, (batch_size,1)))
    
    # Subtract epoch from t_vec; now it is relative to the epoch
    t_vec = t_vec - epoch_vec
    
    # ******************************************************************
    # Predict orbital elements over time
    
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    target_shape = (batch_size, 1)
    a_t = keras.layers.RepeatVector(n=traj_size, name='a_t')(keras.backend.reshape(a, target_shape))
    e_t = keras.layers.RepeatVector(n=traj_size, name='e_t')(keras.backend.reshape(e, target_shape))
    inc_t = keras.layers.RepeatVector(n=traj_size, name='inc_t')(keras.backend.reshape(inc, target_shape))
    Omega_t = keras.layers.RepeatVector(n=traj_size, name='Omega_t')(keras.backend.reshape(Omega, target_shape))
    omega_t = keras.layers.RepeatVector(n=traj_size, name='omega_t')(keras.backend.reshape(omega, target_shape))
    # Repeat the gravitational field strength to vector of shape (batch_size, traj_size)
    mu_t = keras.layers.RepeatVector(n=traj_size, name='mu_t')(keras.backend.reshape(mu, (1, 1)))
    
    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M0_t = keras.layers.RepeatVector(n=traj_size, name='M0_t')(keras.backend.reshape(M, target_shape))
    N0_t = keras.layers.RepeatVector(n=traj_size, name='N0_t')(keras.backend.reshape(N, target_shape))
    # Compute the mean anomaly M(t) as a function of time
    N_mult_t = keras.layers.multiply(inputs=[N0_t, t_vec])
    M_t = keras.layers.add(inputs=[M0_t, N_mult_t])

    # Compute the true anomaly from the mean anomly and eccentricity
    f_t = MeanToTrueAnomaly(name='mean_to_true_anomaly')([M_t, e_t])

    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    elt_t = (a_t, e_t, inc_t, Omega_t, omega_t, f_t, mu_t,)
    
    # ******************************************************************
    # Convert orbital elements to cartesian coordinates 
    
    # Model mapping orbital elements to cartesian coordinates
    model_e2c = make_model_elt_to_pos(batch_size=batch_size)

    # Convert from orbital elements to cartesian coordinates (position only)
    q = model_e2c(elt_t)
    
    # Name the outputs
    q = Identity(name='q')(q)
    # v = Identity(name='v')(v)
    
    # Wrap up the outputs
    # outputs = (q,)

    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=q, name='model_asteroid_pos')
    return model

# Take a one time snapshot of the earth's position
q_earth_np = get_earth_pos()
print(f'q_earth_np loaded, shape = {q_earth_np.shape}')
traj_size = q_earth_np.shape[0]
space_dims=3
#q_earth = keras.backend.constant(q_earth_np.reshape(1, traj_size, space_dims), 
#                      dtype=tf.float32,
#                      shape=(1, traj_size, space_dims),
#                      name='q_earth')
# print(f'q_earth tf constant created, shape = {q_earth.shape}')
q_earth = keras.backend.variable(q_earth_np.reshape(1, traj_size, space_dims), 
                         dtype=tf.float32,
                         name='q_earth')
print(f'q_earth tf variable created, shape = {q_earth.shape}')

# ********************************************************************************************************************* 
def make_model_ast_dir(traj_size:int =14976, batch_size:int =64) -> keras.Model:
    """
    Compute direction from earth to asteroids in the solar system from
    the initial orbital elements with the Kepler model.
    Factory function that returns a functional model.
    Inputs for the model are 6 orbital elements, the epoch, and the desired times for position outputs.
    Outputs of the model are the unit vector (direction) pointing from earth to the asteroid
    INPUTS;
        traj_size: size of the trajectory; defulat 14976 is 40 years sampled daily
        batch_size: defaults to 64
    """
    space_dims = 3
    # Adjust batch size for the number of GPUs
    # batch_size=batch_size*num_gpus
    
    # Inputs: 6 orbital elements; epoch; ts (output times as MJD)
    a = keras.Input(shape=(), batch_size=batch_size, name='a')
    e = keras.Input(shape=(), batch_size=batch_size, name='e')
    inc = keras.Input(shape=(), batch_size=batch_size, name='inc')
    Omega = keras.Input(shape=(), batch_size=batch_size, name='Omega')
    omega = keras.Input(shape=(), batch_size=batch_size, name='omega')
    f = keras.Input(shape=(), batch_size=batch_size, name='f')
    epoch = keras.Input(shape=(), batch_size=batch_size, name='epoch')
    ts = keras.Input(shape=(traj_size,), batch_size=batch_size, name='ts')

    # Wrap these up into one tuple of inputs for the model
    inputs = (a, e, inc, Omega, omega, f, epoch, ts)
    
    # Model with asteroid position
    model_ast_pos = make_model_ast_pos(traj_size=traj_size, batch_size=batch_size)
    # Get the asteroid position with this model
    q = model_ast_pos(inputs)

    # Relative displacement from earth to asteroid
    # q_rel = q - q_earth
    q_rel = tf.subtract(q, q_earth, name='q_rel')
    # Distance to earth
    # r_earth = tf.norm(q_rel, axis=-1, keepdims=True, name='r_earth')
    # Unit vector pointing from earth to asteroid
    # u = tf.divide(q_rel, r_earth, name='q_rel_over_r_earth')
    u = q_rel
    
    # Wrap this into a model
    model = keras.Model(inputs=inputs, outputs=u, name='model_asteroid_dir')
    return model

# ********************************************************************************************************************* 
def test_ast_pos() -> bool:
    """Test asteroid position model"""
    # Load data for the first 1000 asteroids
    ds: tf.data.Dataset = make_dataset_ast_pos(0, 1)
    # Create the model to predict asteroid trajectories
    model: keras.Model = make_model_ast_pos()
    # Compile with MSE (mean squared error) loss
    model.compile(loss='MSE')
    # Evaluate this model
    mse: float = model.evaluate(ds)
    # Threshold for passing
    thresh: float = 0.02
    isOK: bool = (mse < thresh)
    # Report results
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'MSE for asteroid model on first 1000 asteroids = {mse:8.6f}')
    print(f'***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
def test_ast_dir() -> bool:
    """Test the asteroid direction moddel"""
    # Load data for the first 1000 asteroids
    ds: tf.data.Dataset = make_dataset_ast_dir(0, 1)
    # Create the model to predict asteroid trajectories
    model: keras.Model = make_model_ast_dir()
    # Compile with MSE (mean squared error) loss
    model.compile(loss='MSE')
    # Evaluate this model
    mse: float = model.evaluate(ds)
    # Threshold for passing
    thresh: float = 0.02
    isOK: bool = (mse < thresh)
    # Report results
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'MSE for asteroid model on first 1000 asteroids = {mse:8.6f}')
    print(f'***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
def main():
    # test_ast_pos()
    test_ast_dir()
    
# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()
