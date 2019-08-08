"""
Harvard IACS Masters Thesis
Jacobi Coordinates

Michael S. Emanuel
Thu Aug  8 14:51:41 2019
"""

# Library imports
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Local imports
from orbital_element import make_data_orb_elt

# ********************************************************************************************************************* 
# Data sets for testing Jacobi coordinate conversions.
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_data_jacobi(N, num_body):
    """Data set mapping between Jacobi and Cartesian coordinates"""
    # Array shapes
    space_dims = 3
    mass_shape = (N, num_body)
    pos_shape = (N, num_body, space_dims)
    # matrix transforming Cartesian to Jacobi for each draw
    A_shape = (N, num_body, num_body)
    
    # Set random seed for reproducible results
    np.random.seed(seed=42)

    # Initialize masses
    np.random
    m_min = 1.0E-4
    m_max = 1.0E-2
    log_m = np.random.uniform(low=np.log(m_min), high=np.log(m_max), size=mass_shape).astype(np.float32)
    log_m[:, 0] = 0.0
    m = np.exp(log_m)
    
    # Initialize position q and Jacobi coordinate r to zero
    q = np.zeros(pos_shape, dtype=np.float32)
    r = np.zeros(pos_shape, dtype=np.float32)
    
    # Use make_data_orb_elt to populate q for each particle
    a_min = 0.5
    a_max = 32.0
    e_max = 0.20
    inc_max = 0.20
    seed = 42
    # Cartesian coordinates of body j
    for j in range(num_body):
        elts, cart = make_data_orb_elt(N, a_min, a_max, e_max, inc_max, seed+j)
        q[:, j, :] = cart['q']
    
    # Cumulative mass
    M = np.cumsum(m, axis=-1)
    
    # Assemble num_body x num_body square matrix converting from q to r
    A = np.zeros(A_shape)
    for n in range(N):
        A[n, 0, :] = m[n] / M[n, num_body-1]
        for i in range(1, num_body):
            for j in range(i):
                A[n, i, j] = -m[n, j] / M[n, i-1]
            A[n, i, i] = 1.0
        
    r = np.matmul(A, q)
    
    data = {
        'm': m,
        'q': q,
        'r': r,
        }
    
    return data

# ********************************************************************************************************************* 
def make_dataset_cart_to_jac(N, num_body, batch_size=64):
    """Dataset with mapping from Cartesian to Jacobi coordinates"""
    # Delegate to make_data_jacobi
    data = make_data_jacobi(N, num_body)

    # Unpack the data
    m = data['m']
    q = data['q']
    r = data['r']
    
    # Inputs and outputs
    inputs = {'m': m, 'q': q}
    outputs = {'r': r}
    
     # Wrap these into a Dataset object
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))

    # Set shuffle buffer size
    buffer_size = batch_size * 256

    # Shuffle and batch data sets
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    return ds

# ********************************************************************************************************************* 
def make_dataset_jac_to_cart(N, num_body, batch_size=64):
    """Dataset with mapping from Jacobi to Cartesian coordinates"""
    # Delegate to make_data_jacobi
    data = make_data_jacobi(N, num_body)

    # Unpack the data
    m = data['m']
    q = data['q']
    r = data['r']
    
    # Inputs and outputs
    inputs = {'m': m, 'r': q}
    outputs = {'q': r}
    
     # Wrap these into a Dataset object
    ds = tf.data.Dataset.from_tensor_slices((inputs, outputs))

    # Set shuffle buffer size
    buffer_size = batch_size * 256

    # Shuffle and batch data sets
    ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size)
    
    return ds

# ********************************************************************************************************************* 
# Custom layers for converting between configurations (position and velocity) and orbital elements.
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
class CartesianToJacobi(keras.layers.Layer):
    def call(self, inputs):
        """Compute Cartesian coordinate q from masses and Jacobi coordinates m, r"""
        # Unpack inputs
        # m: masses; shape (num_body)
        # q: Cartesian coordinate; shape (num_body, 3)
        m, q = inputs

        # Array shapes
        batch_size, num_body, space_dims = q.shape
        A_shape = (batch_size, num_body, num_body)
        
        # Cumulative sum of mass
        M = tf.math.cumsum(m, axis=-1)
        M_tot = keras.layers.Reshape(target_shape=(1,))(M[:, num_body-1])
        
        # Assemble num_body x num_body square matrix converting from q to r
        # Do the assembly as a numpy matrix
        A_ = np.zeros(A_shape, dtype=np.float32)
        A_[:, 0, :] = m / M_tot
        for i in range(1, num_body):
            for j in range(i):
                A_[:, i, j] = -m[:, j] / M[:, i-1]
            A_[:, i, i] = 1.0

        # Now convert A to a tensor
        A = tf.Variable(A_)
        r = tf.linalg.matmul(A, q)
        
        return r

    def get_config(self):
        return dict()
# ********************************************************************************************************************* 
class JacobiToCartesian(keras.layers.Layer):
    def call(self, inputs):
        """Compute Jacobi coordinate r from masses and Cartesian coordinates m, q"""
        # Unpack inputs
        # m: masses; shape (num_body)
        # r: Jacobi coordinate; shape (num_body, 3)
        m, r = inputs

        # Cumulative sum of mass
        M = tf.math.cumsum(m, axis=-1)
        # 
        r = q
        
        return r

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
def make_model_cart_to_jac(num_body):
    """Model that transforms from cartesian to Jacobi coordinates"""
    # The shape shared by all the inputs
    space_dims = 3
    shape_m = (num_body,)
    shape_q = (num_body, space_dims,)

    # Create input layers    
    m = keras.Input(shape=shape_m, name='m')
    q = keras.Input(shape=shape_q, name='q')
    
    # Wrap these up into one tuple of inputs
    inputs = (m, q)

    # Calculations are in one layer that does all the work...
    r = CartesianToJacobi(name='r')(inputs)
    
    # Wrap up the outputs
    outputs = r

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='cartesian_to_jacobi')
    return model

# ********************************************************************************************************************* 
def make_model_jac_to_cart(num_body):
    """Model that transforms from Jacobi to Cartesian coordinates"""
    # The shape shared by all the inputs
    space_dims = 3
    shape_m = (num_body,)
    shape_r = (num_body, space_dims,)

    # Create input layers    
    m = keras.Input(shape=shape_m, name='m')
    r = keras.Input(shape=shape_r, name='r')
    
    # Wrap these up into one tuple of inputs
    inputs = (m, r)

    # Calculations are in one layer that does all the work...
    q = CartesianToJacobi(name='q')(inputs)
    
    # Wrap up the outputs
    outputs = q

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='cartesian_to_jacobi')
    return model
