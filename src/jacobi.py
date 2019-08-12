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
from tf_utils import Identity
from orbital_element import make_data_orb_elt, G_

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
    
    # Initialize Cartesian position and velocity to zero
    q = np.zeros(pos_shape, dtype=np.float32)
    v = np.zeros(pos_shape, dtype=np.float32)
    # Initialize Jacobi position and velocity to zero
    qj = np.zeros(pos_shape, dtype=np.float32)
    vj = np.zeros(pos_shape, dtype=np.float32)
    
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
        v[:, j, :] = cart['v']
    
    # Cumulative mass
    M = np.cumsum(m, axis=-1)
    
    # Assemble num_body x num_body square matrix converting from q to r
    A = np.zeros(A_shape, dtype=np.float32)
    for n in range(N):
        A[n, 0, :] = m[n] / M[n, num_body-1]
        for i in range(1, num_body):
            for j in range(i):
                A[n, i, j] = -m[n, j] / M[n, i-1]
            A[n, i, i] = 1.0
    
    # Compute position and velocity in Jacobi coordinates
    qj = np.matmul(A, q)
    vj = np.matmul(A, v)
    
    # Gravitational coefficient mu
    mu = G_ * M
    
    data = {
        'm': m,
        'q': q,
        'v': v,
        'qj': qj,
        'vj': vj,
        'mu': mu
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
    v = data['v']
    qj = data['qj']
    vj = data['vj']
    mu = data['mu']
    
    # Inputs and outputs
    inputs = {'m': m, 'q': q, 'v': v}
    outputs = {'qj': qj, 'vj': vj, 'mu': mu}
    
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
    v = data['v']
    qj = data['qj']
    vj = data['vj']
    
    # Inputs and outputs
    inputs = {'m': m, 'qj': qj, 'vj': vj}
    outputs = {'q': q, 'v': v}

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
        # q: Cartesian position coordinate; shape (num_body, 3)
        # v: Cartesian velocity coordinate; shape (num_body, 3)
        m, q, v = inputs

        # Array shapes
        batch_size, num_body, space_dims = q.shape
        
        # Cumulative sum of mass
        M = tf.math.cumsum(m, axis=-1)
        M_tot = keras.layers.Reshape(target_shape=(1,))(M[:, num_body-1])
        
        # Assemble num_body x num_body square matrix converting from q to r
        A_rows = []
        # The first row is for the center of mass; A[0, j] = m[j] / M_tot
        A_rows.append(m/M_tot)
        # The next N-1 rows are for the N-1 Jacobi coordinates
        for i in range(1, num_body):
            # The first block of row i consists of i negative weights A[i, j] = -m[j] / M[i-1]
            # These subtract out the center of mass of the first (i-1) bodies
            block_1 = -m[:, 0:i] / M[:, i-1:i]
            # The second block is a 1 on the main diagonal, A[i, i] = 1.0
            block_2 = tf.ones(shape=(batch_size, 1))
            # The third block is zeroes; the A matrix is lower triangular below the first row
            block_3 = tf.zeros(shape=(batch_size, num_body-i-1))
            # Assemble row i
            # row_inputs = [block_1, block_2, block_3] if i < num_body-1 else [block_1, block_2]
            # row_inputs = [block_1, block_2, block_3]
            current_row = keras.layers.concatenate([block_1, block_2, block_3], axis=-1, name=f'A_row_{i}')
            A_rows.append(current_row)
        # Combine the rows into a matrix; this will have shape (batch_size, num_body^2)
        A = keras.layers.concatenate(A_rows, axis=-1, name='A_flat')
        # Reshape A to (batch_size, )
        A = keras.layers.Reshape(target_shape=(num_body,num_body,), name='A')(A)

        # Do the matrix multiplication
        qj = tf.linalg.matmul(A, q)
        vj = tf.linalg.matmul(A, v)

        # Compute the gravitational field strength mu for each Jacobi coordinate
        G = tf.constant(G_)
        mu = G * M

        return qj, vj, mu

    def get_config(self):
        return dict()
    
# ********************************************************************************************************************* 
class JacobiToCartesian(keras.layers.Layer):
    def call(self, inputs):
        """Compute Cartesian coordinate q from masses and Jacobi coordinates m, r"""
        # Unpack inputs
        # m: masses; shape (num_body)
        # qj: Jacobi position coordinate; shape (num_body, 3)
        # vj: Jacobi velocity coordinate; shape (num_body, 3)
        m, qj, vj = inputs

        # Array shapes
        batch_size, num_body, space_dims = qj.shape
        A_shape = (batch_size, num_body, num_body)
        
        # Cumulative sum of mass
        M = tf.math.cumsum(m, axis=-1)
        M_tot = keras.layers.Reshape(target_shape=(1,))(M[:, num_body-1])
        
        # Assemble num_body x num_body square matrix converting from q to r
        # Do the assembly as a numpy matrix
        A_ = np.zeros(A_shape, dtype=np.float32)
        A_[:, 0, :] = m / M_tot
        for i in range(1, num_body):
            A_[:, i, 0:i] = -m[:, 0:i] / M[:, i-1:i]
            A_[:, i, i] = 1.0

        # Now convert A to a tensor
        A = tf.Variable(A_)
        # Compute the matrix inverse of A
        B = tf.linalg.inv(A)
        # Do the matrix multiplication in Tensorflow
        q = tf.linalg.matmul(B, qj)
        v = tf.linalg.matmul(B, vj)

        return q, v

    def get_config(self):
        return dict()

# ********************************************************************************************************************* 
def make_model_cart_to_jac(num_body: int, batch_size: int = 64):
    """Model that transforms from cartesian to Jacobi coordinates"""
    # The shape shared by all the inputs
    space_dims = 3
    shape_m = (num_body,)
    shape_q = (num_body, space_dims,)

    # Create input layers    
    m = keras.Input(shape=shape_m, batch_size=batch_size, name='m')
    q = keras.Input(shape=shape_q, batch_size=batch_size, name='q')
    v = keras.Input(shape=shape_q, batch_size=batch_size, name='v')
    
    # Wrap these up into one tuple of inputs
    inputs = (m, q, v)

    # Calculations are in one layer that does all the work...
    qj, vj, mu = CartesianToJacobi(name='c2j')(inputs)
    
    # Name outputs
    qj = Identity(name='qj')(qj)
    vj = Identity(name='vj')(vj)
    mu = Identity(name='mu')(mu)
    
    # Wrap up the outputs
    outputs = (qj, vj, mu)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='cartesian_to_jacobi')
    return model

# ********************************************************************************************************************* 
def make_model_jac_to_cart(num_body: int, batch_size: int = 64):
    """Model that transforms from Jacobi to Cartesian coordinates"""
    # The shape shared by all the inputs
    space_dims = 3
    shape_m = (num_body,)
    shape_q = (num_body, space_dims,)

    # Create input layers    
    m = keras.Input(shape=shape_m, batch_size=batch_size, name='m')
    qj = keras.Input(shape=shape_q, batch_size=batch_size, name='qj')
    vj = keras.Input(shape=shape_q, batch_size=batch_size, name='vj')
    
    # Wrap these up into one tuple of inputs
    inputs = (m, qj, vj)

    # Calculations are in one layer that does all the work...
    qj, vj = JacobiToCartesian(name='j2c')(inputs)
    
    # Wrap up the outputs
    outputs = (qj, vj)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='jacobi_to_cartesian')
    return model
