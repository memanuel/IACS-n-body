"""
Harvard IACS Masters Thesis
Jacobi Coordinates

Michael S. Emanuel
Thu Aug  8 14:51:41 2019
"""

# Library imports
import rebound
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
def make_data_jacobi_one(m, a, e, inc, Omega, omega, f):
    """
    Make one data point mapping between Cartesian and Jacobi coordinates
    INPUTS:
        m: array of num_body masses
        a, e, inc, Omega, omega, f:
            arrays of six standard orbital elements.
            num_body objects have (num_body-1) orbital elements
    """
    # The number of bodies
    num_body = m.shape[0]

    # Array shapes
    space_dims = 3
    pos_shape = (num_body, space_dims)
    mu_shape = (num_body,)
    
    # Initialize Cartesian position and velocity to zero
    q = np.zeros(pos_shape, dtype=np.float32)
    v = np.zeros(pos_shape, dtype=np.float32)
    # Initialize Jacobi position and velocity to zero
    qj = np.zeros(pos_shape, dtype=np.float32)
    vj = np.zeros(pos_shape, dtype=np.float32)
    # Initialize gravitational field strength 
    mu = np.zeros(mu_shape, dtype=np.float32)
    
    # Convert m to single
    m = m.astype(np.float32)
    
    # Create a simulation
    sim = rebound.Simulation()
    
    # Set units
    sim.units = ('yr', 'AU', 'Msun')
    
    # Add primary with specified mass at origin with 0 velocity
    sim.add(m=m[0])
    
    # The current center of mass
    com = sim.calculate_com()
    mu[0] = sim.G * com.m
    
    # Add bodies 2 through num_body
    for i in range(num_body-1):
        # Add body i+1 with Jacobi coordinates indexed by i and mass i+1
        sim.add(m=m[i+1], a=a[i], e=e[i], inc=inc[i], Omega=Omega[i], omega=omega[i], f=f[i])
        # The body that was just added
        p = sim.particles[i+1]
        # Jacobi coordinates of this body are its coordinates less the cumulative center of mass
        qj[i+1, :] = [p.x -  com.x,  p.y - com.y,   p.z  - com.z]    
        vj[i+1, :] = [p.vx - com.vx, p.vy - com.vy, p.vz - com.vz]    
        # Update the center of mass
        com = sim.calculate_com()
        # Gravitational field strength; mass includes the newly added point
        mu[i+1] = sim.G * com.m
    
    # Move to center of mass
    sim.move_to_com()
    
    # The first Jacobi coordinate in row 0 is the position and velocity of the COM
    # This has been set to zero above, so don't need to change it from initialization.
    
    # Save the Cartesian coordinates in the COM frame
    for i in range(num_body):
        pi = sim.particles[i]
        q[i, :] = [pi.x, pi.y, pi.z]
        v[i, :] = [pi.vx, pi.vy, pi.vz]
        
    # Create data dictionary
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
def make_data_jacobi(N, num_body):
    """Make a data set mapping between Jacobi and Cartesian coordinates"""
    # Array shapes
    space_dims = 3
    mass_shape = (N, num_body)
    pos_shape = (N, num_body, space_dims)
    elt_shape = (N, num_body-1)
    
    # Set random seed for reproducible results
    seed = 42
    np.random.seed(seed=seed)

    # Initialize masses
    np.random
    m_min = 1.0E-4
    m_max = 1.0E-2
    log_m = np.random.uniform(low=np.log(m_min), high=np.log(m_max), size=mass_shape).astype(np.float32)
    log_m[:, 0] = 0.0
    m_ = np.exp(log_m)
    
    # Parameters for sampling orbital elements in testing
    a_min = 0.5
    a_max = 32.0
    e_max = 0.20
    inc_max = 0.20
    
    # Initialize orbital element by sampling according to the inputs    
    a = np.random.uniform(low=a_min, high=a_max, size=elt_shape).astype(np.float32)
    e = np.random.uniform(low=0.0, high=e_max, size=elt_shape).astype(np.float32)
    inc = np.random.uniform(low=0.0, high=inc_max, size=elt_shape).astype(np.float32)
    Omega = np.random.uniform(low=-np.pi, high=np.pi, size=elt_shape).astype(np.float32)
    omega = np.random.uniform(low=-np.pi, high=np.pi, size=elt_shape).astype(np.float32)
    f = np.random.uniform(low=-np.pi, high=np.pi, size=elt_shape).astype(np.float32)
    
    # Initialize Cartesian position and velocity to zero   
    q = np.zeros(pos_shape, dtype=np.float32)
    v = np.zeros(pos_shape, dtype=np.float32)
    # Initialize Jacobi position and velocity to zero
    qj = np.zeros(pos_shape, dtype=np.float32)
    vj = np.zeros(pos_shape, dtype=np.float32)
    # Initialize mass and mu to zero
    m = np.zeros(mass_shape, dtype=np.float32)
    mu = np.zeros(mass_shape, dtype=np.float32)

    # Generate mappings one draw at a time
    for i in range(N):
        # Generate one data set with row i of the mass and orbital elements drawn
        data_i = make_data_jacobi_one(m=m_[i], a=a[i], e=e[i], inc=inc[i], 
                                      Omega=Omega[i], omega=omega[i], f=f[i])
        # Unpack this draw to row i of the data arrays
        m[i] = data_i['m']
        q[i] = data_i['q']
        v[i] = data_i['v']
        qj[i] = data_i['qj']
        vj[i] = data_i['vj']
        mu[i] = data_i['mu']

    # Wrap the data into a dict
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
def make_dataset_cart_to_cart(N, num_body, batch_size=64):
    """Dataset with mapping from Cartesian to Cartesian coordinates"""
    # Delegate to make_data_jacobi
    data = make_data_jacobi(N, num_body)

    # Unpack the data
    m = data['m']
    q = data['q']
    v = data['v']
    
    # Inputs and outputs
    inputs = {'m': m, 'q': q, 'v': v}
    outputs = {'q_calc': q, 'v_calc': v}

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
    def __init__(self, include_accel:bool = False, **kwargs):
        super(CartesianToJacobi, self).__init__(**kwargs)
        self.include_accel = include_accel

    def call(self, inputs):
        """Compute Cartesian coordinate q from masses and Jacobi coordinates m, r"""
        # Unpack inputs
        # m: masses; shape (num_body)
        # q: Cartesian position coordinate; shape (num_body, 3)
        # v: Cartesian velocity coordinate; shape (num_body, 3)
        if self.include_accel:
            m, q, v, a = inputs
        else:
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
        aj = tf.linalg.matmul(A, a) if self.include_accel else None

        # Compute the gravitational field strength mu for each Jacobi coordinate
        G = tf.constant(G_)
        mu = G * M
        
        # Assemble the outputs
        outputs = (qj, vj, aj, mu) if self.include_accel else (qj, vj, mu)

        return outputs

    def get_config(self):
        return dict()
    
# ********************************************************************************************************************* 
class JacobiToCartesian(keras.layers.Layer):
    def __init__(self, include_accel:bool = False, **kwargs):
        super(JacobiToCartesian, self).__init__(**kwargs)
        self.include_accel = include_accel

    def call(self, inputs):
        """Compute Cartesian coordinate q from masses and Jacobi coordinates m, r"""
        # Unpack inputs
        # m: masses; shape (num_body)
        # qj: Jacobi position coordinate; shape (num_body, 3)
        # vj: Jacobi velocity coordinate; shape (num_body, 3)
        if self.include_accel:
            m, qj, vj, aj = inputs
        else:
            m, qj, vj = inputs

        # Array shapes
        tensor_rank = len(qj.shape)
        if tensor_rank == 3:
            batch_size, num_body, space_dims = qj.shape
        elif tensor_rank == 4:
            batch_size, traj_size, num_body, space_dims = qj.shape
        else:
            raise ValueError('tensor_rank of qj must be 3 or 4.')
        
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
            current_row = keras.layers.concatenate([block_1, block_2, block_3], axis=-1, name=f'A_row_{i}')
            A_rows.append(current_row)
        # Combine the rows into a matrix; this will have shape (batch_size, num_body^2)
        A = keras.layers.concatenate(A_rows, axis=-1, name='A_flat')

        # Reshape A to either (batch_size, num_body, num_body) or (batch_size, 1, num_body, num_body)
        # Number of paddings (1,) entries is tensor_rank-3
        target_shape = (tensor_rank-3)*(1,) + (num_body, num_body)
        A = keras.layers.Reshape(target_shape=target_shape, name='A')(A)

        # Compute the matrix inverse of A
        B = tf.linalg.inv(A)
        
        # Do the matrix multiplication
        q = tf.linalg.matmul(B, qj)
        v = tf.linalg.matmul(B, vj)
        a = tf.linalg.matmul(B, aj) if self.include_accel else None

        # Assemble the outputs
        outputs = (q, v, a) if self.include_accel else (q, v)

        return outputs

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
    q, v = JacobiToCartesian(name='j2c')(inputs)
    
    # Name outputs
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)

    # Wrap up the outputs
    outputs = (q, v)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='jacobi_to_cartesian')
    return model

# ********************************************************************************************************************* 
def make_model_cart_to_cart(num_body: int, batch_size: int = 64):
    """Model that transforms from cartesian to cartesian coordinates"""
    # The shape shared by all the inputs
    space_dims = 3
    shape_m = (num_body,)
    shape_q = (num_body, space_dims,)

    # Create input layers    
    m = keras.Input(shape=shape_m, batch_size=batch_size, name='m')
    q = keras.Input(shape=shape_q, batch_size=batch_size, name='q')
    v = keras.Input(shape=shape_q, batch_size=batch_size, name='v')
    
    # Wrap these up into inputs
    inputs = (m, q, v)

    # Wrap these up into one tuple of inputs for c2j mapping
    inputs_c2j = inputs

    # Convert from Cartesian to Jacobi coordinates in step 1
    qj, vj, mu = CartesianToJacobi(name='c2j')(inputs_c2j)
    
    # Name Jacobi coordinates
    qj = Identity(name='qj')(qj)
    vj = Identity(name='vj')(vj)
    mu = Identity(name='mu')(mu)
    
    # Wrap these up into one tuple of inputs for j2c mapping
    inputs_j2c = (m, qj, vj)

    # Calculations are in one layer that does all the work...
    q_calc, v_calc = JacobiToCartesian(name='j2c')(inputs_j2c)
    
    # Name outputs
    q_calc = Identity(name='q_calc')(q_calc)
    v_calc = Identity(name='v_calc')(v_calc)

    # Wrap up the outputs
    outputs = (q_calc, v_calc)

    # Create a model from inputs to outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name='cartesian_to_cartesian')
    return model

