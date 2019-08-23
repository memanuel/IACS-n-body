"""
Harvard IACS Masters Thesis
General Three Body Problem
Models for general three body problem - neural network

Michael S. Emanuel
Fri Aug 16 16:11:28 2019
"""

# Library imports
import tensorflow as tf
import numpy as np

# Aliases
keras = tf.keras

# Local imports
from tf_utils import Identity
from orbital_element import make_model_cfg_to_elt, make_model_elt_to_cfg
from orbital_element import MeanToTrueAnomaly
from jacobi import CartesianToJacobi, JacobiToCartesian
from g3b import make_physics_model_g3b

# ********************************************************************************************************************* 
# Functional API Models
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def make_position_model_g3b_nn(hidden_sizes, skip_layers=True, 
                               kernel_reg=0.0, activity_reg=0.0, 
                               traj_size = 1001, batch_size=64):
    """
    Compute orbit positions for the general three body problem using a neural
    network that computes adjustments to orbital elements.
    Factory function that returns a functional model.
    """
    # Dimensionality
    num_body = 3
    space_dims = 3
    
    # Input layers
    t = keras.Input(shape=(traj_size,), batch_size=batch_size, name='t')
    q0 = keras.Input(shape=(num_body, space_dims,), batch_size=batch_size, name='q0')
    v0 = keras.Input(shape=(num_body, space_dims,), batch_size=batch_size, name='v0')
    m = keras.Input(shape=(num_body,), batch_size=batch_size, name='m')

    # Wrap these up into one tuple of inputs for the model
    inputs = (t, q0, v0, m)

    # Compute the Jacobi coordinates of the initial conditions
    qj0, vj0, mu0 = CartesianToJacobi()([m, q0, v0])

    # Extract Jacobi coordinates of p1 and p2
    qj0_1 = qj0[:, 1, :]
    qj0_2 = qj0[:, 2, :]
    vj0_1 = vj0[:, 1, :]
    vj0_2 = vj0[:, 2, :]
    
    # Extract gravitational field strength for orbital element conversion of p1 and p2
    mu0_1 = mu0[:, 1:2]
    mu0_2 = mu0[:, 2:3]

    # Manually set the shapes to work around documented bug on slices losing shape info
    jacobi_shape = (batch_size, space_dims)
    qj0_1.set_shape(jacobi_shape)
    qj0_2.set_shape(jacobi_shape)
    vj0_1.set_shape(jacobi_shape)
    vj0_1.set_shape(jacobi_shape)
    mu_shape = (batch_size, 1)
    mu0_1.set_shape(mu_shape)
    mu0_2.set_shape(mu_shape)
    
    # Tuple of inputs for the model converting from configuration to orbital elements
    cfg_1 = (qj0_1, vj0_1, mu0_1)
    cfg_2 = (qj0_2, vj0_2, mu0_2)

    # Model mapping cartesian coordinates to orbital elements
    model_c2e_1 = make_model_cfg_to_elt(name='orbital_element_1')
    model_c2e_2 = make_model_cfg_to_elt(name='orbital_element_2')

    # Extract the orbital elements of the initial conditions
    a1_0, e1_0, inc1_0, Omega1_0, omega1_0, f1_0, M1_0, N1_0 = model_c2e_1(cfg_1)
    a2_0, e2_0, inc2_0, Omega2_0, omega2_0, f2_0, M2_0, N2_0 = model_c2e_2(cfg_2)

    # Alias mu0_i for naming consistency
    mu1_0 = mu0_1
    mu2_0 = mu0_2

    # Reshape t to (batch_size, traj_size, 1)
    t_vec = keras.layers.Reshape(target_shape=(traj_size, 1), name='t_vec')(t)

    # ******************************************************************
    # Kepler-Jacobi Model: Analytical approximation ignoring interaction of two small bodies
    # ******************************************************************

    # ******************************************************************
    # Predict orbital elements for Jacobi coordinates of body 1

    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a1 = keras.layers.RepeatVector(n=traj_size, name='a1_kj')(a1_0)
    e1 = keras.layers.RepeatVector(n=traj_size, name='e1_kj')(e1_0)
    inc1 = keras.layers.RepeatVector(n=traj_size, name='inc1_kj')(inc1_0)
    Omega1 = keras.layers.RepeatVector(n=traj_size, name='Omega1_kj')(Omega1_0)
    omega1 = keras.layers.RepeatVector(n=traj_size, name='omega1_kj')(omega1_0)
    mu1 = keras.layers.RepeatVector(n=traj_size, name='mu1')(mu1_0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M1_0_vec = keras.layers.RepeatVector(n=traj_size, name='M1_0_vec')(M1_0)
    N1_0_vec = keras.layers.RepeatVector(n=traj_size, name='N1_0_vec')(N1_0)
    # Compute the mean anomaly M(t) as a function of time
    N1_t = keras.layers.multiply(inputs=[N1_0_vec, t_vec])
    M1 = keras.layers.add(inputs=[M1_0_vec, N1_t])

    # Compute the true anomaly from the mean anomaly and eccentricity
    f1 = MeanToTrueAnomaly(name='mean_to_true_anomaly_f1')([M1, e1])

    # ******************************************************************
    # Predict orbital elements for Jacobi coordinates of body 2 
    # Repeat the constant orbital elements to be vectors of shape (batch_size, traj_size)
    a2 = keras.layers.RepeatVector(n=traj_size, name='a2_kj')(a2_0)
    e2 = keras.layers.RepeatVector(n=traj_size, name='e2_kj')(e2_0)
    inc2 = keras.layers.RepeatVector(n=traj_size, name='inc2_kj')(inc2_0)
    Omega2 = keras.layers.RepeatVector(n=traj_size, name='Omega2_kj')(Omega2_0)
    omega2 = keras.layers.RepeatVector(n=traj_size, name='omega2_kj')(omega2_0)
    mu2 = keras.layers.RepeatVector(n=traj_size, name='mu2')(mu2_0)

    # Repeat initial mean anomaly M0 and mean motion N0 to match shape of outputs
    M2_0_vec = keras.layers.RepeatVector(n=traj_size, name='M2_0_vec')(M2_0)
    N2_0_vec = keras.layers.RepeatVector(n=traj_size, name='N2_0_vec')(N2_0)
    # Compute the mean anomaly M(t) as a function of time
    N2_t = keras.layers.multiply(inputs=[N2_0_vec, t_vec])
    M2 = keras.layers.add(inputs=[M2_0_vec, N2_t])

    # Compute the true anomaly from the mean anomaly and eccentricity
    f2 = MeanToTrueAnomaly(name='mean_to_true_anomaly_f2')([M2, e2])

    # ******************************************************************
    # Feature extraction: masses & cos/sin of angle variables
    # ******************************************************************

    # Extract masses of p1 and p2
    m1 = m[:, 1:2]
    m2 = m[:, 2:3]
    # Manually set the shapes to work around documented bug on slices losing shape info
    mass_shape = (batch_size, 1)
    m1.set_shape(mass_shape)
    m2.set_shape(mass_shape)

    # Repeat the masses to shape (batch_size, traj_size, num_body-1)
    # skip mass of body 0 because it is a constant = 1.0 solar mass
    m1 =  keras.layers.RepeatVector(n=traj_size, name='m1')(m1)
    m2 =  keras.layers.RepeatVector(n=traj_size, name='m2')(m2)
    
    # Repeat the initial true anomaly f1_0 and f2_0
    f1_0_vec = keras.layers.RepeatVector(n=traj_size, name='f1_0_vec')(f1_0)
    f2_0_vec = keras.layers.RepeatVector(n=traj_size, name='f2_0_vec')(f2_0)    

    # Convert inc1 and inc2 to cosine and sine
    cos_inc1 = keras.layers.Activation(activation=tf.cos, name='cos_inc1')(inc1)
    sin_inc1 = keras.layers.Activation(activation=tf.sin, name='sin_inc1')(inc1)
    cos_inc2 = keras.layers.Activation(activation=tf.cos, name='cos_inc2')(inc2)
    sin_inc2 = keras.layers.Activation(activation=tf.sin, name='sin_inc2')(inc2)
    
    # Convert Omega1 and Omega2 to cosine and sine
    cos_Omega1 = keras.layers.Activation(activation=tf.cos, name='cos_Omega1')(Omega1)
    sin_Omega1 = keras.layers.Activation(activation=tf.sin, name='sin_Omega1')(Omega1)
    cos_Omega2 = keras.layers.Activation(activation=tf.cos, name='cos_Omega2')(Omega2)
    sin_Omega2 = keras.layers.Activation(activation=tf.sin, name='sin_Omega2')(Omega2)
    
    # Convert omega1 and omega2 to cosine and sine
    cos_omega1 = keras.layers.Activation(activation=tf.cos, name='cos_omega1')(omega1)
    sin_omega1 = keras.layers.Activation(activation=tf.sin, name='sin_omega1')(omega1)
    cos_omega2 = keras.layers.Activation(activation=tf.cos, name='cos_omega2')(omega2)
    sin_omega2 = keras.layers.Activation(activation=tf.sin, name='sin_omega2')(omega2)
    
    # Convert f1 and f2 to cosine and sine
    cos_f1_0 = keras.layers.Activation(activation=tf.cos, name='cos_f1_0')(f1_0_vec)
    sin_f1_0 = keras.layers.Activation(activation=tf.sin, name='sin_f1_0')(f1_0_vec)
    cos_f2_0 = keras.layers.Activation(activation=tf.cos, name='cos_f2_0')(f2_0_vec)
    sin_f2_0 = keras.layers.Activation(activation=tf.sin, name='sin_f2_0')(f2_0_vec)

    # ******************************************************************
    # Neural network: feature layers
    # ******************************************************************
    
    # Create an initial array of features: the time, mass and orbital elements
    feature_list = [
        # time of this snapshot and body masses (body 1 constant = 1.0 Msun)
        t_vec, m1, m2,
        # orbital elements 1 (10 features)
        a1, e1, 
        cos_inc1, sin_inc1, 
        cos_Omega1, sin_Omega1, 
        cos_omega1, sin_omega1, 
        cos_f1_0, sin_f1_0,
        # orbital elements 2 (10 features)
        a2, e2, 
        cos_inc2, sin_inc2, 
        cos_Omega2, sin_Omega2, 
        cos_omega2, sin_omega2, 
        cos_f2_0, sin_f2_0]
    # Inputs to neural network is a flattened arrat; 23 features per time snap
    phi_0 = keras.layers.concatenate(inputs=feature_list, name='phi_0')

    # Hidden layers as specified in hidden_sizes
    # Number of hidden layers
    num_layers = len(hidden_sizes)

    # phi_n will update to the last available feature layer for the output portion
    phi_n = phi_0

    # First hidden layer if applicable
    if num_layers > 0:
        phi_1 = keras.layers.Dense(units=hidden_sizes[0], activation='tanh', name='phi_1')(phi_0)
        if skip_layers:
            phi_1 = keras.layers.concatenate(inputs=[phi_0, phi_1], name='phi_1_aug')
        phi_n = phi_1

    # Second hidden layer if applicable
    if num_layers > 1:
        phi_2 = keras.layers.Dense(units=hidden_sizes[1], activation='tanh', name='phi_2')(phi_1)
        if skip_layers:
            phi_2 = keras.layers.concatenate(inputs=[phi_1, phi_2], name='phi_2_aug')
        phi_n = phi_2

    # Third hidden layer if applicable
    if num_layers > 2:
        phi_3 = keras.layers.Dense(units=hidden_sizes[2], activation='tanh', name='phi_3')(phi_2)
        if skip_layers:
            phi_3 = keras.layers.concatenate(inputs=[phi_2, phi_3], name='phi_3_aug')
        phi_n = phi_3

    # ******************************************************************
    # Neural network: layers with time derivatives of orbital elements
    # ******************************************************************
    
    # Set type of regularizer
    # Set strength of activity regularizer using activity_reg input
    reg_type = keras.regularizers.l1
    
    # Semimajor axis
    ddt_a1 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_a1')(phi_n)
    ddt_a2 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_a2')(phi_n)

    # Eccentricity
    ddt_e1 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_e1')(phi_n)
    ddt_e2 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_e2')(phi_n)

    # Inclination
    ddt_inc1 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_inc1')(phi_n)
    ddt_inc2 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_inc2')(phi_n)
    
    # Longitude of ascending node
    ddt_Omega1 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_Omega1')(phi_n)
    ddt_Omega2 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_Omega2')(phi_n)
    
    # Argument of periapsis
    ddt_omega1 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_omega1')(phi_n)
    ddt_omega2 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_omega2')(phi_n)

    # True anomaly
    ddt_f1 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=keras.regularizers.l1(activity_reg), 
        name='ddt_f1')(phi_n)    
    ddt_f2 = keras.layers.Dense(
        units=1, kernel_initializer='zeros', bias_initializer='zeros',
        kernel_regularizer=reg_type(kernel_reg),
        activity_regularizer=reg_type(activity_reg), 
        name='ddt_f2')(phi_n)    

    # ******************************************************************
    # Apply adjustments to Kepler-Jacobi orbital elements
    # ******************************************************************
        
    a1 = keras.layers.add(inputs=[a1, ddt_a1 * t_vec], name='a1_raw')
    a2 = keras.layers.add(inputs=[a2, ddt_a2 * t_vec], name='a2_raw')
    
    e1 = keras.layers.add(inputs=[e1, ddt_e1 * t_vec], name='e1_raw')
    e2 = keras.layers.add(inputs=[e2, ddt_e2 * t_vec], name='e2_raw')

    inc1 = keras.layers.add(inputs=[inc1, ddt_inc1 * t_vec], name='inc1_raw')
    inc2 = keras.layers.add(inputs=[inc2, ddt_inc2 * t_vec], name='inc2_raw')

    Omega1 = keras.layers.add(inputs=[Omega1, ddt_Omega1 * t_vec], name='Omega1')
    Omega2 = keras.layers.add(inputs=[Omega2, ddt_Omega2 * t_vec], name='Omega2')

    omega1 = keras.layers.add(inputs=[omega1, ddt_omega1 * t_vec], name='omega1')
    omega2 = keras.layers.add(inputs=[omega2, ddt_omega2 * t_vec], name='omega2')

    f1 = keras.layers.add(inputs=[f1, ddt_f1 * t_vec], name='f1')
    f2 = keras.layers.add(inputs=[f2, ddt_f2 * t_vec], name='f2')    

    # Limit a to be non-negative
    a1 = keras.layers.ReLU(name='a1')(a1)
    a2 = keras.layers.ReLU(name='a2')(a2)

    # Limit e to be in interval [0.0, 0.9999]
    ecc_min = 0.0000
    ecc_max = 0.9900
    clip_func_e = lambda x : tf.clip_by_value(x, ecc_min, ecc_max)
    e1 = keras.layers.Activation(activation=clip_func_e, name='e1')(e1)
    e2 = keras.layers.Activation(activation=clip_func_e, name='e2')(e2)
    
    # Limit inc to be in interval [0.0, pi]
    inc_min = 0.0
    inc_max = np.pi
    clip_func_inc = lambda x : tf.clip_by_value(x, inc_min, inc_max)
    inc1 = keras.layers.Activation(activation=clip_func_inc, name='inc1')(inc1)
    inc2 = keras.layers.Activation(activation=clip_func_inc, name='inc2')(inc2)
    
    # The remaining elements can take any value; angles can wrap around past 2pi

    # ******************************************************************
    # Convert orbital elements to cartesian Jacobi coordinates and then to Cartesian body coordinates
    # ******************************************************************
    
    # The position of Jacobi coordinate 0 over time comes from the average velocity
    # We always use center of momentum coordinates, so this is zero
    qjt_0 = keras.backend.zeros(shape=[batch_size, traj_size, space_dims])
    vjt_0 = keras.backend.zeros(shape=[batch_size, traj_size, space_dims])
    ajt_0 = keras.backend.zeros(shape=[batch_size, traj_size, space_dims])
    
    # Model mapping orbital elements to cartesian coordinates
    model_e2c_1 = make_model_elt_to_cfg(include_accel=True, batch_size=batch_size, name='elt_to_jac_1')
    model_e2c_2 = make_model_elt_to_cfg(include_accel=True, batch_size=batch_size, name='elt_to_jac_2')

    # Wrap orbital elements into one tuple of inputs for layer converting to cartesian coordinates
    elt1 = (a1, e1, inc1, Omega1, omega1, f1, mu1,)
    elt2 = (a2, e2, inc2, Omega2, omega2, f2, mu2,)

    # Convert from orbital elements to cartesian coordinates
    # This is the position and velocity of the Jacobi coordinate 
    qjt_1, vjt_1, ajt_1 = model_e2c_1(elt1)
    qjt_2, vjt_2, ajt_2 = model_e2c_2(elt2)
    
    # Reshape the Jacobi coordinates to include an axis for body number
    particle_traj_shape = (-1, 1, 3)
    particle_traj_shape_layer = keras.layers.Reshape(target_shape=particle_traj_shape, name='particle_traj_shape')
    qjt_0 = particle_traj_shape_layer(qjt_0)
    qjt_1 = particle_traj_shape_layer(qjt_1)
    qjt_2 = particle_traj_shape_layer(qjt_2)
    vjt_0 = particle_traj_shape_layer(vjt_0)
    vjt_1 = particle_traj_shape_layer(vjt_1)
    vjt_2 = particle_traj_shape_layer(vjt_2)
    ajt_0 = particle_traj_shape_layer(ajt_0)
    ajt_1 = particle_traj_shape_layer(ajt_1)
    ajt_2 = particle_traj_shape_layer(ajt_2)

    # Assemble the Jacobi coordinates over time
    qj = keras.layers.concatenate(inputs=[qjt_0, qjt_1, qjt_2], axis=-2, name='qj')
    vj = keras.layers.concatenate(inputs=[vjt_0, vjt_1, vjt_2], axis=-2, name='vj')
    aj = keras.layers.concatenate(inputs=[ajt_0, ajt_1, ajt_2], axis=-2, name='aj')

    # Convert the Jacobi coordinates over time to Cartesian coordinates
    q, v, a = JacobiToCartesian(include_accel=True, batch_size=batch_size)([m, qj, vj, aj])
    
    # Name the outputs
    q = Identity(name='q')(q)
    v = Identity(name='v')(v)
    a = Identity(name='a')(a)
    
    # Wrap up the outputs
    outputs = (q, v, a)

    # Diagnostic - include computed orbital elements with the output
    # outputs = outputs + elt1 + elt2

    # Wrap this into a model
    suffix = '_'.join(str(sz) for sz in hidden_sizes)
    model_name = f'model_g3b_position_nn_{suffix}'
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model

# ********************************************************************************************************************* 
def make_model_g3b_nn(hidden_sizes, skip_layers=True, use_autodiff: bool = False,
                      kernel_reg=1.0E-6, activity_reg=1.0E-6, 
                      traj_size = 1001, batch_size = 64):
    """Create a NN model for the general three body problem; wrapper for entire work flow"""
    # Build the position model
    position_model = make_position_model_g3b_nn(hidden_sizes=hidden_sizes, skip_layers=skip_layers, 
                                                kernel_reg=kernel_reg, activity_reg=activity_reg,
                                                traj_size=traj_size, batch_size=batch_size)
    
    # Build the model with this position layer and the input trajectory size
    return make_physics_model_g3b(position_model=position_model, use_autodiff=use_autodiff,
                                  traj_size=traj_size, batch_size=batch_size)

# ********************************************************************************************************************* 
def baseline_loss(model, ds):
    """Make initial entry on history table before training starts"""
    loss_keys = ['loss', 
                 'q_loss', 'v_loss', 'a_loss',
                 'q0_rec_loss', 'v0_rec_loss',
                 'H_loss', 'P_loss', 'L_loss']
    loss_vals = model.evaluate(ds)
    loss_baseline = {key: loss_vals[i] for i, key in enumerate(loss_keys)}
    
    # Set dummy batch_num and time
    loss_baseline['batch_num'] = 0
    loss_baseline['time'] = 0.0

    # hist0 is a dictionary of numpy arrays
    hist0 = {key: np.array([val], dtype=np.float32) for key, val in loss_baseline.items()}

    # Baseline position loss
    q_loss = loss_baseline['q_loss']
    
    return hist0, q_loss
