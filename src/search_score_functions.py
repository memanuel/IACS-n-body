"""
Michael S. Emanuel
Fri Oct 25 19:20:56 2019
"""
# Library imports
import tensorflow as tf
import numpy as np

# Local imports
from observation_data import random_direction

# ********************************************************************************************************************* 
# Functions for scoring predicted asteroid directions vs. observations
# ********************************************************************************************************************* 

# ********************************************************************************************************************* 
def score_mean(A: tf.Tensor):
    """
    Expected value of the score function 
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    If we use cylindrical coordinates (z, phi), with r^2 +z^2 = 1, x^2 + y^2 = r^2, and
    x = r cos(phi), y = r sin(phi), z = z
    and set the true direction at (0, 0, 1), we can get the EV by integrating over the sphere.
    epsilon is a right triangle with sides (1-z) and r, regardless of phi, so
    epsilon^2 = (1-z)^2 + r^2 = 1 -2z + z^2 + r^2 = 1 + (r^2+z^2) - 2z = 2 - 2z = 2(1-z)
    This can be integrated symbolically.
    """
    # The expected value is (1 - e^-2A) / 2A
    # return (1.0 - np.exp(-2.0*A)) / (2*A)
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # Do calculation in double
    minus_two_A = tf.multiply(A, -2.0, name='minus_two_A')
    expm1_m2a = tf.math.expm1(minus_two_A, name='expm1_m2a')
    mean = tf.divide(expm1_m2a, minus_two_A)
    return tf.cast(mean, dtype)

# ********************************************************************************************************************* 
def score_var(A: tf.Tensor):
    """
    Variance of the score function 
    f(epsilon) = exp(-1/2 A epsilon^2)
    This can be integrated symbolically.
    """
    # The variance is (A*Coth(A)) * E[f]^2
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # Do calculation in double
    # Calculate the leading factor
    # B = (A / tf.tanh(A)) - 1.0
    tanh_A = tf.tanh(A, name='tanh_A')
    A_div_tanh_A = tf.divide(A, tanh_A, name='A_div_tanh_A')
    B = tf.subtract(A_div_tanh_A, 1.0, name='B')

    # Calculate the mean
    mean: tf.Tensor = score_mean(A)
    mean2 = tf.square(mean, name='mean2')
    var = tf.multiply(B, mean2, name='var')
    return tf.cast(var, dtype)

# ********************************************************************************************************************* 
def score_std(A: tf.Tensor):
    """
    Standard deviation of the score function 
    f(epsilon) = exp(-1/2 A epsilon^2)
    This can be integrated symbolically.
    """
    # The variance is (A*Coth(A)) * E[f]^2
    # The std deviation is sqrt(A*Coth(A)) * E[f]
    # B = (A / tf.tanh(A)) - 1.0
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # Do calculation as double
    tanh_A = tf.tanh(A, name='tanh_A')
    A_div_tanh_A = tf.divide(A, tanh_A, name='A_div_tanh_A')
    B2 = tf.subtract(A_div_tanh_A, 1.0, name='B2')
    B = tf.sqrt(B2, name='B')
    # Calculate the mean
    mean: tf.Tensor = score_mean(A)
    std = tf.multiply(B, mean, name='std')
    return tf.cast(std, dtype)

# ********************************************************************************************************************* 
def score_mean_2d_exact(A: tf.Tensor):
    """
    Expected value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    # The expected value is (1 - e^-2A) / 2A
    # return np.exp(-A) * np.i0(A)
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # Do calculation as double
    minus_A = tf.multiply(A, -1.0, name='minus_A')
    exp_ma = tf.math.exp(minus_A, name='exp_ma')
    bessel = tf.math.bessel_i0(A, name='bessel')
    mean = tf.multiply(exp_ma, bessel)
    
    # Return result in original data type
    return tf.cast(mean, dtype)

# ********************************************************************************************************************* 
def score_mean_2d_approx(A: tf.Tensor):
    """
    Expected value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    # The expected value is (1 - e^-2A) / 2A
    # return np.exp(-A) * np.i0(A)
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # Do calculation as double
    # Approximate value of bessel for large x is e^x / sqrt(2pi x)
    # Approximate value of mean is therefore 1 / sqrt(2pi x)
    two_pi = tf.constant(np.pi * 2.0, dtype=tf.float64)
    two_pi_A = tf.multiply(A, two_pi)
    sqrt_two_pi_x = tf.sqrt(two_pi_A)
    mean = tf.divide(1.0, sqrt_two_pi_x)
    
    # Return result in original data type
    return tf.cast(mean, dtype)

# ********************************************************************************************************************* 
def score_mean_2d(A: tf.Tensor):
    """
    Expected value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    # The expected value is (1 - e^-2A) / 2A
    mean_exact = score_mean_2d_exact(A)
    mean_approx = score_mean_2d_approx(A)
    # cond = tf.math.is_nan(mean_exact)
    cond = tf.math.greater(A, 700)
    
    # Return result in original data type
    return tf.where(cond, mean_approx, mean_exact)

# ********************************************************************************************************************* 
def score_var_2d_exact(A: tf.Tensor):
    """
    Variance value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    # The variance is
    # E^(-2 a) * (a^2 BesselI[0, 2 a] + Sinh[a] (-2 a BesselI[0, a] + Sinh[a])) / a^2
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # m_a = tf.multiply(A, -1.0, name='m_a')
    m_2a = tf.multiply(A, -2.0, name='m_2a')
    # exp_ma = tf.math.exp(m_a, name='exp_ma')
    exp_m2a = tf.math.exp(m_2a, name='exp_m2a')
    a2 = tf.square(A, name='a2')
    bes_a = tf.math.bessel_i0(A, name='bes_a')
    bes_2a = tf.math.bessel_i0(tf.multiply(A, 2.0), name='bes_2a')
    a2_bes_2a = tf.multiply(a2, bes_2a, name='a2_bes_2a')
    sinh_a = tf.math.sinh(A, name='sinh_a')
    m2a_bes_a = tf.multiply(m_2a, bes_a, name='m2s_bes_1')
    m2a_bes_a_p_sinh_a = tf.add(m2a_bes_a, sinh_a, name='m2a_bes_a_p_sinh_a')
    sa_x_bes_p_sinh = tf.multiply(sinh_a, m2a_bes_a_p_sinh_a, name='sa_x_bes_p_sinh')
    term = tf.add(a2_bes_2a, sa_x_bes_p_sinh, name='term')
    num = tf.multiply(exp_m2a, term, name='num')
    var = tf.divide(num, a2)
    return tf.cast(var, dtype)

# ********************************************************************************************************************* 
def var_ratio_2d(A):
    return score_var_2d_exact(A) / score_mean_2d_exact(A)

# ********************************************************************************************************************* 
def score_var_2d_approx(A: tf.Tensor):
    """
    Variance value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    # The variance is
    # E^(-2 a) * (a^2 BesselI[0, 2 a] + Sinh[a] (-2 a BesselI[0, a] + Sinh[a])) / a^2
    # Save original data type and cast to double
    try:
        dtype = A.dtype
    except:
        dtype = tf.float32
    A = tf.cast(A, tf.float64)

    # Series expansion for log(var_2d(a)) according to Mathematica is
    ratio = var_ratio_2d(tf.constant(300.0, dtype=tf.float64))
    mean = score_mean_2d(A)
    var = tf.multiply(mean, ratio)
    return tf.cast(var, dtype)

# ********************************************************************************************************************* 
def score_var_2d(A: tf.Tensor):
    """
    Expected value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    # The expected value is (1 - e^-2A) / 2A
    var_exact = score_var_2d_exact(A)
    var_approx = score_var_2d_approx(A)
    cond = tf.math.greater(A, 350)
    
    # Return result in original data type
    return tf.where(cond, var_approx, var_exact)

# ********************************************************************************************************************* 
def score_std_2d(A: tf.Tensor):
    """
    Expected value of the score function assuming planar distribution of observations
    f(epsilon) = exp(-1/2 A epsilon^2)
    epsilon is the Euclidean distance between the predicted and observed direction; 
    epsilon^2 = dx^2 + dy^2 + dz^2 with dx = u_obs_x - u_pred_x, etc.
    Assume (conservatively) that asteroids are distributed on the unit circle within
    the unit sphere with inclination zero. This can be integrated symbolically.
    """
    var = score_var_2d(A)
    std = tf.sqrt(var)
    return std

# ********************************************************************************************************************* 
def test_score_moments_one(A: float, mean_exp: float, var_exp: float) -> bool:
    """Test the mean and variance for the score function with known values"""
    # Test the mean
    mean_calc = score_mean(A)
    isOK: bool = np.isclose(mean_exp, mean_calc, atol=1.0E-7)     
    if not isOK:
        print(f'Failed on A = 1.0, expected mean {mean_exp}, got mean {mean_calc}.')

    # Test the variance
    var_calc = score_var(A)
    isOK = isOK and np.isclose(var_exp, var_calc, atol=1.0E-7)
    if not isOK:
        print(f'Failed on A = 1.0, expected var {var_exp}, got var {var_calc}.')
        
    return isOK

# ********************************************************************************************************************* 
def test_score_moments_2d_one(A: float, mean_exp: float, var_exp: float) -> bool:
    """Test the mean and variance for the score function with known values"""
    # Test the mean
    mean_calc = score_mean_2d(A)
    isOK: bool = np.isclose(mean_exp, mean_calc, atol=1.0E-7)     
    if not isOK:
        print(f'Failed on A = 1.0, expected mean {mean_exp}, got mean {mean_calc}.')

    # Test the variance
    var_calc = score_var_2d(A)
    isOK = isOK and np.isclose(var_exp, var_calc, atol=1.0E-7)
    if not isOK:
        print(f'Failed on A = 1.0, expected var {var_exp}, got var {var_calc}.')
        
    return isOK

# ********************************************************************************************************************* 
def test_score_moments():
    """Test the mean and variance for the score function with known values"""
    # When A = 1.0, mean = 0.432332, var = 0.0585098
    A = np.array(1.0)
    mean_exp = 0.432332
    var_exp = 0.0585098
    isOK: bool = test_score_moments_one(A, mean_exp, var_exp)
    
    # When A = 32.828063500117445 (10 degrees), mean = 0.0152309, var = 0.00738346
    A = np.array(1.0 / np.deg2rad(10.0)**2)
    mean_exp = 0.0152309
    var_exp = 0.00738346
    isOK = isOK and test_score_moments_one(A, mean_exp, var_exp)
    
    # On the plane,
    mean_exp = 0.0698984
    var_exp = 0.0474321
    isOK = isOK and test_score_moments_2d_one(A, mean_exp, var_exp)
    
    # Report results
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'Test score moments:\n***** {msg} *****')    
    return isOK

# ********************************************************************************************************************* 
def test_score_num(N: int, R_deg: float):
    """
    Numerical test of score_mean and score_var functions
    INPUTS:
        N: number of points to test
        R_deg: resolution factor in degrees
    """
    # Convert resolution factor to A
    R: float = np.deg2rad(R_deg)
    A: float = 1.0 / R**2
    # Calculated mean and variance
    mean_calc: float = score_mean(A)
    std_calc: float = score_std(A)

    # Numerical samples
    directions: np.array = random_direction(N)
    # Difference with north pole    
    north_pole = np.array([0.0, 0.0, 1.0])
    epsilon = directions- north_pole
    epsilon2 = np.sum(epsilon*epsilon, axis=-1)

    # Simulated mean and stdev
    scores = np.exp(-0.5 * A * epsilon2)
    mean_num: float = np.mean(scores)
    mean_error_rel = np.abs(mean_num-mean_calc) / mean_calc
    std_num: float = np.std(scores)
    std_error_rel = np.abs(std_num-std_calc) / std_calc

    # Test the mean
    rtol: float = 2.0 / N**0.4
    isOK: bool = np.isclose(mean_calc, mean_num, rtol=rtol)
    print(f'\nTest score function with N={N}, R={R_deg} degrees:')
    print(f'Mean:')
    print(f'calc      = {mean_calc:10.8f}')
    print(f'num       = {mean_num:10.8f}')
    print(f'error_rel = {mean_error_rel:10.8f}')
    
    # Test the standard deviation
    rtol = 2.0 / N**0.4
    isOK = isOK and np.isclose(mean_calc, mean_num, rtol=rtol)
    print(f'\nStandard Deviation:')
    print(f'calc      = {std_calc:10.8f}')
    print(f'num       = {std_num:10.8f}')
    print(f'error_rel = {std_error_rel:10.8f}')
    
    # Summary results    
    msg: str = 'PASS' if isOK else 'FAIL'
    print(f'***** {msg} *****')
    return isOK

# ********************************************************************************************************************* 
def main():
    # Test the score function
    test_score_num(10000, 10.0)
    test_score_moments()

# ********************************************************************************************************************* 
if __name__ == '__main__':
    main()

