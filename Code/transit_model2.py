import numpy as np
import numpy.random as rng
import copy
import scipy.special

# How many parameters are there?
num_params = 6

# Load the data
data = np.loadtxt('transit_data.txt')

# Some properties of the data
t_min, t_max = data[:,0].min(), data[:,0].max()
t_range = t_max - t_min
N = data.shape[0] # Number of data points

# Some idea of how big the Metropolis proposals should be
jump_sizes = np.array([200., 10., t_range, t_range, np.log(1E3), 2.])

def from_prior():
  """
  A function to generate parameter values from the prior.
  Returns a numpy array of parameter values.
  """
  A = -100. + 200.*rng.rand()
  b = 10.*rng.rand()
  tc = t_min + t_range*rng.rand()
  width = t_range*rng.rand()
  log_nu = np.log(0.1) + np.log(1000.)*rng.rand()
  u_K = -1. + 2.*rng.rand()

  return np.array([A, b, tc, width, log_nu, u_K])

def log_prior(params):
  """
  Evaluate the (log of the) prior distribution
  """
  A, b, tc, width, log_nu, u_K = params[0], params[1], params[2], params[3]\
                                    , params[4], params[5]

  # Minus infinity, if out of bounds
  if A < -100. or A > 100.:
    return -np.Inf
  if b < 0. or b > 10.:
    return -np.Inf
  if tc < t_min or tc > t_max:
    return -np.Inf
  if width < 0. or width > t_range:
    return -np.Inf
  if log_nu < np.log(0.1) or log_nu > np.log(100.):
    return -np.Inf
  if u_K < -1. or u_K > 1.:
    return -np.Inf

  return 0.

def log_likelihood(params):
  """
  Evaluate the (log of the) likelihood function
  """
  # Rename the parameters
  A, b, tc, width, log_nu, u_K = params[0], params[1], params[2]\
                                ,params[3], params[4], params[5]

  # Parameter is really log_nu
  nu = np.exp(log_nu)

  # Compute K and 'inflated' error bars
  if u_K < 0.:
    K = 1.
  else:
      K = 1. - np.log(1. - u_K)
  sig = K*data[:,2]

  # First calculate the expected signal
  mu = A*np.ones(N)
  mu[np.abs(data[:,0] - tc) < 0.5*width] = A - b

  # Student t distribution
  return N*scipy.special.gammaln(0.5*(nu+1.))\
       - N*scipy.special.gammaln(0.5*nu)\
       - np.sum(np.log(sig*np.sqrt(np.pi*nu)))\
       - 0.5*(nu + 1.)*np.sum(np.log(1. + (data[:,1] - mu)**2/nu/sig**2))


def proposal(params):
  """
  Generate new values for the parameters, for the Metropolis algorithm.
  """
  # Copy the parameters
  new = copy.deepcopy(params)

  # Which one should we change?
  which = rng.randint(num_params)
  new[which] += jump_sizes[which]*10.**(1.5 - 6.*rng.rand())*rng.randn()
  return new


