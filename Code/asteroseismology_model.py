import numpy as np
import numpy.random as rng
import copy

# How many parameters are there?
num_params = 3

# Load the data
data = np.loadtxt('asteroseismology_data.txt')

# Some properties of the data
nu_min, nu_max = data[:,0].min(), data[:,0].max()
nu_range = nu_max - nu_min
N = data.shape[0] # Number of data points

# Some idea of how big the Metropolis proposals should be
jump_sizes = np.array([1., 1., 1.])

def from_prior():
  """
  A function to generate parameter values from the prior.
  Returns a numpy array of parameter values.
  """
  u = rng.rand(3)

  return u

def log_prior(params):
  """
  Evaluate the (log of the) prior distribution
  """

  # Minus infinity, if out of bounds
  if np.any(params < 0) or np.any(params > 1):
    return -np.Inf

  return 0.

def log_likelihood(params):
  """
  Evaluate the (log of the) likelihood function
  """
  # Rename the parameters
  # Also transform away from U(0, 1)
  A = -10.*np.log(1. - params[0])
  nu_c = nu_min + nu_range*params[1]
  width = np.exp(np.log(1E-3*nu_range) + np.log(1E3)*nu_range)

  # First calculate the expected signal
  mu = 10. + np.zeros(N)

  # Add the Lorentzians
  mu += A/(np.pi)/(1. + ((data[:,0] - nu_c)/width)**2)

  # Exponential distribution
  return np.sum(-np.log(mu) - data[:,1]/mu)


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


