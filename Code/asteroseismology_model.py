import numpy as np
import numpy.random as rng
import copy

# How many parameters are there?
num_params = 32

# Load the data
data = np.loadtxt('asteroseismology_data.txt')

# Some properties of the data
x_min, x_max = data[:,0].min(), data[:,0].max()
x_range = x_max - x_min
N = data.shape[0] # Number of data points

# Some idea of how big the Metropolis proposals should be
jump_sizes = np.ones(num_params)
jump_sizes[0] = 10.
jump_sizes[1] = np.log(1E6)

def from_prior():
  """
  A function to generate parameter values from the prior.
  Returns a numpy array of parameter values.
  """
  params = rng.rand(num_params)

  # Number of components (round down to get integer)
  params[0] = 10.*params[0]

  # log of background
  params[1] = np.log(1E-3) + np.log(1E6)*rng.rand()

  # The rest are for the amplitudes, centers, and widths of the peaks,
  # and have U(0, 1) priors (they'll be transformed before being used)

  return params

def log_prior(params):
  """
  Evaluate the (log of the) prior distribution
  """

  # Minus infinity, if out of bounds
  if params[0] < 0. or params[0] > 10.:
    return -np.Inf

  if params[1] < np.log(1E-3) or params[1] > np.log(1E3):
    return -np.Inf

  if np.any(params[2:] < 0) or np.any(params[2:] > 1):
    return -np.Inf

  return 0.

def log_likelihood(params):
  """
  Evaluate the (log of the) likelihood function
  """
  # Rename the parameters
  num_peaks = int(params[0])
  B = np.exp(params[1])

  # Calculate the expected/model signal
  mu = B + np.zeros(N)

  # Add the peaks
  k = 2
  for i in range(0, num_peaks):
    # Get the parameters
    A = -20.*np.log(1. - params[k])
    xc = x_min + x_range*params[k+1]
    width = np.exp(np.log(1E-3*x_range) + np.log(1E3)*params[k+2])

    # Add the Lorentzian peak
    mu += A/(1. + ((data[:,0] - xc)/width)**2)
    k += 3

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
  new[0] = np.mod(new[0], 10.)
  return new

