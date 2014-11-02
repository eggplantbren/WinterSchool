import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt
import copy

# Set the seed
rng.seed(0)

# Import the model
from transit_model import from_prior, log_prior, log_likelihood, proposal,\
                              num_params

# Number of particles
N = 10

# Number of NS iterations
steps = 100000

# MCMC steps per NS iteration
mcmc_steps = 1000

# Generate N particles from the prior
# and calculate their log likelihoods
particles = []
logl = np.empty(N)
for i in range(0, N):
 x = from_prior()
 particles.append(x)
 logl[i] = log_likelihood(x)

# Storage for results
keep = np.empty(steps)


# Main NS loop
for i in range(0, steps):
  # Find worst particle
  worst = np.nonzero(logl == logl.min())[0]

  # Save its likelihood
  keep[i] = logl[worst]

  # Copy survivor
  if N > 1:
    which = rng.randint(N)
    while which == worst:
      which = rng.randint(N)
    particles[worst] = copy.deepcopy(particles[which])

  # Evolve within likelihood constraint
  
