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
N = 5

# Number of NS iterations
steps = 100000

# MCMC steps per NS iteration
mcmc_steps = 1000

# Generate N particles from the prior
# and calculate their log likelihoods
particles = []
logp = np.empty(N)
logl = np.empty(N)
for i in range(0, N):
 x = from_prior()
 particles.append(x)
 logl[i] = log_likelihood(x)

# Storage for results
keep = np.empty(steps)

plt.ion()
plt.hold(False)

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

  threshold = copy.deepcopy(logl[worst])

  # Evolve within likelihood constraint using Metropolis
  for j in range(0, mcmc_steps):
    new = proposal(particles[worst])
    logp_new = log_prior(new)
    logl_new = log_likelihood(new)
    loga = logp_new - logp[worst]
    if loga > 0.:
      loga = 0.

    # Accept
    if log_likelihood(new) >= threshold and rng.rand() <= np.exp(loga):
      particles[worst] = new
      logp[worst] = logp_new
      logl[worst] = logl_new

  logX = -(np.arange(0, i+1) + 1.)/N

  plt.plot(logX, keep[0:(i+1)], 'bo-')
  # Smart ylim
  temp = keep[0:(i+1)].copy()
  if len(temp) >= 2:
    np.sort(temp)
    plt.ylim([temp[0.2*len(temp)], temp[-1]])
  plt.xlabel('$\\log(X)$')
  plt.ylabel('$\\log(L)$')
  plt.draw()

plt.ioff()
plt.show()

