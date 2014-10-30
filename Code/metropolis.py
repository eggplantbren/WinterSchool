import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Import the model
from transit_model import from_prior, log_prior, log_likelihood, proposal,\
                              num_params

# Generate a starting point from the prior
# (Feel free to use a better recipe if you have one)
params = from_prior()
logp = log_prior(params)
logl = log_likelihood(params)

# Total number of iterations
steps = 100000

# How often to save
skip = 10

# How often to plot (should be divisible by skip)
plot_skip = 100

# Storage array for the results
keep = np.empty((steps//skip, num_params))

# Plotting stuff
plt.ion()
plt.hold(False)

# Main loop
for i in range(0, steps):
  # Generate proposal
  new = proposal(params)

  # Evaluate prior and likelihood for the proposal
  logp_new, logl_new = log_prior(new), log_likelihood(new)

  # Acceptance probability
  log_alpha = (logl_new - logl) + (logp_new - logp)
  if log_alpha > 0.:
    log_alpha = 0.

  # Accept?
  if rng.rand() <= np.exp(log_alpha):
    params = new
    logp = logp_new
    logl = logl_new

  # Save results
  if (i+1)%skip == 0:
    index = (i+1)//skip - 1
    keep[(i+1)//skip-1, :] = params

    if (i+1)%plot_skip == 0:
      plt.plot(keep[(index//4):(index+1), 0], 'b')
      plt.draw()

plt.ioff()
plt.show()

