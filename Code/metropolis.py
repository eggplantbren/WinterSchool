import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Set the seed
rng.seed(0)

# Import the model
from asteroseismology_model import from_prior, log_prior, log_likelihood, proposal,\
                              num_params

# Generate a starting point from the prior
# (Feel free to use a better recipe if you have one)
params = from_prior()
logp, logl = log_prior(params), log_likelihood(params)

# Total number of iterations
steps = 200000

# How often to save
skip = 100

# How often to plot (should be divisible by skip)
plot_skip = 1000

# Storage array for the results
# Extra column is for the log likelihood.
keep = np.empty((steps//skip, num_params + 1))

# Plotting stuff
plt.ion()
plt.hold(False)

# Main loop
for i in range(0, steps):
  # Generate proposal
  new = proposal(params)

  # Evaluate prior and likelihood for the proposal
  logp_new = log_prior(new)
  if logp_new != -np.Inf:
    logl_new = log_likelihood(new)

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
    keep[(i+1)//skip-1, 0:-1] = params
    keep[(i+1)//skip-1, -1] = logl

    if (i+1)%plot_skip == 0:
      # Plot one of the parameters over time
      # Ignore the first 25% as burn-in
      plt.plot(keep[(index//4):(index+1), -1], 'b')
      plt.xlabel('Iteration')
      plt.draw()

# Save the results to a file
np.savetxt('keep.txt', keep)

plt.ioff()
plt.show()

