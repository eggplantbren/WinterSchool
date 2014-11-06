import numpy as np
import matplotlib.pyplot as plt

from asteroseismology_model import *

keep = np.loadtxt('keep.txt')

# Trace plot of the number of peaks
plt.plot(keep[:,0].astype('int'))
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Number of Peaks', fontsize=18)
plt.show()

# Histogram of the number of peaks
plt.hist(keep[:,0].astype('int'), 100)
plt.xlabel('Number of peaks', fontsize=18)
plt.ylabel('Number of Posterior Samples', fontsize=18)
plt.show()



def signal(params):
  """
  Calculate the expected curve from the parameters
  (mostly copied from log_likelihood)
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
    width = np.exp(np.log(1E-2*x_range) + np.log(1E2)*params[k+2])

    # Add the Lorentzian peak
    mu += A/(1. + ((data[:,0] - xc)/width)**2)
    k += 3

  # Exponential distribution
  return mu

# Plot a movie of the fits
# Only use the second half of the run.

# Also, accumulate all x-values (frequencies) and amplitudes
# in these arrays:
all_x = np.array([])
all_A = np.array([])

plt.ion()
for i in range(keep.shape[0]//2, keep.shape[0]):
  # Plotting
  plt.hold(False)
  plt.plot(data[:,0], data[:,1], 'b.')
  mu = signal(keep[i, :])
  plt.hold(True)
  plt.plot(data[:,0], mu, 'r-', linewidth=2)
  plt.title('Model {i}/{n}'.format(i=(i+1), n=keep.shape[0]))
  plt.xlabel('Frequency')
  plt.ylabel('Power')
  plt.draw()

  # Accumulate
  num_peaks = keep[i, 0].astype('int')
  A = -10*np.log(1. - keep[i, 2::3][0:num_peaks])
  x = 10.*keep[i, 3::3][0:num_peaks]
  all_x = np.hstack([all_x, x])
  all_A = np.hstack([all_A, A])
plt.ioff()
plt.show()

plt.hist(all_x, 200)
plt.xlabel('Frequency')
plt.ylabel('Number of Posterior Samples')
plt.show()

plt.plot(all_x, all_A, 'b.', markersize=1)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$A$', fontsize=18)
plt.show()

