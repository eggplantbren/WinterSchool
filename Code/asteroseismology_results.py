import numpy as np
import matplotlib.pyplot as plt

from asteroseismology_model import *

keep = np.loadtxt('keep.txt')

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
plt.ion()
for i in range(keep.shape[0]//2, keep.shape[0]):
  plt.hold(False)
  plt.plot(data[:,0], data[:,1], 'b.')
  mu = signal(keep[i, :])
  plt.hold(True)
  plt.plot(data[:,0], mu, 'r-', linewidth=2)
  plt.title('Model {i}/{n}'.format(i=(i+1), n=keep.shape[0]))
  plt.ylabel('Power')
  plt.draw()
plt.ioff()
plt.show()


