import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Seed random number generator
rng.seed(0)

# Some timestamps
t = np.linspace(0., 10., 101)

# Make a noise-free signal
y = 10. + np.zeros(t.shape)

# Put in a transit
y[np.abs(t - 3.5) < 1.] = 5.

# Plot the signal
plt.plot(t, y, 'r-', alpha=0.2)

# Add noise
sig = 1.
y += sig*rng.randn(y.size)

# Make the data into a 3-column array and save it
data = np.empty((y.size, 3))
data[:,0], data[:,1], data[:,2] = t, y, sig
np.savetxt('transit_data.txt', data)

# Plot the noisy data
plt.errorbar(t, y, yerr=data[:,2], fmt='b.', alpha=0.2)
plt.axis([-1., 11., 0., 15.])
plt.show()

