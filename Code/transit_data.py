import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Plotting stuff
plt.rc("font", size=16, family="serif", serif="Computer Sans")
plt.rc("text", usetex=True)

# Seed random number generator
rng.seed(0)

# Some timestamps
t = np.linspace(0., 10., 101)

# Make a noise-free signal
y = 10. + np.zeros(t.shape)

# Put in a transit
y[np.abs(t - 3.5) < 1.] = 5.

# Plot the signal
plt.plot(t, y, 'r-', alpha=0.5)

# Add noise
sig = 1.
y += sig*rng.randn(y.size)

# Make the data into a 3-column array and save it
data = np.empty((y.size, 3))
data[:,0], data[:,1], data[:,2] = t, y, sig
np.savetxt('transit_data.txt', data)

# Plot the noisy data
plt.errorbar(t, y, yerr=data[:,2], fmt='b.', alpha=0.5)
plt.axis([-1., 11., 0., 15.])
plt.xlabel('Time')
plt.ylabel('Magnitude')

# Save the figure
plt.savefig('transit_data.pdf', bbox_inches='tight')
plt.show()

