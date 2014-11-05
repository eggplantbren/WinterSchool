import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

# Plotting stuff
plt.rc("font", size=16, family="serif", serif="Computer Sans")
plt.rc("text", usetex=True)

# Seed random number generator
rng.seed(0)

# Some frequencies
nu = np.linspace(0., 10., 201)

# Make a noise-free signal
y = 1. + np.zeros(nu.shape)

# Put in some Lorentzians
y = y + 30./(1. + ((nu - 3.)/0.2)**2)
y = y + 50./(1. + ((nu - 5.)/0.2)**2)
y = y + 30./(1. + ((nu - 7.)/0.2)**2)
y = y + 10./(1. + ((nu - 6.)/0.2)**2)
y = y + 10./(1. + ((nu - 4.)/0.2)**2)

# Plot the signal
plt.plot(nu, y, 'r-', linewidth=2, alpha=0.5)

# Add noise (exponential distribution this time)
y = -y*np.log(rng.rand(y.size))

# Make the data into a 3-column array and save it
data = np.empty((y.size, 2))
data[:,0], data[:,1] = nu, y
np.savetxt('asteroseismology_data.txt', data)

# Plot the noisy data
plt.plot(nu, y, 'b.', markersize=10, alpha=0.5)
plt.axis([-1., 11., 0., 100.])
plt.xlabel('Frequency')
plt.ylabel('Power')

# Save the figure
plt.savefig('asteroseismology_data.pdf', bbox_inches='tight')
plt.show()

