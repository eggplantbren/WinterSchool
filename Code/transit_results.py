import numpy as np
import matplotlib.pyplot as plt

keep = np.loadtxt('keep.txt')

# Plotting stuff
plt.rc("font", size=16, family="serif", serif="Computer Sans")
plt.rc("text", usetex=True)

plt.plot(keep[:,0])
plt.title('Trace Plot')
plt.xlabel('Iteration')
plt.ylabel('$A$')
plt.savefig('trace_plot.pdf', bbox_inches='tight')
plt.show()

plt.hist(keep[100:,0], 100, alpha=0.5)
plt.title('Marginal Posterior Distribution')
plt.xlabel('$A$')
plt.ylabel('Number of samples')
plt.savefig('marginal_posterior.pdf', bbox_inches='tight')
plt.show()

plt.plot(keep[100:,0], keep[100:,1], 'b.', markersize=1)
plt.xlabel('$A$')
plt.ylabel('$b$')
plt.savefig('joint_posterior.pdf', bbox_inches='tight')
plt.show()

