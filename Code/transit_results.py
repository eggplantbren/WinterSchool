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


# If we're doing transit_model2
if keep.shape[1] == 7:
  plt.figure(figsize=(13, 6))
  plt.subplot(1, 2, 1)
  plt.hist(keep[100:,4], 100, alpha=0.5)
  plt.xlabel('$\\log(\\nu)$')
  plt.ylabel('Number of samples')
  plt.xlim([np.log(0.1), np.log(100.)])

  plt.subplot(1, 2, 2)
  plt.hist(keep[100:,5], 100, alpha=0.5)
  plt.xlabel('$u_K$')
  plt.ylabel('Number of samples')
  plt.xlim([-1., 1.])

  plt.savefig('nu_K.pdf', bbox_inches='tight')
  plt.show()

