from pylab import *

rc("font", size=16, family="serif", serif="Computer Sans")
rc("text", usetex=True)

"""
Make nested sampling figures
"""

seed(123)

###################
# Plot of L vs. X #
###################

X = linspace(0.001, 1, 1001)
L = exp(sqrt(-log(X)))

plot(X, L, 'k', linewidth=2, label='True Curve')
xlabel('$X$')
ylabel('Likelihood')

# Colour the area below the curve
ax = gca()
ax.fill_between(X, L, alpha=0.2)

# Add some random points
X = exp(-5 + 5*rand(20))
L = exp(sqrt(-log(X)))
plot(X, L, 'ko', markersize=10, label='Points Obtained')
legend(numpoints=1)

savefig('nested1.pdf', bbox_inches='tight')
show()

#############################
# Plot of log(L) vs. log(X) #
#############################

logX = linspace(-30, 0, 1001)
logL = sqrt(-logX/0.02)
plot(logX, logL, 'k', linewidth=2, label='Log Likelihood')
xlabel('$\log(X)$')

post = exp(logX + logL)
post /= trapz(post, x=logX)

plot(logX, 40*exp(logX), 'r--', linewidth=2, label='Prior')
plot(logX, 40*post, 'b', linewidth=2, label='Posterior')
legend()
savefig('nested2.pdf', bbox_inches='tight')
show()

