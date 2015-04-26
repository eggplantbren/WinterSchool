from pylab import *

rc("font", size=16, family="serif", serif="Computer Sans")
rc("text", usetex=True)

"""
Make nested sampling figures
"""

X = linspace(0, 1, 1001)
L = exp(sqrt(-log(X)))

plot(X, L, 'b')
xlabel('$X$')
ylabel('$L$')
show()

