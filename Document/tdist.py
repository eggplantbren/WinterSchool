from pylab import *
from scipy.special import gamma as gam

# Make t distribution figure
rc("font", size=16, family="serif", serif="Computer Sans")
rc("text", usetex=True)

x = linspace(-10, 10, 5001)

nu1, nu2, nu3 = 1., 3., 50.
p1 = gam((nu1 + 1.)/2.)/gam(nu1/2.)/sqrt(pi*nu1)*(1. + x**2/nu1)**(-(nu1 + 1.)/2.)
p2 = gam((nu2 + 1.)/2.)/gam(nu2/2.)/sqrt(pi*nu2)*(1. + x**2/nu2)**(-(nu2 + 1.)/2.)
p3 = gam((nu3 + 1.)/2.)/gam(nu3/2.)/sqrt(pi*nu3)*(1. + x**2/nu3)**(-(nu3 + 1.)/2.)

plot(x, p1, 'b', linewidth=2, label='$\\nu=1$')
plot(x, p2, 'r--', linewidth=2, label='$\\nu=3$')
plot(x, p3, 'g:', linewidth=3, label='$\\nu=50$')
xlabel('$x$')
ylabel('Probability Density')
legend()
savefig('tdist.pdf', bbox_inches='tight')
show()

