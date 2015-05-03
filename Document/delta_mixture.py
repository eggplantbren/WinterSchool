from pylab import *

rc("font", size=16, family="serif", serif="Computer Sans")
rc("text", usetex=True)

x = linspace(1., 5., 1001)
p = 0.5*exp(-(x - 1.))

plot(x, p, 'b', linewidth=2)
hold(True)
plot([0., 1.], [0., 0.], 'b', linewidth=2)
xlabel('$K$')
ylabel('Probability Density')
xlim([0., 5.])
ylim([0., 1.2])
ax = gca()
ax.arrow(1., 0., 0., 1., head_width=0.05, head_length=0.08, fc='b', ec='b')
ax.fill_between(x, p, facecolor='b', alpha=0.2)
title('Prior for $K$')
savefig('delta_mixture.pdf', bbox_inches='tight')
show()

