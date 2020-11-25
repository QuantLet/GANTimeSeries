import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
x = np.linspace(-4, 8, 2000)
y = stats.gamma.pdf(x, a=2, loc=0, scale=1)
plt.plot(x, y, label='Gamma(2,1)')
y = stats.norm.pdf(x, 0)
plt.plot(x, y, label='std. Gaussian')
plt.ylabel('pdf')
plt.xlabel('x')
plt.savefig(fname='dist_gamma_gaussian.pdf', transparent=True)
plt.savefig(fname='dist_gamma_gaussian.svg', transparent=True)
plt.savefig(fname='dist_gamma_gaussian.png', transparent=True)
