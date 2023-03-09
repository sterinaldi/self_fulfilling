import numpy as np
from self_fulfilling.model import SF
from figaro.plot import plot_1d_dist

matricione = np.loadtxt('/Users/stefanorinaldi/Documents/Repo/self_fulfilling/scripts/matricione.txt')

max_P = 267

SF = SF(matricione)
draws = SF.rvs(1000)
plot_1d_dist(np.arange(1, max_P+1), draws, name = 'fast_P', label = 'P')
