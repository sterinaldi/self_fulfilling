from tqdm import tqdm
import numpy as np
from memspectrum import MESA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import decimate
from scipy.stats import dirichlet, multinomial
from scipy.stats import beta, norm
from numpy.random import choice

T = 32
df = 1./T
old_sampling_rate = 4096
new_sampling_rate = 4096
dt = 1./new_sampling_rate

N = T*new_sampling_rate
time = np.linspace(0,T,N)
data  = np.loadtxt("../data/H-H1_GWOSC_4KHZ_R1-1126259447-32.txt")[:T*old_sampling_rate]
#data = np.random.normal(0,1,size=N)
if new_sampling_rate != old_sampling_rate:
    data = decimate(data, int(old_sampling_rate/new_sampling_rate), zero_phase=True)
M = MESA()
M_test = MESA()
#train, data = np.array_split(data, 2)

#M.solve(train, optimisation_method = "FPE", early_stop = False)
#print("train(p) =", M.get_p())
#frequencies, PSD = M.spectrum(dt, onesided=True)
chunk_length = 1024#int(2**(np.ceil(np.log2(M.get_p()))))
chunk_duration = dt*chunk_length
n_chunks       = int(N/chunk_length)
print("chunk length = ",chunk_length,"chunk duration = ",chunk_duration,"n_chunks = ",n_chunks)

#early_time, late_time = np.array_split(time,2)
times = np.array_split(time, n_chunks)
chunks = np.array_split(data, n_chunks)

p_max    = int(2*chunk_length / np.log(2*chunk_length))

probability_matrix = np.zeros((p_max,n_chunks))

for i in tqdm(range(1,p_max)):
    for j in tqdm(range(n_chunks)):
        M.solve(chunks[j], optimisation_method = "Fixed", early_stop = False, m = i)
        probability_matrix[i,j] = M.logL(chunks[j], dt)

f = plt.figure()
ax = f.add_subplot(111)
ax.matshow(probability_matrix)
plt.show()
np.savetxt('matricione.txt', probability_matrix)
