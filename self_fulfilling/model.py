import numpy as np
from scipy.special import logsumexp
from scipy.stats import dirichlet
from tqdm import tqdm

class SF:
    def __init__(self, P_max,
                       log_p_matrix,
                       ):
        self.P_max        = P_max
        self.log_p_matrix = log_p_matrix
        self.tot_chunks   = log_p_matrix.shape[-1]
        self._initialise()
    
    def _initialise(self):
        self.counts   = np.zeros(self.P_max)
        self.N_chunks = 0
    
    def _add_chunk(self, chunk_id):
        # Computes probability for P
        logp = np.zeros(self.P_max)
        logp = self.log_p_matrix[:,chunk_id] + np.log(self.counts + 1./self.P_max) - np.log(self.N_chunks + self.P_max)
        log_norm = logsumexp(logp)
        # Random assignment
        id = np.random.choice(self.P_max, p = np.exp(logp-log_norm)) + 1
        self.counts[id-1] += 1
        self.N_chunks     += 1
    
    def _draw_realisation(self):
        ids = np.arange(self.tot_chunks)
        np.random.shuffle(ids)
        for id in ids:
            self._add_chunk(id)
        d = dirichlet(self.counts + 1./self.P_max).rvs().flatten()
        self._initialise()
        return d
    
    def rvs(self, size):
        draws = np.array([self._draw_realisation() for _ in tqdm(range(int(size)), desc = 'Sampling')])
        return draws
        
