import numpy as np
from scipy.special import logsumexp
from scipy.stats import dirichlet
from tqdm import tqdm

class SF:
    def __init__(self, log_p_matrix):
        """
        Class to infer the distribution of autoregressive order process
        
        Arguments:
            :np.ndarray log_p_matrix: matrix with precomputed log probability

        Returns:
            :SF: instance of SF class
        """
        self.log_p_matrix = log_p_matrix
        self.P_max        = log_p_matrix.shape[0]
        self.tot_chunks   = log_p_matrix.shape[-1]
        self._initialise()
    
    def _initialise(self):
        """
        Initialises the class
        """
        self.counts   = np.zeros(self.P_max)
        self.N_chunks = 0
    
    def _add_chunk(self, chunk_id):
        """
        Add chunk to P class
        
        Arguments:
            :int chunk_id: index of the chunk to be added
        """
        # Computes probability for P
        logp = np.zeros(self.P_max)
        logp = self.log_p_matrix[:,chunk_id] + np.log(self.counts + 1./self.P_max) - np.log(self.N_chunks + self.P_max)
        log_norm = logsumexp(logp)
        # Random assignment
        id = np.random.choice(self.P_max, p = np.exp(logp-log_norm)) + 1
        self.counts[id-1] += 1
        self.N_chunks     += 1
    
    def _draw_realisation(self):
        """
        Draw a single realisation of P assignments
        
        Returns:
            :np.ndarray: realisation
        """
        ids = np.arange(self.tot_chunks)
        np.random.shuffle(ids)
        for id in ids:
            self._add_chunk(id)
        d = dirichlet(self.counts + 1./self.P_max).rvs().flatten()
        self._initialise()
        return d
    
    def rvs(self, size):
        """
        Draw multiple realisations (wrapper for _draw_realisation() method)
        
        Argument:
            :int size: number of realisations to draw
        
        Returns:
            :np.ndarray: realisations
        """
        draws = np.array([self._draw_realisation() for _ in tqdm(range(int(size)), desc = 'Sampling')])
        return draws
        
