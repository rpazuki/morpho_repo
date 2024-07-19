import numpy as np
import scipy as sp

def sample_matrix(n, p_diag, p_off_diag):
    """Sample a random matrix
       n:          the size of the matrix (n,n).
       p_diag:     The function that samples n elements for diagonal terms
                   it must have a signiture like p_diag(n).
       p_off_diag: The function that samples (n,n) elements for off-diagonal terms
                   it must have a signiture like p_off_diag(n).
    """
        
    # Create an n by n matrices, sampled from 'p_off_diag' distribution
    m = p_off_diag(n)
    # Fill the diagonal terms by sampling from 'p_diag' distribution
    off_diag_indices = np.arange(0,n).astype(np.int32)
    m[off_diag_indices, off_diag_indices] = p_diag(n)
    
    return m

def matrix_intervals(m, axis=1):
    """Turn a matrix to row(column)-wise intervals
    
       args:
           m: (n,n) numpy array.
           axis:  use one for row-wise and zero for
                  column-wise intervals.
    """
    # Get the diagonal terms
    ds = np.diag(m)
    # Sum of absolute value of off-diagonal elements
    rs = np.sum(np.abs(m - np.diag(ds)), axis = axis)
    # create intrvals around the diagonal terms
    return np.array([ (d-r, d+r) for d, r in zip(ds, rs) ])

def alg1(m, axis=1):
    """Specifies the stabaility property of the matrix.
    
       args:
           m: (n,n) numpy array.
           axis:  use one for row-wise and zero for
                  column-wise intervals.
       return:
               0 - super-stable
               1 - inconclusive
               2 - unstable
    
    """
    n = m.shape[0]
    intervals = matrix_intervals(m,axis)
    us = np.array([u for _,u in intervals])
    ls = np.array([l for l,_ in intervals])
    u_max_index = np.argmax(us)
    u_max = us[u_max_index]    
    if u_max < 0:# Super-stable
        return 0
    
    l_i = ls[u_max_index] 
    if l_i < 0:# Inconclusive
        return 1
    indices = [i for i in range(0, n) if i != u_max_index]
    for j in indices:
        l_j, u_j = ls[j], us[j]
        if l_i < u_j:
            if l_j < l_i:
                l_i = l_j
            if l_i < 0:# Inconclusive
                return 1
    return 2# unstable            

def alg2(m, axis=1):
    """Specifies if the stabaility property can be tightened.
    
       args:
           m: (n,n) numpy array.
           axis:  use one for row-wise and zero for
                  column-wise intervals.
       return:
               0 - super-stable
               1 - inconclusive
               2 - unstable
    
    """
    n = m.shape[0]
    # Get the diagonal terms
    ds = np.diag(m)
    # Sum of absolute value of off-diagonal elements
    rs = np.sum(np.abs(m - np.diag(ds)), axis = axis)
    a_max_index = np.argmax(ds)
    a_ii = ds[a_max_index]
    r_i = rs[a_max_index]
    indices = [i for i in range(0, n) if i != a_max_index]
    if a_ii > 0:        
        for j in indices:
            r_j = rs[j]
            a_jj = m[j,j]
            if axis == 0:
                a_ji = m[j, a_max_index]
            else:
                a_ji = m[a_max_index, j]
            c_0 = r_i
            c_1 = a_jj-a_ii+r_j-np.abs(a_ji)
            c_2 = np.abs(a_ji)                        
            d_max = np.real(np.max(np.roots([c_2, c_1, c_0])))
            if d_max <= r_i/a_ii:
                return 1 # Inconclusive
            if c_1 >= 0:                
                return 1 # Inconclusive
            if c_1*c_1 <= 4*np.abs(a_ji*r_i):                
                return 1 # Inconclusive
        return 2 # Unstable
    else:
        # select r_i, a_jj and a_ji
        if axis == 0:
            parts = [(rs[j], m[j,j], m[j, a_max_index]) for j in indices]
        else:
            parts = [(rs[j], m[j,j], m[a_max_index, j]) for j in indices]
        values = [(np.abs(a_jj)-r_j)/np.abs(a_ji)  for (r_j, a_jj, a_ji) in parts]
        if r_i/np.abs(a_ii) >= np.min(values):
            return 1 # Inconclusive
        return 0 # Super-stable

def alg(m):
    """Runs alg1 and alg2 consequtivly for rows and columns"""
    ret = alg1(m, axis = 0)
    if ret != 1:
        return ret
    ret = alg1(m, axis = 1)
    if ret != 1:
        return ret
    ret = alg2(m, axis = 0)
    if ret != 1:
        return ret
    return alg2(m, axis = 1)

### Functions for sampling diagonal/off-diagonal terms
def p_diag_uniform(low):
    """Create a uniform distribution as U(-low, 0) for diagonal terms"""
    def uniform(n):
        return np.random.uniform(low, 0, n)    
    return uniform

def p_off_diag_exp(lamb):
    """Create an exponential distribution as exp(1/lamb) for off-diagonal terms"""
    def exp(n):        
        p = np.random.uniform(0, 1)        
        return (
            np.random.exponential(1/lamb, (n,n))*
            np.where(np.random.binomial(1, p, (n, n)) == 0, -1, 1)            
        )
    return exp

def p_off_diag_normal(mean, std):
    """Create an Gaussian distribution as N(mean, std) for off-diagonal terms"""
    def normal(n):        
        return np.random.normal(mean, std, (n,n))
    return normal

def is_unstable(m):
    """Find the linear stability of a matrix"""
    return np.any(np.real(sp.linalg.eigvals(m)) > 0)
