"""
Generate scintillated signals matching Gaussian pulse profiles and exponential
intensity distributions using theoretical PDFs. 

Scintillation on narrowband signals references:

Cordes & Lazio 1991:
http://articles.adsabs.harvard.edu/pdf/1991ApJ...376..123C

Cordes, Lazio, & Sagan 1997:
https://iopscience.iop.org/article/10.1086/304620/pdf
"""
import sys
import numpy as np
from scipy.stats import norm
import scipy.linalg

import setigen as stg
from setigen.funcs import func_utils
from . import factors
from . import ts_statistics


def find_nearest(arr, val):
    """
    Return index of closest value.
    """
    idx = (np.abs(np.array(arr) - val)).argmin()
    return idx


def bpdf(g1, g2, ac):
    """
    Calculate join probability of g1, g2 separated by a temporal auto-correlation value of ac,
    according to Eq. B12 of Cordes, Lazio, & Sagan 1997.
    """
    # Calculate modified Bessel term, which can diverge.
    i_factor = scipy.special.i0(2*np.sqrt(g1*g2*ac)/(1-ac))
    # In divergent case, use exponential approximation to simplify terms.
    if np.any(np.isinf(i_factor)):
        #https://math.stackexchange.com/questions/376758/exponential-approximation-of-the-modified-bessel-function-of-first-kind-equatio
        return 1/(1-ac)*np.exp((-(g1+g2)+2*np.sqrt(g1*g2*ac))/(1-ac))*np.sqrt(4*np.pi*np.sqrt(g1*g2*ac)/(1-ac))
    else:
        return 1/(1-ac)*np.exp(-(g1+g2)/(1-ac))*i_factor

    
def get_ts_pdf(t_d, dt, num_samples, max_g=5, steps=1000):
    """
    Produce time series data via bivariate pdf for the gain.
    """
    ac_arr = stg.func_utils.gaussian(np.arange(0, steps),
                                     0, 
                                     t_d / dt / factors.hwem_m)
    
    possible_g = np.linspace(0, max_g, steps, endpoint=False)
#     F_2g = np.empty(shape=(n, n))
#     for i in range(n):
#         F_2g[i, :] = bpdf(possible_g[i], possible_g, rho[1])
#     F_2g = F_2g / np.sum(F_2g, axis=1, keepdims=True)

    ts_idx = np.zeros(num_samples, dtype=int)

    init_g = max_g + 1
    while init_g > max_g:
        init_g = np.random.exponential()
    ts_idx[0] = find_nearest(possible_g, init_g)
    
    update_freq = int(np.ceil(t_d / dt))
    for i in range(1, num_samples):
#         offset = i % update_freq
#         if offset == 1:
#             last_i = i - 1
#         if offset == 0:
#             offset = update_freq
#         raw_p = bpdf(possible_g[ts_idx[last_i]], possible_g, ac_arr[offset])
        
        raw_p = bpdf(possible_g[ts_idx[i-1]], possible_g, ac_arr[1])

        p = raw_p / np.sum(raw_p)
        try:
            ts_idx[i] = np.random.choice(np.arange(steps), p=p)
        except:
#             print(F_2g[i])
            print(i)
            sys.exit(1)
    Y = possible_g[ts_idx]
    return Y