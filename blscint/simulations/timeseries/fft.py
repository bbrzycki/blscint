import numpy as np
from blscint import diag_stats


def get_ts_fft(t_d, dt, num_samples, pow=5/3, seed=None):
    """
    Produce time series data via FFT/IFFT. Based on code from Jim Cordes.

    Parameters
    ----------
    t_d : float
        Scintillation timescale (s)
    dt : float
        Time resolution (s)
    num_samples : int
        Number of synthetic samples to produce
    pow : float, optional
        Exponent for ACF fit, either 5/3 or 2 (arising from phase structure function) 
    seed : None, int, Generator, optional
        Random seed or seed generator

    Returns
    -------
    Y : np.ndarray
        Final synthetic scintillated time series
    """
    rng = np.random.default_rng(seed)
    lags = np.linspace(0, num_samples - 1, num_samples) - num_samples / 2
    lags = np.fft.fftshift(lags)

    noise = (rng.standard_normal(num_samples) + 1j * rng.standard_normal(num_samples)) / np.sqrt(2)
    spectrum = np.fft.fft(diag_stats.scint_acf(lags, 
                                               t_d / dt * 2**(1/pow), 
                                               pow=pow)).real

    I = np.abs(np.fft.ifft(noise * np.sqrt(spectrum)))**2
    return I / np.mean(I)
