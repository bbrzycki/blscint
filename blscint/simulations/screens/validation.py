"""Quantitative diagnostics for screen-propagation simulations."""

import numpy as np
from scipy import signal, stats


def fit_power_law(x, y, xmin=None, xmax=None):
    """Fit ``y = amplitude * x**slope`` in log-log space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if xmin is not None:
        mask &= x >= xmin
    if xmax is not None:
        mask &= x <= xmax
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two finite positive samples are required")

    slope, intercept = np.polyfit(np.log10(x[mask]), np.log10(y[mask]), deg=1)
    return {
        "slope": slope,
        "amplitude": 10**intercept,
        "mask": mask,
        "x": x[mask],
        "y": y[mask],
    }


def phase_structure_function(field, spacing=1.0, max_lag=None, axis=1):
    """Estimate a one-dimensional phase structure function."""
    field = np.asarray(field, dtype=float)
    if field.ndim != 2:
        raise ValueError("field must be 2-D")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    if max_lag is None:
        max_lag = field.shape[axis] // 3
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")

    lags = np.arange(1, max_lag + 1)
    values = []
    for lag in lags:
        if axis == 1:
            delta = field[:, lag:] - field[:, :-lag]
        else:
            delta = field[lag:, :] - field[:-lag, :]
        values.append(np.mean(delta**2))
    return lags * spacing, np.asarray(values)


def radial_power_spectrum_2d(field, dx=1.0, dy=1.0, bins=32):
    """Return an azimuthally averaged 2-D power spectrum."""
    field = np.asarray(field, dtype=float)
    if field.ndim != 2:
        raise ValueError("field must be 2-D")
    ny, nx = field.shape
    centered = field - np.mean(field)
    fft_power = np.abs(np.fft.fftshift(np.fft.fft2(centered)))**2
    qx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2 * np.pi
    qy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2 * np.pi
    qx_grid, qy_grid = np.meshgrid(qx, qy)
    q_mag = np.hypot(qx_grid, qy_grid)

    valid = q_mag > 0
    q_values = q_mag[valid]
    power_values = fft_power[valid]
    edges = np.geomspace(q_values.min(), q_values.max(), int(bins) + 1)
    which = np.digitize(q_values, edges) - 1

    q_centers = []
    radial_power = []
    counts = []
    for idx in range(int(bins)):
        in_bin = which == idx
        if not np.any(in_bin):
            continue
        q_centers.append(np.exp(np.mean(np.log(q_values[in_bin]))))
        radial_power.append(np.mean(power_values[in_bin]))
        counts.append(np.count_nonzero(in_bin))
    return np.asarray(q_centers), np.asarray(radial_power), np.asarray(counts)


def normalized_autocorrelation_1d(values):
    """Return non-negative lags of a normalized one-dimensional ACF."""
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    corr = signal.correlate(centered, centered, mode="full")
    lags = signal.correlation_lags(values.size, values.size, mode="full")
    keep = lags >= 0
    corr = corr[keep]
    lags = lags[keep]
    if corr[0] == 0:
        return lags, np.full_like(corr, np.nan, dtype=float)
    return lags, corr / corr[0]


def mean_axis_autocorrelation(data, axis=0):
    """Average normalized ACFs along one axis of a 2-D array."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be 2-D")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")

    arr = data if axis == 0 else data.T
    acfs = []
    for idx in range(arr.shape[1]):
        _, acf = normalized_autocorrelation_1d(arr[:, idx])
        if np.all(np.isfinite(acf)):
            acfs.append(acf)
    if not acfs:
        raise ValueError("no finite ACFs could be computed")
    return np.arange(arr.shape[0]), np.nanmean(acfs, axis=0)


def normalized_cross_correlation_2d(reference, candidate):
    """Return a normalized 2-D cross-correlation map."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    dref = reference - np.mean(reference)
    dcand = candidate - np.mean(candidate)
    corr = signal.correlate2d(dcand, dref, mode="same", boundary="fill")
    denom = np.sqrt(np.sum(dref**2) * np.sum(dcand**2))
    if denom == 0:
        return np.full_like(corr, np.nan, dtype=float)
    return corr / denom


def peak_correlation_lag(corr):
    """Return ``(row_lag, column_lag, peak_value)`` for a centered 2-D map."""
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2:
        raise ValueError("corr must be 2-D")
    row, col = np.unravel_index(np.nanargmax(corr), corr.shape)
    center_row = (corr.shape[0] - 1) // 2
    center_col = (corr.shape[1] - 1) // 2
    return int(row - center_row), int(col - center_col), float(corr[row, col])


def decorrelation_lag(lags, acf, level=np.exp(-1)):
    """Interpolate the first lag where an ACF falls to ``level``."""
    lags = np.asarray(lags, dtype=float)
    acf = np.asarray(acf, dtype=float)
    if lags.shape != acf.shape:
        raise ValueError("lags and acf must have the same shape")
    below = np.flatnonzero(acf <= level)
    below = below[below > 0]
    if below.size == 0:
        return np.nan
    idx = below[0]
    x0, x1 = lags[idx - 1], lags[idx]
    y0, y1 = acf[idx - 1], acf[idx]
    if y0 == y1:
        return x1
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def exponential_gain_diagnostic(intensity):
    """Compare mean-normalized intensity samples to an exponential gain law."""
    values = np.asarray(intensity, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("intensity must contain finite samples")
    normalized = values / np.mean(values)
    ks = stats.kstest(normalized, "expon")
    return {
        "mean": float(np.mean(normalized)),
        "variance": float(np.var(normalized)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }
