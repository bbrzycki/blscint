import numpy as np

from blscint.simulations.screens import validation


def test_fit_power_law_recovers_slope():
    x = np.geomspace(1, 100, 40)
    y = 3.5 * x**(-5 / 3)

    result = validation.fit_power_law(x, y)

    assert np.isclose(result["slope"], -5 / 3)
    assert np.isclose(result["amplitude"], 3.5)


def test_phase_structure_function_linear_ramp():
    x = np.arange(20, dtype=float)
    field = np.tile(x, (10, 1))

    lags, dphi = validation.phase_structure_function(field, max_lag=4, axis=1)

    assert np.allclose(lags, [1, 2, 3, 4])
    assert np.allclose(dphi, lags**2)


def test_decorrelation_lag_interpolates_first_crossing():
    lags = np.array([0.0, 1.0, 2.0, 3.0])
    acf = np.array([1.0, 0.7, 0.3, 0.1])

    lag = validation.decorrelation_lag(lags, acf, level=0.5)

    assert np.isclose(lag, 1.5)


def test_peak_correlation_lag_uses_centered_coordinates():
    corr = np.zeros((5, 7))
    corr[1, 5] = 1

    row_lag, col_lag, peak = validation.peak_correlation_lag(corr)

    assert (row_lag, col_lag, peak) == (-1, 2, 1.0)


def test_exponential_gain_diagnostic_mean_normalizes():
    values = np.array([0.5, 1.0, 1.5, 2.0])

    diag = validation.exponential_gain_diagnostic(values)

    assert np.isclose(diag["mean"], 1.0)
    assert diag["ks_statistic"] >= 0
