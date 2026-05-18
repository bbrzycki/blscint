import sys
from pathlib import Path

import numpy as np
from astropy import units as u


# Import the screen package directly so these tests do not depend on optional
# top-level blscint conveniences or observation-planning dependencies.
sys.path.insert(0, str(Path(__file__).parents[1] / "blscint" / "simulations"))

from screens import c95, hl07, rd18  # noqa: E402
from screens.base_classes import DynamicSpectrum, get_frequency_axis, get_spatial_axis  # noqa: E402


def test_dynamic_spectrum_axes_and_normalization():
    ds = DynamicSpectrum(
        intensity=np.array([[1.0, 3.0], [2.0, 4.0]]),
        frequencies=get_frequency_axis(1 * u.GHz, 1 * u.kHz, 2),
        spatial_axis=get_spatial_axis(2, 10 * u.cm),
    )

    normalized = ds.normalized("mean").with_time_axis(5 * u.cm / u.s)

    assert normalized.intensity.shape == (2, 2)
    assert np.isclose(np.mean(normalized.intensity), 1.0)
    assert normalized.frequencies.unit.is_equivalent(u.Hz)
    assert normalized.spatial_axis.unit.is_equivalent(u.cm)
    assert normalized.time_axis.unit.is_equivalent(u.s)


def test_fresnel_matches_angular_spectrum_in_paraxial_limit():
    q_mag = np.array([0.0, 1e-5, 2e-5]) / u.cm
    fresnel = hl07.fresnel_transfer_function(q_mag, 1e5 * u.cm, 1 * u.GHz)
    angular = hl07.angular_spectrum_transfer_function(q_mag, 1e5 * u.cm, 1 * u.GHz)

    assert np.allclose(fresnel, angular, atol=1e-5)


def test_c95_dynamic_spectrum_result_smoke():
    source = c95.RadioSource(2e13 * u.cm)
    screen = c95.Screen(
        distance=1e13 * u.cm,
        dx=1e9 * u.cm,
        dy=1e9 * u.cm,
        dz=1e12 * u.cm,
        shape=(8, 8),
        m_b2=0.1,
        seed=1,
    )
    model = c95.ScatteringModel(
        source,
        [screen],
        dx=1e9 * u.cm,
        dy=1e9 * u.cm,
        shape=(8, 8),
    )

    ds = model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        1 * u.kHz,
        3,
        v_trans=1e6 * u.cm / u.s,
        normalize="mean",
        progress=False,
    )

    assert ds.intensity.shape == (8, 3)
    assert np.all(np.isfinite(ds.intensity))
    assert np.isclose(np.mean(ds.intensity), 1.0)
    assert ds.time_axis.unit.is_equivalent(u.s)


def test_rd18_dynamic_spectrum_result_smoke():
    source = rd18.RadioSource(2e10 * u.m)
    screen = rd18.Screen(
        distance=1e10 * u.m,
        N=8,
        dr=5e5 * u.m,
        dz=1e10 * u.m,
        C_n2=1e-3 * u.m ** (-20 / 3),
        seed=1,
    )
    model = rd18.ScatteringModel(source, [screen], N=8, dr=5e5 * u.m)

    ds = model.observer_dynamic_spectrum_result(
        270 * u.MHz,
        1 * u.kHz,
        3,
        v_trans=1e5 * u.m / u.s,
        normalize="mean",
        progress=False,
    )

    assert ds.intensity.shape == (8, 3)
    assert np.all(np.isfinite(ds.intensity))
    assert np.isclose(np.mean(ds.intensity), 1.0)
    assert ds.time_axis.unit.is_equivalent(u.s)
