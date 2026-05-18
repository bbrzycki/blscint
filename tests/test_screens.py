import sys
from pathlib import Path

import numpy as np
from astropy import units as u


# Import the screen package directly so these tests do not depend on optional
# top-level blscint conveniences or observation-planning dependencies.
sys.path.insert(0, str(Path(__file__).parents[1] / "blscint" / "simulations"))

from screens import c95, hl07, rd18  # noqa: E402
from screens.base_classes import DynamicSpectrum, get_frequency_axis, get_spatial_axis  # noqa: E402
from screens.power_law import (  # noqa: E402
    C_n2_from_m_b2,
    C_n2_from_density_rms,
    C_n2_from_scattering_measure,
    ScatteringStrengthSpectrum,
    ThinScreenSpectrum,
    density_rms_from_C_n2,
    electron_column_from_phase,
    electron_density_from_phase,
    fractional_density_rms_from_C_n2,
    m_b2_from_C_n2,
    phase_from_electron_column,
    phase_from_electron_density,
    scattering_measure_from_C_n2,
)


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


def test_c95_narrowband_tone_profile_matches_single_channel_dynamic_spectrum():
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

    profile = model.observer_narrowband_tone_profile(
        1 * u.GHz,
        intrinsic_intensity=3.0,
        v_trans=1e6 * u.cm / u.s,
    )
    field = model.observer_narrowband_tone_field(
        1 * u.GHz,
        v_trans=1e6 * u.cm / u.s,
    )
    single_channel = model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        1 * u.kHz,
        1,
        v_trans=1e6 * u.cm / u.s,
        progress=False,
    )

    assert profile.intensity.shape == (8, 1)
    assert profile.frequencies.unit.is_equivalent(u.Hz)
    assert profile.time_axis.unit.is_equivalent(u.s)
    assert profile.metadata["signal"] == "perfect_narrowband_tone"
    assert field.electric_field.shape == (8,)
    assert np.allclose(field.intensity[:, np.newaxis], single_channel.intensity)
    assert np.allclose(field.as_dynamic_spectrum(3.0).intensity, profile.intensity)
    assert np.allclose(profile.intensity, 3.0 * single_channel.intensity)

    times = np.array([0.0, 0.25, 0.5]) * u.ns
    spatial_index = 4
    analytic_voltage = field.sample_voltage(
        times,
        spatial_index=spatial_index,
        amplitude=2.0,
        analytic=True,
    )
    expected = 2.0 * field.electric_field[spatial_index] * np.exp(
        1j * 2 * np.pi * (1 * u.GHz * times).decompose().value
    )

    assert np.allclose(analytic_voltage, expected)
    assert np.allclose(
        field.sample_voltage(times, spatial_index=spatial_index, amplitude=2.0),
        np.real(expected),
    )

    cycle_times = np.linspace(0, 64, 4096, endpoint=False) * u.ns
    voltage = field.sample_voltage(cycle_times, spatial_index=spatial_index)
    assert np.isclose(
        2 * np.mean(voltage**2),
        field.intensity[spatial_index],
        rtol=1e-12,
    )


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


def test_c95_rd18_match_with_shared_parameters_and_phase_sign():
    shape = (16, 16)
    dx = 5e5 * u.cm
    source_distance = 2e13 * u.cm
    screen_distance = 1e13 * u.cm
    dz = 1e12 * u.cm
    m_b2 = 5
    seed = 7

    c95_screen = c95.Screen(
        distance=screen_distance,
        dx=dx,
        dy=dx,
        dz=dz,
        shape=shape,
        m_b2=m_b2,
        seed=seed,
    )
    rd18_screen = rd18.Screen(
        distance=screen_distance,
        N=shape[0],
        dr=dx,
        dz=dz,
        m_b2=m_b2,
        phase_sign=1,
        seed=seed,
    )

    assert np.allclose(c95_screen.phases(1 * u.GHz), rd18_screen.phases(1 * u.GHz))

    c95_model = c95.ScatteringModel(
        c95.RadioSource(source_distance),
        [c95_screen],
        dx=dx,
        dy=dx,
        shape=shape,
    )
    rd18_model = rd18.ScatteringModel(
        rd18.RadioSource(source_distance),
        [rd18_screen],
        N=shape[0],
        dr=dx,
    )

    c95_ds = c95_model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        1 * u.MHz,
        4,
        normalize="mean",
        progress=False,
    )
    rd18_ds = rd18_model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        1 * u.MHz,
        4,
        normalize="mean",
        progress=False,
    )

    assert np.allclose(c95_ds.intensity, rd18_ds.intensity)


def test_m_b2_and_C_n2_conversion_round_trip_and_spectrum_equivalence():
    f = 1 * u.GHz
    distance = 1e13 * u.cm
    dz = 1e12 * u.cm
    m_b2 = 10.0

    C_n2 = C_n2_from_m_b2(m_b2, f, distance, dz)

    assert C_n2.unit.is_equivalent(u.m ** (-20 / 3))
    assert np.isclose(m_b2_from_C_n2(C_n2, f, distance, dz), m_b2)

    coles_spectrum = ScatteringStrengthSpectrum(
        m_b2=m_b2,
        distance=distance,
        dz=dz,
    )
    thin_screen_spectrum = ThinScreenSpectrum(
        C_n2=C_n2,
        distance=distance,
        dz=dz,
    )

    q = np.array([1e-8, 3e-8]) / u.cm

    assert np.isclose(thin_screen_spectrum.m_b2_at(f), m_b2)
    assert np.isclose(
        coles_spectrum.C_n2(f).to_value(u.m ** (-20 / 3)),
        C_n2.to_value(u.m ** (-20 / 3)),
    )
    assert np.allclose(
        coles_spectrum.Phi(q, f).decompose().value,
        thin_screen_spectrum.Phi(q, f).decompose().value,
    )


def test_phase_column_density_and_density_fluctuation_conversions():
    f = 1 * u.GHz
    dz = 1e12 * u.cm
    electron_column = 2e12 / u.cm**2
    delta_n_e = 0.2 / u.cm**3

    phase = phase_from_electron_column(electron_column, f)
    assert electron_column_from_phase(phase, f).unit.is_equivalent(1 / u.cm**2)
    assert np.isclose(
        electron_column_from_phase(phase, f).to_value(1 / u.cm**2),
        electron_column.to_value(1 / u.cm**2),
    )

    density_phase = phase_from_electron_density(delta_n_e, dz, f)
    assert np.isclose(
        electron_density_from_phase(density_phase, dz, f).to_value(1 / u.cm**3),
        delta_n_e.to_value(1 / u.cm**3),
    )

    C_n2 = 1e-4 * u.m ** (-20 / 3)
    SM = scattering_measure_from_C_n2(C_n2, 1 * u.kpc)
    assert SM.unit.is_equivalent(u.kpc * u.m ** (-20 / 3))
    assert np.isclose(
        C_n2_from_scattering_measure(SM, 1 * u.kpc).to_value(u.m ** (-20 / 3)),
        C_n2.to_value(u.m ** (-20 / 3)),
    )

    outer_scale = 1 * u.pc
    inner_scale = 1000 * u.km
    delta_n_e_rms = density_rms_from_C_n2(
        C_n2,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )
    round_trip_C_n2 = C_n2_from_density_rms(
        delta_n_e_rms,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )
    frac = fractional_density_rms_from_C_n2(
        C_n2,
        mean_n_e=0.03 / u.cm**3,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
    )

    assert delta_n_e_rms.unit.is_equivalent(1 / u.m**3)
    assert frac > 0
    assert np.isclose(
        round_trip_C_n2.to_value(u.m ** (-20 / 3)),
        C_n2.to_value(u.m ** (-20 / 3)),
    )
