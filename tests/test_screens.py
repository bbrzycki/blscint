import sys
from pathlib import Path

import numpy as np
from astropy import units as u
import setigen as stg


# Import the screen package directly so these tests do not depend on optional
# top-level blscint conveniences or observation-planning dependencies.
sys.path.insert(0, str(Path(__file__).parents[1] / "blscint" / "simulations"))

from screens import c95, hl07, rd18, setigen_bridge, voltage  # noqa: E402
from screens.base_classes import (  # noqa: E402
    DynamicSpectrum,
    ElectricFieldSpectrum,
    get_frequency_axis,
    get_spatial_axis,
)
from screens.power_law import (  # noqa: E402
    C_n2_from_m_b2,
    C_n2_from_density_rms,
    C_n2_from_scattering_measure,
    PowerLawPhaseScreen,
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


def _small_c95_model(shape=(8, 8)):
    source = c95.RadioSource(2e13 * u.cm)
    screen = c95.Screen(
        distance=1e13 * u.cm,
        dx=1e9 * u.cm,
        dy=1e9 * u.cm,
        dz=1e12 * u.cm,
        shape=shape,
        m_b2=0.1,
        seed=1,
    )
    return c95.ScatteringModel(
        source,
        [screen],
        dx=1e9 * u.cm,
        dy=1e9 * u.cm,
        shape=shape,
    )


def _constant_transfer(value, sample_count=4, frequency_count=8):
    return ElectricFieldSpectrum(
        electric_field=np.full((sample_count, frequency_count), value, dtype=np.complex128),
        frequencies=(1 * u.GHz + np.linspace(-4, 4, frequency_count) * u.kHz),
        spatial_axis=get_spatial_axis(sample_count, 10 * u.cm),
        time_axis=np.linspace(0, sample_count - 1, sample_count) * u.s,
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


def test_electric_field_spectrum_normalization_and_dynamic_conversion():
    field = ElectricFieldSpectrum(
        electric_field=np.array([[1 + 1j, 2 + 0j], [0.5 + 0j, 1j]]),
        frequencies=get_frequency_axis(1 * u.GHz, 1 * u.kHz, 2),
        spatial_axis=get_spatial_axis(2, 10 * u.cm),
    )

    normalized = field.normalized("mean").with_time_axis(5 * u.cm / u.s)
    ds = normalized.as_dynamic_spectrum(intrinsic_intensity=2.0)

    assert np.isclose(np.mean(normalized.intensity), 1.0)
    assert np.allclose(ds.intensity, 2.0 * normalized.intensity)
    assert normalized.phase.shape == field.electric_field.shape
    assert normalized.time_axis.unit.is_equivalent(u.s)


def test_fresnel_matches_angular_spectrum_in_paraxial_limit():
    q_mag = np.array([0.0, 1e-5, 2e-5]) / u.cm
    fresnel = hl07.fresnel_transfer_function(q_mag, 1e5 * u.cm, 1 * u.GHz)
    angular = hl07.angular_spectrum_transfer_function(q_mag, 1e5 * u.cm, 1 * u.GHz)

    assert np.allclose(fresnel, angular, atol=1e-5)


def test_c95_dynamic_spectrum_result_smoke():
    model = _small_c95_model()

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


def test_c95_field_spectrum_matches_dynamic_spectrum_result():
    model = _small_c95_model()

    field = model.observer_field_spectrum_result(
        get_frequency_axis(1 * u.GHz, 1 * u.kHz, 3),
        v_trans=1e6 * u.cm / u.s,
        normalize="mean",
        progress=False,
    )
    ds = model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        1 * u.kHz,
        3,
        v_trans=1e6 * u.cm / u.s,
        normalize="mean",
        progress=False,
    )

    assert field.electric_field.shape == (8, 3)
    assert np.allclose(field.intensity, ds.intensity)
    assert field.time_axis.unit.is_equivalent(u.s)


def test_power_law_subharmonics_are_stable_between_evaluations():
    spectrum = ScatteringStrengthSpectrum(
        m_b2=1.0,
        distance=1e13 * u.cm,
        dz=1e12 * u.cm,
    )
    phase_screen = PowerLawPhaseScreen(
        shape=(16, 16),
        dx=1e9 * u.cm,
        dy=1e9 * u.cm,
        seed=5,
        subharmonic_levels=1,
    )

    first = phase_screen.phases(spectrum, 1 * u.GHz)
    second = phase_screen.phases(spectrum, 1 * u.GHz)

    assert np.allclose(first, second)


def test_c95_narrowband_tone_profile_matches_single_channel_dynamic_spectrum():
    model = _small_c95_model()

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


def test_setigen_bridge_frame_gain_and_signal_injection():
    model = _small_c95_model()
    frame = stg.Frame.from_data(
        df=1 * u.kHz,
        dt=1 * u.s,
        fch1=1 * u.GHz,
        ascending=True,
        data=np.ones((4, 3)),
    )

    gain = setigen_bridge.screen_gain_for_frame(
        model,
        frame,
        v_trans=1e6 * u.cm / u.s,
        normalize="mean",
        progress=False,
    )
    scintillated = setigen_bridge.scintillate_frame(frame, gain=gain, in_place=False)

    assert gain.intensity.shape == frame.data.shape
    assert np.allclose(frame.data, 1.0)
    assert np.allclose(scintillated.frame.data, gain.intensity)
    assert scintillated.frame.metadata["blscint_scintillation"]["operation"] == "scintillate_frame"

    clean_signal = np.full(frame.data.shape, 2.0)
    injected = setigen_bridge.add_scintillated_signal(
        frame,
        clean_signal,
        gain=gain,
        in_place=False,
    )

    assert np.allclose(injected.scintillated_signal, 2.0 * gain.intensity)
    assert np.allclose(injected.frame.data, frame.data + injected.scintillated_signal)
    assert np.allclose(frame.data, 1.0)


def test_setigen_bridge_voltage_stream_injection_matches_callable():
    model = _small_c95_model()
    stream = stg.voltage.DataStream(
        sample_rate=8 * u.GHz,
        fch1=0 * u.Hz,
        ascending=True,
        seed=1,
    )

    result = setigen_bridge.add_scintillated_voltage_signal(
        stream,
        model,
        f_start=1 * u.GHz,
        v_trans=1e6 * u.cm / u.s,
        level=2.0,
        normalize="mean",
    )
    stream.set_time(0)
    samples = stream.get_samples(16)
    times = np.linspace(0, 16 * stream.dt, 16, endpoint=False)
    expected = result.signal_functions[0](times)

    assert len(result.streams) == 1
    assert result.field.time_axis.unit.is_equivalent(u.s)
    assert samples.shape == (16,)
    assert np.all(np.isfinite(samples))
    assert np.allclose(samples, expected)


def test_voltage_iq_propagation_applies_constant_complex_transfer():
    transfer = _constant_transfer(0.5 + 0.25j)
    rng = np.random.default_rng(10)
    samples = (
        rng.standard_normal((2, 128))
        + 1j * rng.standard_normal((2, 128))
    )

    result = voltage.propagate_iq(
        samples,
        transfer,
        sample_rate=8 * u.kHz,
        center_frequency=1 * u.GHz,
        axis=-1,
        block_size=32,
    )

    assert result.samples.shape == samples.shape
    assert result.frequency_axis.unit.is_equivalent(u.Hz)
    assert result.block_time_axis.unit.is_equivalent(u.s)
    assert np.allclose(result.samples, samples * (0.5 + 0.25j), atol=1e-12)


def test_voltage_real_propagation_can_return_analytic_samples():
    transfer = _constant_transfer(np.exp(1j * 0.3), frequency_count=64)
    sample_rate = 1024 * u.Hz
    times = np.arange(1024) / sample_rate.to_value(u.Hz)
    real_voltage = np.cos(2 * np.pi * 128 * times)

    result = voltage.propagate_real_voltage(
        real_voltage,
        transfer,
        sample_rate=sample_rate,
        reference_frequency=1 * u.GHz,
        output="analytic",
        block_size=128,
        window="rectangular",
    )
    expected = voltage.analytic_signal(real_voltage) * np.exp(1j * 0.3)

    assert result.samples.shape == real_voltage.shape
    assert result.input_was_real
    assert np.iscomplexobj(result.samples)
    assert np.allclose(result.samples, expected, atol=1e-12)


def test_setigen_bridge_post_generation_voltage_propagation_supports_antenna_shape():
    transfer = _constant_transfer(2.0 + 0j, frequency_count=64)
    stream = stg.voltage.DataStream(
        sample_rate=1024 * u.Hz,
        fch1=1 * u.GHz,
        ascending=True,
        seed=1,
    )
    stream.add_constant_signal(
        f_start=1 * u.GHz + 128 * u.Hz,
        drift_rate=0 * u.Hz / u.s,
        level=1.0,
    )
    samples = stream.get_samples(256)

    result = setigen_bridge.propagate_setigen_voltage(
        samples[np.newaxis, np.newaxis, :],
        transfer=transfer,
        sample_rate=stream.sample_rate * u.Hz,
        fch1=stream.fch1 * u.Hz,
        block_size=64,
        output="analytic",
        window="rectangular",
    )
    expected = voltage.analytic_signal(samples) * 2.0

    assert result.samples.shape == (1, 1, 256)
    assert np.allclose(result.samples[0, 0], expected, atol=1e-12)


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
