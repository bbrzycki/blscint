"""Voltage and IQ propagation helpers for screen transfer functions.

The functions in this module are intentionally array-oriented.  They do not
depend on setigen or GNU Radio object models; callers pass NumPy arrays plus
the sample-rate and frequency convention needed to map FFT bins onto physical
RF frequencies.  Package-specific integrations can wrap these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict

import numpy as np
from astropy import units as u
import setigen as stg

from .base_classes import ElectricFieldSpectrum


@dataclass(frozen=True)
class IQPropagationResult:
    """Result of applying a complex screen transfer to voltage/IQ samples."""

    samples: np.ndarray
    transfer: ElectricFieldSpectrum
    sample_rate: u.Quantity
    center_frequency: u.Quantity
    frequency_axis: u.Quantity
    block_time_axis: u.Quantity
    block_size: int
    hop_size: int
    input_was_real: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_quantity(value: Any, unit: u.Unit) -> u.Quantity:
    return stg.cast_value(value, unit)


def _axis_values(axis: u.Quantity, unit: u.Unit) -> np.ndarray:
    return np.asarray(_as_quantity(axis, unit).to_value(unit), dtype=float)


def _align_time_values(
    target_times: u.Quantity,
    source_times: u.Quantity,
    time_origin: str,
) -> np.ndarray:
    target = _axis_values(target_times, u.s)
    source = _axis_values(source_times, u.s)
    if len(target) == 0:
        return target
    if time_origin == "start":
        return target - target[0] + source[0]
    if time_origin == "center":
        return target - 0.5 * (target[0] + target[-1]) + 0.5 * (source[0] + source[-1])
    if time_origin == "zero":
        return target
    raise ValueError("time_origin must be 'start', 'center', or 'zero'")


def _prepare_targets(
    target: np.ndarray,
    source: np.ndarray,
    edge_mode: str,
) -> np.ndarray:
    if len(source) == 1:
        return np.full_like(target, source[0], dtype=float)
    lo = float(source[0])
    hi = float(source[-1])
    if edge_mode == "clip":
        return np.clip(target, lo, hi)
    if edge_mode == "wrap":
        period = hi - lo
        if period <= 0:
            return np.clip(target, lo, hi)
        return ((target - lo) % period) + lo
    if edge_mode == "raise":
        if np.any((target < lo) | (target > hi)):
            raise ValueError("target coordinates extend beyond the sampled screen axis")
        return target
    raise ValueError("edge_mode must be 'clip', 'wrap', or 'raise'")


def _interp_vector(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    values: np.ndarray,
    edge_mode: str,
) -> np.ndarray:
    if len(source_axis) == len(target_axis) and np.allclose(source_axis, target_axis):
        return np.array(values, copy=True)
    if len(source_axis) == 1:
        return np.repeat(values[:1], len(target_axis), axis=0)

    order = np.argsort(source_axis)
    source_axis = source_axis[order]
    values = values[order]
    target_axis = _prepare_targets(target_axis, source_axis, edge_mode)

    flat = values.reshape(len(source_axis), -1)
    out = np.empty((len(target_axis), flat.shape[1]), dtype=values.dtype)
    if np.iscomplexobj(values):
        for idx in range(flat.shape[1]):
            out[:, idx] = (
                np.interp(target_axis, source_axis, flat[:, idx].real)
                + 1j * np.interp(target_axis, source_axis, flat[:, idx].imag)
            )
    else:
        for idx in range(flat.shape[1]):
            out[:, idx] = np.interp(target_axis, source_axis, flat[:, idx])
    return out.reshape((len(target_axis),) + values.shape[1:])


def _interp_axis(
    values: np.ndarray,
    source_axis: u.Quantity,
    target_axis: u.Quantity,
    unit: u.Unit,
    axis: int,
    edge_mode: str,
) -> np.ndarray:
    source = _axis_values(source_axis, unit)
    target = _axis_values(target_axis, unit)
    moved = np.moveaxis(np.asarray(values), axis, 0)
    interpolated = _interp_vector(source, target, moved, edge_mode)
    return np.moveaxis(interpolated, 0, axis)


def iq_frequency_axis(
    sample_rate: u.Quantity,
    center_frequency: u.Quantity,
    fft_size: int,
    *,
    ascending: bool = True,
) -> u.Quantity:
    """Return physical RF frequencies for complex IQ FFT bins.

    ``center_frequency`` is the RF frequency corresponding to zero baseband
    frequency.  Set ``ascending=False`` for conventions where positive FFT
    frequency corresponds to lower RF frequency.
    """
    fft_size = int(fft_size)
    if fft_size < 1:
        raise ValueError("fft_size must be at least 1")
    sample_rate = _as_quantity(sample_rate, u.Hz)
    center_frequency = _as_quantity(center_frequency, u.Hz)
    offsets = np.fft.fftfreq(fft_size, d=1 / sample_rate.to_value(u.Hz)) * u.Hz
    if not ascending:
        offsets = -offsets
    return center_frequency + offsets


def field_spectrum_for_baseband(
    model: Any,
    *,
    sample_rate: u.Quantity,
    center_frequency: u.Quantity,
    v_trans: u.Quantity,
    bandwidth: u.Quantity | None = None,
    frequency_samples: int = 256,
    normalize: str | None = "mean",
    progress: bool = True,
) -> ElectricFieldSpectrum:
    """Sample a screen model over a baseband/IQ frequency span."""
    if not hasattr(model, "observer_field_spectrum_result"):
        raise TypeError("model must provide observer_field_spectrum_result(frequencies, ...)")
    sample_rate = _as_quantity(sample_rate, u.Hz)
    center_frequency = _as_quantity(center_frequency, u.Hz)
    if bandwidth is None:
        bandwidth = sample_rate
    bandwidth = _as_quantity(bandwidth, u.Hz)
    frequency_samples = int(frequency_samples)
    if frequency_samples < 2:
        raise ValueError("frequency_samples must be at least 2")
    offsets = np.linspace(
        -0.5 * bandwidth.to_value(u.Hz),
        0.5 * bandwidth.to_value(u.Hz),
        frequency_samples,
    ) * u.Hz
    return model.observer_field_spectrum_result(
        center_frequency + offsets,
        v_trans=v_trans,
        normalize=normalize,
        progress=progress,
    )


def sample_electric_field_transfer(
    transfer: ElectricFieldSpectrum,
    *,
    times: u.Quantity,
    frequencies: u.Quantity,
    time_origin: str = "start",
    edge_mode: str = "clip",
) -> np.ndarray:
    """Interpolate a complex field transfer onto time/frequency targets."""
    values = np.asarray(transfer.electric_field)
    target_times = _as_quantity(times, u.s)
    if transfer.time_axis is None:
        if values.shape[0] != 1:
            raise ValueError("transfer must have time_axis unless it has one time sample")
        values = np.repeat(values, len(np.atleast_1d(target_times.value)), axis=0)
    else:
        aligned_times = _align_time_values(target_times, transfer.time_axis, time_origin)
        values = _interp_axis(
            values,
            transfer.time_axis,
            aligned_times * u.s,
            u.s,
            axis=0,
            edge_mode=edge_mode,
        )

    return _interp_axis(
        values,
        transfer.frequencies,
        _as_quantity(frequencies, u.Hz),
        u.Hz,
        axis=1,
        edge_mode=edge_mode,
    )


def analytic_signal(samples: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Return the complex analytic representation of a real voltage array."""
    samples = np.asarray(samples)
    moved = np.moveaxis(samples, axis, -1)
    n = moved.shape[-1]
    if n < 1:
        raise ValueError("samples axis must not be empty")

    weights = np.zeros(n)
    weights[0] = 1
    if n % 2 == 0:
        weights[1:n // 2] = 2
        weights[n // 2] = 1
    else:
        weights[1:(n + 1) // 2] = 2

    shaped_weights = weights.reshape((1,) * (moved.ndim - 1) + (n,))
    analytic = np.fft.ifft(np.fft.fft(moved, axis=-1) * shaped_weights, axis=-1)
    return np.moveaxis(analytic, -1, axis)


def _window_values(window: str | None, block_size: int) -> tuple[np.ndarray, str]:
    if window in (None, "none", "boxcar", "rectangular"):
        return np.ones(block_size), "rectangular"
    if window in ("sqrt_hann", "hann"):
        n = np.arange(block_size)
        return np.sin(np.pi * (n + 0.5) / block_size), "sqrt_hann"
    raise ValueError("window must be 'sqrt_hann', 'hann', 'rectangular', or None")


def _default_hop_size(window_name: str, block_size: int) -> int:
    if window_name == "rectangular":
        return block_size
    return max(1, block_size // 2)


def _flatten_sample_axis(samples: np.ndarray, axis: int) -> tuple[np.ndarray, tuple[int, ...], int]:
    moved = np.moveaxis(samples, axis, -1)
    original_prefix = moved.shape[:-1]
    sample_count = moved.shape[-1]
    flat = moved.reshape((-1, sample_count))
    return flat, original_prefix, sample_count


def _restore_sample_axis(flat: np.ndarray, original_prefix: tuple[int, ...], axis: int) -> np.ndarray:
    moved = flat.reshape(original_prefix + (flat.shape[-1],))
    return np.moveaxis(moved, -1, axis)


def propagate_iq(
    samples: np.ndarray,
    transfer: ElectricFieldSpectrum,
    *,
    sample_rate: u.Quantity,
    center_frequency: u.Quantity,
    axis: int = -1,
    block_size: int = 1024,
    hop_size: int | None = None,
    window: str | None = "sqrt_hann",
    ascending: bool = True,
    time_origin: str = "start",
    edge_mode: str = "clip",
) -> IQPropagationResult:
    """Apply a complex screen transfer to complex IQ samples.

    The transfer is treated as constant within each FFT block and interpolated
    onto that block's center time and FFT-bin frequencies.  All dimensions
    other than ``axis`` are treated as independent voltage streams, so arrays
    shaped like ``(pol, samples)`` or ``(antenna, pol, samples)`` work without
    special handling.
    """
    samples = np.asarray(samples)
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.complex128)
    else:
        samples = samples.astype(np.complex128, copy=False)
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be at least 1")

    window_values, window_name = _window_values(window, block_size)
    if hop_size is None:
        hop_size = _default_hop_size(window_name, block_size)
    hop_size = int(hop_size)
    if hop_size < 1:
        raise ValueError("hop_size must be at least 1")
    if hop_size > block_size:
        raise ValueError("hop_size must not exceed block_size")

    sample_rate = _as_quantity(sample_rate, u.Hz)
    center_frequency = _as_quantity(center_frequency, u.Hz)
    sample_rate_value = sample_rate.to_value(u.Hz)
    flat, original_prefix, sample_count = _flatten_sample_axis(samples, axis)
    output = np.zeros((flat.shape[0], sample_count + block_size), dtype=np.complex128)
    weight_sum = np.zeros(sample_count + block_size)
    fft_frequencies = iq_frequency_axis(
        sample_rate,
        center_frequency,
        block_size,
        ascending=ascending,
    )

    starts = list(range(0, sample_count, hop_size))
    block_times = np.array(
        [(start + 0.5 * (block_size - 1)) / sample_rate_value for start in starts]
    ) * u.s
    for start, block_time in zip(starts, block_times):
        valid = min(block_size, sample_count - start)
        frame = np.zeros((flat.shape[0], block_size), dtype=np.complex128)
        frame[:, :valid] = flat[:, start:start + valid]
        frame *= window_values[np.newaxis, :]

        block_transfer = sample_electric_field_transfer(
            transfer,
            times=np.array([block_time.to_value(u.s)]) * u.s,
            frequencies=fft_frequencies,
            time_origin=time_origin,
            edge_mode=edge_mode,
        )[0]
        propagated = np.fft.ifft(
            np.fft.fft(frame, axis=-1) * block_transfer[np.newaxis, :],
            axis=-1,
        )
        output[:, start:start + block_size] += propagated * window_values[np.newaxis, :]
        weight_sum[start:start + block_size] += window_values**2

    output = output[:, :sample_count]
    valid_weights = weight_sum[:sample_count] > np.finfo(float).eps
    output[:, valid_weights] /= weight_sum[:sample_count][valid_weights]
    restored = _restore_sample_axis(output, original_prefix, axis)

    return IQPropagationResult(
        samples=restored,
        transfer=transfer,
        sample_rate=sample_rate,
        center_frequency=center_frequency,
        frequency_axis=fft_frequencies,
        block_time_axis=block_times,
        block_size=block_size,
        hop_size=hop_size,
        input_was_real=False,
        metadata={
            "mode": "complex_iq",
            "window": window_name,
            "ascending": bool(ascending),
            "time_origin": time_origin,
            "edge_mode": edge_mode,
        },
    )


def propagate_real_voltage(
    samples: np.ndarray,
    transfer: ElectricFieldSpectrum,
    *,
    sample_rate: u.Quantity,
    reference_frequency: u.Quantity,
    axis: int = -1,
    output: str = "real",
    block_size: int = 1024,
    hop_size: int | None = None,
    window: str | None = "sqrt_hann",
    ascending: bool = True,
    time_origin: str = "start",
    edge_mode: str = "clip",
) -> IQPropagationResult:
    """Apply a screen transfer to real voltage via its analytic signal.

    ``reference_frequency`` is the RF frequency corresponding to zero
    post-sampling frequency.  For setigen voltage streams this is typically
    the stream's ``fch1``; for ordinary complex baseband use
    :func:`propagate_iq` directly.
    """
    if output not in ("real", "analytic"):
        raise ValueError("output must be 'real' or 'analytic'")
    analytic = analytic_signal(np.asarray(samples), axis=axis)
    result = propagate_iq(
        analytic,
        transfer,
        sample_rate=sample_rate,
        center_frequency=reference_frequency,
        axis=axis,
        block_size=block_size,
        hop_size=hop_size,
        window=window,
        ascending=ascending,
        time_origin=time_origin,
        edge_mode=edge_mode,
    )
    output_samples = result.samples if output == "analytic" else np.real(result.samples)
    return replace(
        result,
        samples=output_samples,
        input_was_real=True,
        metadata={
            **dict(result.metadata),
            "mode": "real_voltage",
            "output": output,
        },
    )


def propagate_voltage_samples(
    samples: np.ndarray,
    transfer: ElectricFieldSpectrum,
    *,
    sample_rate: u.Quantity,
    reference_frequency: u.Quantity,
    axis: int = -1,
    input_mode: str = "auto",
    output: str = "same",
    block_size: int = 1024,
    hop_size: int | None = None,
    window: str | None = "sqrt_hann",
    ascending: bool = True,
    time_origin: str = "start",
    edge_mode: str = "clip",
) -> IQPropagationResult:
    """Dispatch to real-voltage or complex-IQ propagation.

    ``input_mode='auto'`` treats complex arrays as IQ and real arrays as real
    voltages.  For real input, ``output='same'`` returns real samples and
    ``output='analytic'`` returns complex analytic samples.  For complex input,
    the output is always complex.
    """
    samples = np.asarray(samples)
    if input_mode not in ("auto", "real", "complex"):
        raise ValueError("input_mode must be 'auto', 'real', or 'complex'")
    if input_mode == "auto":
        input_mode = "complex" if np.iscomplexobj(samples) else "real"
    if input_mode == "real":
        real_output = "real" if output == "same" else output
        return propagate_real_voltage(
            samples,
            transfer,
            sample_rate=sample_rate,
            reference_frequency=reference_frequency,
            axis=axis,
            output=real_output,
            block_size=block_size,
            hop_size=hop_size,
            window=window,
            ascending=ascending,
            time_origin=time_origin,
            edge_mode=edge_mode,
        )
    if output not in ("same", "complex", "analytic"):
        raise ValueError("complex input supports output 'same', 'complex', or 'analytic'")
    return propagate_iq(
        samples,
        transfer,
        sample_rate=sample_rate,
        center_frequency=reference_frequency,
        axis=axis,
        block_size=block_size,
        hop_size=hop_size,
        window=window,
        ascending=ascending,
        time_origin=time_origin,
        edge_mode=edge_mode,
    )


propagate_gnuradio_iq = propagate_iq
