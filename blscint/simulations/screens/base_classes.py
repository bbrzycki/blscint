"""Shared interfaces and containers for screen-based propagation models.

The screen modules are intended to model propagation as a reusable physical
stage: a source emits a complex field, one or more media modify and propagate
that field, and the result can be sampled either as electric field or as an
intensity dynamic spectrum.  Higher-level setigen or receiver simulators can
then decide whether to apply the output in voltage space or as a cheaper
spectrogram-domain gain.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np 
from astropy import units as u
import setigen as stg
from tqdm import tqdm


def get_wavelength(f):
    return stg.cast_value(f, u.Hz).to(u.cm, equivalencies=u.spectral())


def get_k(f):
    return 2 * np.pi / get_wavelength(f)


def get_rF(distance, f):
    return (stg.cast_value(distance, u.cm) / get_k(f))**0.5


def get_frequency_axis(fmin, df, fchans):
    """Return the frequency axis used by a dynamic spectrum."""
    fchans = int(fchans)
    if fchans < 1:
        raise ValueError("fchans must be at least 1")
    return stg.cast_value(fmin, u.Hz) + np.arange(fchans) * stg.cast_value(df, u.Hz)


def get_spatial_axis(samples, spacing, center_index=None):
    """Return a centered one-dimensional spatial axis."""
    samples = int(samples)
    if samples < 1:
        raise ValueError("samples must be at least 1")
    if center_index is None:
        center_index = samples // 2
    return (np.arange(samples) - center_index) * stg.cast_value(spacing, u.cm)


def get_time_axis(spatial_axis, v_trans):
    """Convert a spatial cut through the observer plane to time."""
    return spatial_axis / stg.cast_value(v_trans, u.cm / u.s)


@dataclass(frozen=True)
class DynamicSpectrum:
    """Intensity dynamic spectrum sampled from a propagated electric field.

    The array convention is ``intensity[spatial_or_time, frequency]``.  The
    spatial axis is always present because screen propagation naturally produces
    a spatial observer-plane pattern.  A time axis can be attached by choosing
    an effective transverse velocity.
    """

    intensity: np.ndarray
    frequencies: u.Quantity
    spatial_axis: u.Quantity
    time_axis: Optional[u.Quantity] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self, mode="mean"):
        """Return a copy normalized as a gain field."""
        if mode in (None, False, "none"):
            return self
        if mode == "mean":
            scale = np.nanmean(self.intensity)
        elif mode == "median":
            scale = np.nanmedian(self.intensity)
        else:
            raise ValueError("normalization mode must be 'mean', 'median', or 'none'")
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("cannot normalize dynamic spectrum with zero or non-finite scale")
        return replace(self, intensity=self.intensity / scale)

    def with_time_axis(self, v_trans):
        """Return a copy with a time axis derived from transverse speed."""
        return replace(self, time_axis=get_time_axis(self.spatial_axis, v_trans))


def make_observer_dynamic_spectrum(model, fmin, df, fchans, sample_count,
                                   sample_spacing, row_index, v_trans=None,
                                   normalize=None, progress=True,
                                   metadata=None):
    """Sample ``model.observer_electric_field`` into a dynamic spectrum.

    This keeps the current screen modules focused on propagation while giving
    downstream code one common result object for voltage-domain and
    spectrogram-domain workflows.
    """
    frequencies = get_frequency_axis(fmin, df, fchans)
    complex_spectrum = np.zeros((int(sample_count), int(fchans)), dtype=np.complex128)
    iterator = tqdm(np.arange(int(fchans)), disable=not progress)
    for idx in iterator:
        complex_spectrum[:, idx] = model.observer_electric_field(frequencies[idx])[row_index]

    result = DynamicSpectrum(
        intensity=np.abs(complex_spectrum)**2,
        frequencies=frequencies,
        spatial_axis=get_spatial_axis(sample_count, sample_spacing),
        metadata=dict(metadata or {}),
    )
    if v_trans is not None:
        result = result.with_time_axis(v_trans)
    return result.normalized(normalize)


class BaseRadioSource(ABC):
    @abstractmethod
    def emit(self):
        return
    

class BasePhaseSpectrum(ABC):
    @abstractmethod
    def Phi(self, q, f):
        return


class BaseScreen(ABC):
    @abstractmethod
    def phases(self, f):
        return
    
    @abstractmethod
    def propagate_phase_screen(self, E, f):
        return
    
    @abstractmethod
    def propagate_free_space(self, E, z, f):
        return
    

class BaseScatteringModel(ABC):
    @abstractmethod
    def propagate_free_space(self, E, z, f):
        return
    
    @abstractmethod
    def observer_electric_field(self, f):
        return
    
    @abstractmethod
    def observer_dynamic_spectrum(self, fmin, df, fchans):
        return
    
