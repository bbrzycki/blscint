"""Shared power-law phase-screen generation.

Methodology
-----------
This module is the numerical screen engine used by the paper-specific wrappers.
It intentionally separates the physical phase spectrum from the paper
parameterization used to request that spectrum.

Coles-style code can specify a screen through ``m_b2`` or ``s0``.  Ravi and
Deshpande-style code can specify the same thin-screen spectrum through a
physical ``C_n2`` and screen thickness ``dz``.  Both routes are converted to
the intermediate 2-D phase-spectrum amplitude ``T`` before generating
``Phi_phi(q, f)``.

Once two wrappers provide the same grid, seed, phase sign, propagation ordering,
transfer-function kernel, and ``Phi_phi(q, f)``, they are expected to produce
the same phase screen and observer-plane intensity.  That equivalence is a
deliberate implementation check, not evidence that this module is an
independent reproduction of every discrete step in Ravi and Deshpande 2018.
"""

from dataclasses import dataclass

import numpy as np
import scipy.special
from astropy import units as u
from astropy.constants import a0, alpha as fine_structure_alpha
import setigen as stg

from .base_classes import get_k, get_rF, get_wavelength


r_e = a0 * fine_structure_alpha**2


def phase_from_electron_column(electron_column, f, phase_sign=-1):
    """Convert electron column density to cold-plasma phase.

    ``electron_column`` is :math:`N_e = \\int n_e dz`.  The default sign follows
    the usual cold-plasma convention for the carrier phase; wrappers may still
    use their own ``phase_sign`` convention when applying generated screens.
    """
    electron_column = stg.cast_value(electron_column, 1 / u.cm**2)
    return phase_sign * (r_e * get_wavelength(f) * electron_column).decompose().value


def electron_column_from_phase(phase, f, phase_sign=-1):
    """Convert cold-plasma phase to electron column density."""
    return phase / (phase_sign * r_e * get_wavelength(f))


def phase_from_electron_density(delta_n_e, dz, f, phase_sign=-1):
    """Convert a slab-averaged electron-density perturbation to phase."""
    electron_column = stg.cast_value(delta_n_e, 1 / u.cm**3) * stg.cast_value(dz, u.cm)
    return phase_from_electron_column(electron_column, f, phase_sign=phase_sign)


def electron_density_from_phase(phase, dz, f, phase_sign=-1):
    """Convert phase to a slab-averaged electron-density perturbation."""
    return electron_column_from_phase(phase, f, phase_sign=phase_sign) / stg.cast_value(dz, u.cm)


def scattering_measure_from_C_n2(C_n2, dz):
    """Return scattering measure ``SM = C_n2 * dz`` for a uniform screen."""
    return C_n2 * stg.cast_value(dz, u.kpc)


def C_n2_from_scattering_measure(scattering_measure, dz):
    """Return uniform-screen ``C_n2`` from scattering measure and thickness."""
    return scattering_measure / stg.cast_value(dz, u.kpc)


def density_variance_from_C_n2(C_n2, outer_scale, inner_scale=None, beta=11 / 3):
    """Estimate density-fluctuation variance from a 3-D power-law spectrum.

    This evaluates ``integral d^3q P_delta_ne(q) / (2*pi)^3`` for
    ``P_delta_ne(q) = C_n2 q^-beta`` between outer and inner cutoffs.  It is a
    statistical link to ``delta n_e``; it is not a unique conversion to mean
    electron density.
    """
    q_outer = 2 * np.pi / stg.cast_value(outer_scale, u.m)
    exponent = 3 - beta
    if inner_scale is None:
        if exponent >= 0:
            raise ValueError("inner_scale is required for beta <= 3")
        q_inner_term = 0 * q_outer**exponent
    else:
        q_inner = 2 * np.pi / stg.cast_value(inner_scale, u.m)
        q_inner_term = q_inner**exponent
    integral = (q_inner_term - q_outer**exponent) / exponent
    return (4 * np.pi / (2 * np.pi) ** 3 * C_n2 * integral).to(1 / u.m**6)


def density_rms_from_C_n2(C_n2, outer_scale, inner_scale=None, beta=11 / 3):
    """Estimate rms electron-density fluctuation from ``C_n2`` and cutoffs."""
    return np.sqrt(density_variance_from_C_n2(
        C_n2,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
        beta=beta,
    ))


def C_n2_from_density_rms(delta_n_e_rms, outer_scale, inner_scale=None, beta=11 / 3):
    """Infer ``C_n2`` from rms density fluctuation and spectral cutoffs."""
    unit_C_n2 = u.m ** (-(beta + 3))
    variance_per_unit_C_n2 = density_variance_from_C_n2(
        1 * unit_C_n2,
        outer_scale=outer_scale,
        inner_scale=inner_scale,
        beta=beta,
    )
    return (stg.cast_value(delta_n_e_rms, 1 / u.m**3) ** 2 / variance_per_unit_C_n2) * unit_C_n2


def fractional_density_rms_from_C_n2(C_n2, mean_n_e, outer_scale,
                                     inner_scale=None, beta=11 / 3):
    """Return ``delta_n_e,rms / mean_n_e`` for an assumed mean density."""
    return (
        density_rms_from_C_n2(C_n2, outer_scale, inner_scale=inner_scale, beta=beta)
        / stg.cast_value(mean_n_e, 1 / u.m**3)
    ).decompose().value


def scattering_strength_factor(f, distance, alpha=5 / 3):
    """Return the factor that maps phase-spectrum amplitude ``T`` to ``m_b2``."""
    return (
        4
        * np.pi
        * scipy.special.gamma(1 - alpha / 2)
        * np.cos(alpha * np.pi / 4)
        * get_rF(distance, f) ** alpha
        / alpha
    )


def phase_amplitude_from_m_b2(m_b2, f, distance, alpha=5 / 3):
    """Convert Coles-style ``m_b2`` to 2-D phase-spectrum amplitude ``T``."""
    return m_b2 / scattering_strength_factor(f, distance, alpha)


def m_b2_from_phase_amplitude(T, f, distance, alpha=5 / 3):
    """Convert 2-D phase-spectrum amplitude ``T`` to Coles-style ``m_b2``."""
    return (T * scattering_strength_factor(f, distance, alpha)).decompose().value


def phase_amplitude_from_C_n2(C_n2, dz, f):
    """Convert physical 3-D ``C_n2`` to 2-D phase-spectrum amplitude ``T``."""
    return 2 * np.pi * stg.cast_value(dz, u.cm) * (get_wavelength(f) * r_e) ** 2 * C_n2


def C_n2_from_phase_amplitude(T, dz, f):
    """Convert 2-D phase-spectrum amplitude ``T`` to physical 3-D ``C_n2``."""
    return T / (2 * np.pi * stg.cast_value(dz, u.cm) * (get_wavelength(f) * r_e) ** 2)


def C_n2_from_m_b2(m_b2, f, distance, dz, alpha=5 / 3):
    """Convert Coles-style ``m_b2`` to physical 3-D ``C_n2``."""
    T = phase_amplitude_from_m_b2(m_b2, f, distance, alpha)
    return C_n2_from_phase_amplitude(T, dz, f)


def m_b2_from_C_n2(C_n2, f, distance, dz, alpha=5 / 3):
    """Convert physical 3-D ``C_n2`` to Coles-style ``m_b2``."""
    T = phase_amplitude_from_C_n2(C_n2, dz, f)
    return m_b2_from_phase_amplitude(T, f, distance, alpha)


def inner_scale_taper(x):
    """Approximation to the inner-scale cutoff used by the Coles scratch code."""
    a1 = 1.4284
    a2 = 1.1987
    a3 = 0.1414
    return (1 + a1 * x + a2 * x**2 + a3 * x**3) * np.exp(-x)


class ScatteringStrengthSpectrum:
    """Power-law phase spectrum parameterized by Coles-style ``m_b2``."""

    def __init__(self, m_b2=None, s0=None, l0=0, alpha=5 / 3,
                 distance=None, dz=None):
        if m_b2 is None and s0 is None:
            raise ValueError("one of m_b2 or s0 must be provided")
        self.dz = dz
        self.distance = distance
        self.alpha = alpha
        self.l0 = stg.cast_value(l0, u.cm)
        self.K1 = (
            2**alpha
            * scipy.special.gamma(1 + alpha / 2)
            * np.cos(alpha * np.pi / 4)
        )
        self.A = (
            scipy.special.gamma(1 + alpha)
            * np.sin((alpha - 1) * np.pi / 2)
            / (4 * np.pi**2)
        )
        self.m_b2 = m_b2
        self._s0 = None if s0 is None else stg.cast_value(s0, u.cm)

    def T(self, f):
        return phase_amplitude_from_m_b2(self.m_b2_at(f), f, self.distance, self.alpha)

    def C_n2(self, f):
        """Return the equivalent physical 3-D turbulence strength."""
        return C_n2_from_phase_amplitude(self.T(f), self.dz, f)

    def physical_C_n2(self, f):
        """Alias for explicit callers."""
        return self.C_n2(f)

    def m_b2_at(self, f):
        if self.m_b2 is not None:
            return self.m_b2
        return self.K1 * self.D_TS(get_rF(self.distance, f), f).decompose().value

    def s0(self, f):
        if self._s0 is not None:
            return self._s0
        return (self.m_b2 / self.K1) ** (-1 / self.alpha) * get_rF(self.distance, f)

    def D_TS(self, s, f):
        return (stg.cast_value(s, u.cm) / self.s0(f)) ** self.alpha

    def Phi(self, q, f):
        return self.T(f) * q ** (-self.alpha - 2) * inner_scale_taper(q * self.l0)


class ThinScreenSpectrum:
    """Ravi & Deshpande Appendix B thin-screen phase spectrum."""

    def __init__(self, C_n2=None, alpha=5 / 3, distance=None, dz=None):
        if C_n2 is None:
            raise ValueError("C_n2 must be provided")
        self.dz = dz
        self.distance = distance
        self.alpha = alpha
        self.beta = alpha + 2
        self.C_n2 = C_n2

    def Phi(self, q, f):
        return self.T(f) * q ** (-self.beta)

    def T(self, f):
        return phase_amplitude_from_C_n2(self.C_n2, self.dz, f)

    def m_b2_at(self, f):
        return m_b2_from_phase_amplitude(self.T(f), f, self.distance, self.alpha)

    def s0(self, f):
        K1 = (
            2**self.alpha
            * scipy.special.gamma(1 + self.alpha / 2)
            * np.cos(self.alpha * np.pi / 4)
        )
        return (self.m_b2_at(f) / K1) ** (-1 / self.alpha) * get_rF(self.distance, f)

    def D_TS(self, s, f):
        return (stg.cast_value(s, u.cm) / self.s0(f)) ** self.alpha


@dataclass
class PowerLawPhaseScreen:
    """Real phase screen generated from a 2-D power-law spectrum."""

    shape: tuple
    dx: object
    dy: object
    seed: object = None
    subharmonic_levels: int = 0

    def __post_init__(self):
        self.Ny, self.Nx = self.shape
        self.dx = stg.cast_value(self.dx, u.cm)
        self.dy = stg.cast_value(self.dy, u.cm)
        self.Lx = self.Nx * self.dx
        self.Ly = self.Ny * self.dy
        self.rng = np.random.default_rng(self.seed)

        qx = np.fft.fftfreq(self.Nx, d=self.dx.to_value(u.cm)) * 2 * np.pi / u.cm
        qy = np.fft.fftfreq(self.Ny, d=self.dy.to_value(u.cm)) * 2 * np.pi / u.cm
        self.q_x, self.q_y = np.meshgrid(qx, qy)
        self.q_mag = (self.q_x**2 + self.q_y**2) ** 0.5

        self._white_fft = np.fft.fft2(self.rng.standard_normal(self.shape))
        self._x = (np.arange(self.Nx) - self.Nx // 2) * self.dx
        self._y = (np.arange(self.Ny) - self.Ny // 2) * self.dy
        self._xx, self._yy = np.meshgrid(self._x, self._y)
        self._subharmonic_modes = self._build_subharmonic_modes()

    def phases(self, spectrum, f, reference_scale=None, reference_dphi=1.0):
        """Generate a phase screen for a spectrum and optional D_phi target."""
        phase = self._base_phases(spectrum, f)
        if self.subharmonic_levels:
            phase = phase + self._subharmonic_phases(spectrum, f)
        if reference_scale is not None:
            measured = self._structure_at_scale(phase, reference_scale)
            if np.isfinite(measured) and measured > 0:
                phase = phase * np.sqrt(reference_dphi / measured)
        return phase

    def _base_phases(self, spectrum, f):
        with np.errstate(divide="ignore", invalid="ignore"):
            filter_power = spectrum.Phi(self.q_mag, f) * 4 * np.pi**2 / (self.dx * self.dy)
        filter_power = _quantity_value(filter_power)
        filter_power = np.nan_to_num(filter_power, nan=0.0, posinf=0.0, neginf=0.0)
        filter_power[0, 0] = 0
        phase_fft = self._white_fft * np.sqrt(np.maximum(filter_power, 0))
        return np.fft.ifft2(phase_fft).real

    def _subharmonic_phases(self, spectrum, f):
        phase = np.zeros(self.shape)
        for qx, qy, dq_area, theta in self._subharmonic_modes:
            q = (qx**2 + qy**2) ** 0.5
            amp2 = spectrum.Phi(q, f) * dq_area
            amp = float(np.sqrt(max(_quantity_value(amp2), 0)))
            arg = (qx * self._xx + qy * self._yy).decompose().value + theta
            phase += amp * np.cos(arg)
        return phase

    def _build_subharmonic_modes(self):
        modes = []
        qx0 = 2 * np.pi / self.Lx
        qy0 = 2 * np.pi / self.Ly
        for level in range(1, int(self.subharmonic_levels) + 1):
            qx_step = qx0 / (3**level)
            qy_step = qy0 / (3**level)
            dq_area = qx_step * qy_step
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    if ix == 0 and iy == 0:
                        continue
                    qx = ix * qx_step
                    qy = iy * qy_step
                    theta = self.rng.uniform(0, 2 * np.pi)
                    modes.append((qx, qy, dq_area, theta))
        return modes

    def _structure_at_scale(self, phase, scale):
        scale = stg.cast_value(scale, u.cm).to_value(u.cm)
        estimates = []
        for axis, spacing in ((1, self.dx.to_value(u.cm)), (0, self.dy.to_value(u.cm))):
            lag_float = scale / spacing
            max_lag = min(
                int(np.ceil(lag_float)) + 1,
                phase.shape[axis] // 2,
            )
            if max_lag < 1:
                continue
            lags = np.arange(1, max_lag + 1)
            values = []
            for lag in lags:
                if axis == 1:
                    delta = phase[:, lag:] - phase[:, :-lag]
                else:
                    delta = phase[lag:, :] - phase[:-lag, :]
                values.append(np.mean(delta**2))
            estimates.append(np.interp(lag_float, lags, values))
        if not estimates:
            return np.nan
        return float(np.mean(estimates))


def _quantity_value(value):
    if hasattr(value, "decompose"):
        return value.decompose().value
    return np.asarray(value)
