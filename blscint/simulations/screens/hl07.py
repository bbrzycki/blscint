"""Free-space diffraction transfer functions.

Reference
---------
Li, J., Peng, Z., & Fu, Y. 2007, "Diffraction transfer function and its
calculation of classic diffraction formula", Optics Communications, 280,
243-248. doi:10.1016/j.optcom.2007.08.053

This module is intentionally not an ISM model.  It is the numerical
propagation layer that the scattering-screen models rely on: Li et al. discuss
Fresnel, Kirchhoff, Rayleigh-Sommerfeld, and angular-spectrum transfer
functions and their FFT sampling constraints.  The screen modules currently use
the paraxial Fresnel transfer function; the exact angular-spectrum kernel is
provided here for validation and future higher-fidelity paths.
"""

import numpy as np
from astropy import units as u
import setigen as stg

from .base_classes import get_k


def _as_inverse_length(q_mag):
    if hasattr(q_mag, "unit"):
        return q_mag.to(1 / u.cm)
    return np.asarray(q_mag) / u.cm


def fresnel_transfer_function(q_mag, z, f):
    """Paraxial free-space transfer function in transverse wavenumber space.

    ``q_mag`` is the transverse angular wavenumber magnitude in rad / length.
    The returned kernel omits the global carrier phase, matching the convention
    used by the existing Coles-style screen propagation code.
    """
    q_mag = _as_inverse_length(q_mag)
    z = stg.cast_value(z, u.cm)
    k = get_k(f).to(1 / u.cm)
    phase = (q_mag**2 * z / (2 * k)).decompose().value
    return np.exp(-1j * phase)


def angular_spectrum_transfer_function(q_mag, z, f, include_global_phase=False):
    """Exact angular-spectrum free-space transfer function.

    When ``include_global_phase`` is false, the carrier phase ``exp(i k z)`` is
    divided out.  In the paraxial limit this reduces to
    ``fresnel_transfer_function``.
    """
    q_mag = _as_inverse_length(q_mag)
    z = stg.cast_value(z, u.cm)
    k = get_k(f).to(1 / u.cm)

    q_value = q_mag.to_value(1 / u.cm)
    k_value = k.to_value(1 / u.cm)
    kz = np.sqrt(k_value**2 - q_value**2 + 0j) / u.cm
    phase = kz * z
    if not include_global_phase:
        phase = phase - k * z
    return np.exp(1j * phase.decompose().value)
