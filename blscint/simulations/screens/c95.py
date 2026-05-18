"""Coles-style multi-screen scattering simulation.

Primary reference
-----------------
Coles, W. A., Rickett, B. J., Gao, J. J., Hobbs, G., & Verbiest, J. P. W.
2010, "Scattering of pulsar radio emission by the interstellar plasma",
The Astrophysical Journal, 717, 1206-1221. doi:10.1088/0004-637X/717/2/1206

The 2010 paper points back to Coles et al. 1995a for implementation details;
the current filename is kept for compatibility with the original scratch work.

Methodology role
----------------
This is the primary numerical validation path for the shared screen backend.
The code uses Coles-style scattering-strength parameters and structure-function
normalization, then delegates the actual random phase-screen realization to
``power_law.PowerLawPhaseScreen``.  Other paper-facing wrappers should match
this module only when they request the same phase spectrum and use the same
propagation conventions.

Equation map
------------
* Phase structure function ``D(s) = (s / s0)**alpha`` follows Coles et al.
  2010 Section 3.
* ``m_b2`` follows the Born scintillation-strength definition in Coles et al.
  2010 Section 3.
* ``s0`` is derived from the paper's relation between ``m_b2`` and
  ``D(r_F)``.
* Free-space propagation uses the FFT angular-spectrum/Fresnel approach
  described in Coles et al. 2010 Section 2 and implemented with the Li et al.
  2007 transfer-function helper in ``hl07``.
"""

import numpy as np 

from . import hl07
from .base_classes import (
    make_narrowband_tone_field,
    make_narrowband_tone_profile,
    make_observer_dynamic_spectrum,
    BaseRadioSource,
    BasePhaseSpectrum,
    BaseScreen,
    BaseScatteringModel,
)
from .power_law import (
    PowerLawPhaseScreen,
    ScatteringStrengthSpectrum,
    inner_scale_taper,
)


def f_l0(x):
    return inner_scale_taper(x)


class RadioSource(BaseRadioSource):
    def __init__(self, distance):
        self.distance = distance

    def emit(self):
        return 1


class PhaseSpectrum(ScatteringStrengthSpectrum, BasePhaseSpectrum):
    def __init__(self, 
                 m_b2=None,
                 s0=None,
                 l0=0,
                 alpha=5/3,
                 distance=None,
                 dz=None):
        super().__init__(
            m_b2=m_b2,
            s0=s0,
            l0=l0,
            alpha=alpha,
            distance=distance,
            dz=dz,
        )


class Screen(BaseScreen):
    def __init__(self, 
                 distance, 
                 dx, 
                 dy, 
                 dz, 
                 shape=(16, 16), 
                 alpha=5/3,
                 m_b2=None,
                 s0=None,
                 l0=0,
                 subharmonic_levels=0,
                 normalize_structure=True,
                 phase_sign=1,
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.distance = distance 
        self.dz = dz
        self.shape = shape
        self.phase_sign = phase_sign
        self.Ny, self.Nx = self.shape 
        self.dx, self.dy = dx, dy
        self.Ly, self.Lx = self.Ny * self.dy, self.Nx * self.dx 

        # wavenumber components in x, y directions
        self.q_x, self.q_y = np.meshgrid((np.arange(self.Nx) - self.Nx//2) * 2 * np.pi / self.Lx, 
                                         (np.arange(self.Ny) - self.Ny//2) * 2 * np.pi / self.Ly)
        self.q_mag = (self.q_x**2 + self.q_y**2)**0.5

        # Set up phase spectrum
        self.spectrum = PhaseSpectrum(m_b2=m_b2,
                                      s0=s0,
                                      l0=l0,
                                      alpha=alpha,
                                      distance=distance,
                                      dz=dz)

        self.normalize_structure = normalize_structure
        self.phase_screen = PowerLawPhaseScreen(
            shape=self.shape,
            dx=self.dx,
            dy=self.dy,
            seed=seed,
            subharmonic_levels=subharmonic_levels,
        )

    def phases(self, f):
        reference_scale = self.spectrum.s0(f) if self.normalize_structure else None
        return self.phase_screen.phases(
            self.spectrum,
            f,
            reference_scale=reference_scale,
            reference_dphi=1.0,
        )

    def propagate_phase_screen(self, E, f):
        return E * np.exp(1j * self.phase_sign * self.phases(f))

    def propagate_free_space(self, E, z, f):
        if isinstance(E, (int, float)):
            E = np.full(self.shape, E)
        E = np.fft.ifft2(np.fft.fft2(E) * hl07.fresnel_transfer_function(self.q_mag, z, f))
        return E
    

class ScatteringModel(BaseScatteringModel):
    """
    Basic class to simulate diffractive interstellar scintillation (DISS).
    """
    def __init__(self, 
                 source, 
                 screens, 
                 dx, 
                 dy, 
                 shape=(16, 16)):
        self.source = source
        self.screens = screens
        self.shape = shape
        self.Ny, self.Nx = self.shape 
        self.dx, self.dy = dx, dy
        self.Ly, self.Lx = self.Ny * self.dy, self.Nx * self.dx 

        # wavenumber components in x, y directions
        self.q_x, self.q_y = np.meshgrid((np.arange(self.Nx) - self.Nx//2) * 2 * np.pi / self.Lx, 
                                         (np.arange(self.Ny) - self.Ny//2) * 2 * np.pi / self.Ly)
        self.q_mag = (self.q_x**2 + self.q_y**2)**0.5

    def propagate_free_space(self, E, z, f):
        if isinstance(E, (int, float)):
            E = np.full(self.shape, E)
        E = np.fft.ifft2(np.fft.fft2(E) * hl07.fresnel_transfer_function(self.q_mag, z, f))
        return E
            
    def observer_electric_field(self, f):
        E = self.source.emit()
        for idx in range(len(self.screens)):
            if idx == 0:
                dz = self.source.distance - self.screens[0].distance
            else:
                dz = self.screens[idx].distance - self.screens[idx-1].distance
            E = self.screens[idx].propagate_free_space(E, dz, f)
            E = self.screens[idx].propagate_phase_screen(E, f)
        E = self.propagate_free_space(E, self.screens[-1].distance, f)
        return E

    def observer_dynamic_spectrum_result(self, fmin, df, fchans, v_trans=None,
                                         normalize=None, progress=True):
        return make_observer_dynamic_spectrum(
            self,
            fmin=fmin,
            df=df,
            fchans=fchans,
            sample_count=self.Nx,
            sample_spacing=self.dx,
            row_index=self.Ny // 2,
            v_trans=v_trans,
            normalize=normalize,
            progress=progress,
            metadata={
                "model": "c95",
                "axis_order": "(observer_plane_x, frequency)",
            },
        )

    def observer_dynamic_spectrum(self, fmin, df, fchans):
        return self.observer_dynamic_spectrum_result(fmin, df, fchans).intensity

    def observer_narrowband_tone_field(self, frequency, v_trans=None):
        return make_narrowband_tone_field(
            self,
            frequency=frequency,
            sample_count=self.Nx,
            sample_spacing=self.dx,
            row_index=self.Ny // 2,
            v_trans=v_trans,
            metadata={
                "model": "c95",
                "axis_order": "(observer_plane_x, tone_frequency)",
            },
        )

    def observer_narrowband_tone_profile(self, frequency, intrinsic_intensity=1.0,
                                         v_trans=None, normalize=None):
        return make_narrowband_tone_profile(
            self,
            frequency=frequency,
            sample_count=self.Nx,
            sample_spacing=self.dx,
            row_index=self.Ny // 2,
            intrinsic_intensity=intrinsic_intensity,
            v_trans=v_trans,
            normalize=normalize,
            metadata={
                "model": "c95",
                "axis_order": "(observer_plane_x, tone_frequency)",
            },
        )
