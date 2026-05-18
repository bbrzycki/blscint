"""Ravi & Deshpande thin-screen DISS simulation.

Reference
---------
Ravi, K., & Deshpande, A. A. 2018, "Scintillation-based Search for
Off-pulse Radio Emission from Pulsars", The Astrophysical Journal, 859, 22.
doi:10.3847/1538-4357/aab60d

Methodology role
----------------
This module is an RD18-facing adapter over the shared screen backend.  It keeps
the Appendix B physical normalization route from ``C_n2`` to the 2-D phase
spectrum and preserves the RD18 default phase-sign convention.  It should not
be read as an independent line-by-line reproduction of the RD18 discrete
simulation.

Appendix B correspondence:

* Eq. B1: 3-D power-law electron-density spectrum.
* Eq. B2: equivalent 2-D phase-screen spectrum.
* Eq. B4: represented by the shared FFT phase-screen synthesis in
  ``power_law.PowerLawPhaseScreen``.
* Eq. B5-B8: represented by the shared phase-screen and Fresnel
  transfer-function propagation path used by the other wrappers.

Although the paper is pulsar-motivated, the dynamic spectrum itself is the
piece we want for technosignature propagation: a frequency-dependent intensity
gain pattern that can be sampled spatially or converted to time with a
transverse velocity.
"""

import numpy as np 
from astropy import units as u

from . import hl07
from .base_classes import (
    get_wavelength,
    get_k,
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
    ThinScreenSpectrum,
)


class RadioSource(BaseRadioSource):
    def __init__(self, distance):
        self.distance = distance

    def emit(self):
        return 1


class PhaseSpectrum(ThinScreenSpectrum, BasePhaseSpectrum):
    def __init__(self, 
                 C_n2=None,
                 alpha=5/3,
                 distance=None,
                 dz=None):
        super().__init__(C_n2=C_n2, alpha=alpha, distance=distance, dz=dz)


class Screen(BaseScreen):
    def __init__(self, 
                 distance, 
                 N,
                 dr,
                 dz, 
                 alpha=5/3,
                 C_n2=None,
                 m_b2=None,
                 s0=None,
                 l0=0,
                 subharmonic_levels=0,
                 normalize_structure=None,
                 phase_sign=-1,
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.distance = distance 
        self.dz = dz
        self.N = N
        self.Nc = self.N // 2
        self.dr = dr
        self.phase_sign = phase_sign
        self.shape = (N, N)
        self.Ny, self.Nx = self.shape 
        self.dx, self.dy = dr, dr
        self.Ly, self.Lx = self.Ny * self.dy, self.Nx * self.dx 

        # wavenumber components in x, y directions
        self.q_x, self.q_y = np.meshgrid((np.arange(self.Nx) - self.Nx//2) * 2 * np.pi / self.Lx, 
                                         (np.arange(self.Ny) - self.Ny//2) * 2 * np.pi / self.Ly)
        self.q_mag = (self.q_x**2 + self.q_y**2)**0.5

        self.jj, self.ii = np.meshgrid(np.arange(self.N), np.arange(self.N))

        if m_b2 is not None or s0 is not None:
            self.spectrum = ScatteringStrengthSpectrum(
                m_b2=m_b2,
                s0=s0,
                l0=l0,
                alpha=alpha,
                distance=distance,
                dz=dz,
            )
            self.normalize_structure = True if normalize_structure is None else normalize_structure
        else:
            self.spectrum = PhaseSpectrum(C_n2=C_n2,
                                          alpha=alpha,
                                          distance=distance,
                                          dz=dz)
            self.normalize_structure = False if normalize_structure is None else normalize_structure

        self.phase_screen = PowerLawPhaseScreen(
            shape=self.shape,
            dx=self.dr,
            dy=self.dr,
            seed=seed,
            subharmonic_levels=subharmonic_levels,
        )

    def random_field_noise(self):
        """Return a screen realization without frequency-dependent scaling."""
        return self.phase_screen._base_phases(self.spectrum, 1 * u.Hz)

    def phases(self, f):
        reference_scale = None
        if self.normalize_structure and hasattr(self.spectrum, "s0"):
            reference_scale = self.spectrum.s0(f)
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

    def propagate_free_space_impulse(self, E, z, f):
        if isinstance(E, (int, float)):
            E = np.full(self.shape, E)

        k = get_k(f)
        xx, yy = self.dr * (self.jj - self.Nc, self.ii - self.Nc)
        h = np.exp(1j*k*z)/(1j*get_wavelength(f)*z)*np.exp(1j*k/(2*z)*(xx**2+yy**2))

        E = np.fft.ifft2(np.fft.fft2(E) * np.fft.fft2(h)) * self.dr**2
        return E
    

class ScatteringModel(BaseScatteringModel):
    """
    Basic class to simulate diffractive interstellar scintillation (DISS).
    """
    def __init__(self, 
                 source, 
                 screens, 
                 N,
                 dr):
        self.source = source
        self.screens = screens
        self.N = N
        self.Nc = self.N // 2
        self.dr = dr
        self.shape = (N, N)
        self.Ny, self.Nx = self.shape 
        self.dx, self.dy = dr, dr
        self.Ly, self.Lx = self.Ny * self.dy, self.Nx * self.dx 

        # wavenumber components in x, y directions
        self.q_x, self.q_y = np.meshgrid((np.arange(self.Nx) - self.Nx//2) * 2 * np.pi / self.Lx, 
                                         (np.arange(self.Ny) - self.Ny//2) * 2 * np.pi / self.Ly)
        self.q_mag = (self.q_x**2 + self.q_y**2)**0.5
    
        self.jj, self.ii = np.meshgrid(np.arange(self.N), np.arange(self.N))

    def propagate_free_space(self, E, z, f):
        if isinstance(E, (int, float)):
            E = np.full(self.shape, E)

        E = np.fft.ifft2(np.fft.fft2(E) * hl07.fresnel_transfer_function(self.q_mag, z, f))
        return E

    def propagate_free_space_impulse(self, E, z, f):
        if isinstance(E, (int, float)):
            E = np.full(self.shape, E)

        k = get_k(f)
        xx, yy = self.dr * (self.jj - self.Nc, self.ii - self.Nc)
        h = np.exp(1j*k*z)/(1j*get_wavelength(f)*z)*np.exp(1j*k/(2*z)*(xx**2+yy**2))

        E = np.fft.ifft2(np.fft.fft2(E) * np.fft.fft2(h)) * self.dr**2
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
            sample_count=self.N,
            sample_spacing=self.dr,
            row_index=self.Nc,
            v_trans=v_trans,
            normalize=normalize,
            progress=progress,
            metadata={
                "model": "rd18",
                "axis_order": "(observer_plane_r, frequency)",
            },
        )

    def observer_dynamic_spectrum(self, fmin, df, fchans):
        return self.observer_dynamic_spectrum_result(fmin, df, fchans).intensity

    def observer_narrowband_tone_field(self, frequency, v_trans=None):
        return make_narrowband_tone_field(
            self,
            frequency=frequency,
            sample_count=self.N,
            sample_spacing=self.dr,
            row_index=self.Nc,
            v_trans=v_trans,
            metadata={
                "model": "rd18",
                "axis_order": "(observer_plane_r, tone_frequency)",
            },
        )

    def observer_narrowband_tone_profile(self, frequency, intrinsic_intensity=1.0,
                                         v_trans=None, normalize=None):
        return make_narrowband_tone_profile(
            self,
            frequency=frequency,
            sample_count=self.N,
            sample_spacing=self.dr,
            row_index=self.Nc,
            intrinsic_intensity=intrinsic_intensity,
            v_trans=v_trans,
            normalize=normalize,
            metadata={
                "model": "rd18",
                "axis_order": "(observer_plane_r, tone_frequency)",
            },
        )
