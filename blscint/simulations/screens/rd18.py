"""Ravi & Deshpande thin-screen DISS simulation.

Reference
---------
Ravi, K., & Deshpande, A. A. 2018, "Scintillation-based Search for
Off-pulse Radio Emission from Pulsars", The Astrophysical Journal, 859, 22.
doi:10.3847/1538-4357/aab60d

This module follows Appendix B of Ravi & Deshpande 2018:

* Eq. B1: 3-D power-law electron-density spectrum.
* Eq. B2: equivalent 2-D phase-screen spectrum.
* Eq. B4: FFT-based random phase-screen synthesis.
* Eq. B5-B8: phase-screen field and observer-plane intensity.

Although the paper is pulsar-motivated, the dynamic spectrum itself is the
piece we want for technosignature propagation: a frequency-dependent intensity
gain pattern that can be sampled spatially or converted to time with a
transverse velocity.
"""

import numpy as np 
from astropy import units as u
from astropy.constants import a0, alpha
r_e = a0 * alpha**2

from .base_classes import (
    get_wavelength,
    get_k,
    make_observer_dynamic_spectrum,
    BaseRadioSource,
    BasePhaseSpectrum,
    BaseScreen,
    BaseScatteringModel,
)


class RadioSource(BaseRadioSource):
    def __init__(self, distance):
        self.distance = distance

    def emit(self):
        return 1


class PhaseSpectrum(BasePhaseSpectrum):
    def __init__(self, 
                 C_n2=None,
                 alpha=5/3,
                 distance=None,
                 dz=None):
        self.dz = dz 
        self.distance = distance

        self.alpha = alpha
        self.beta = self.alpha + 2
        
        self.C_n2 = C_n2

    def Phi(self, q, f):
        return 2 * np.pi * self.dz * (get_wavelength(f) * r_e)**2 * self.C_n2 * q**(-self.beta)


class Screen(BaseScreen):
    def __init__(self, 
                 distance, 
                 N,
                 dr,
                 dz, 
                 alpha=5/3,
                 C_n2=None,
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.distance = distance 
        self.dz = dz
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

        # Set up phase spectrum
        self.spectrum = PhaseSpectrum(C_n2=C_n2,
                                      alpha=alpha,
                                      distance=distance,
                                      dz=dz)

        self.random_field_component = self.random_field_noise()

    def random_field_noise(self):
        """Return the bracketed FFT component in Ravi & Deshpande Eq. B4."""
        M = self.rng.standard_normal(self.shape) + 1j * self.rng.standard_normal(self.shape)
        g = np.fft.ifft2(M)
        
        with np.errstate(divide='ignore'):
            M0 = ((self.ii - self.Nc)**2 + (self.jj - self.Nc)**2)**(-self.spectrum.beta/4)
            M0[int(self.Nc), int(self.Nc)] = 0
        return np.fft.ifft2(g * M0)

    def phases(self, f):
        C0 = 2 * np.pi * (2 * np.pi)**(-self.spectrum.beta / 2)
        C1 = (self.N * self.dr)**(-1 + self.spectrum.beta / 2)
        C2 = (2 * np.pi * self.dz * (get_wavelength(f) * r_e)**2 * self.spectrum.C_n2)**0.5
        phi = C0 * C1 * C2 * self.random_field_component
        return phi.real

    def propagate_phase_screen(self, E, f):
        return E * np.exp(-1j * self.phases(f))

    def propagate_free_space(self, E, z, f):
        if isinstance(E, (int, float)):
            E = np.full(self.shape, E)
        
        k = get_k(f)
        xx, yy = self.dr * (self.jj - self.Nc, self.ii - self.Nc)
        # xx, yy = dr * (jj, ii)
        h = np.exp(1j*k*z)/(1j*get_wavelength(f)*z)*np.exp(1j*k/(2*z)*(xx**2+yy**2))

        E = np.fft.ifft2(np.fft.fft2(E) * np.fft.fft2(h))
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
        
        k = get_k(f)
        xx, yy = self.dr * (self.jj - self.Nc, self.ii - self.Nc)
        # xx, yy = dr * (jj, ii)
        h = np.exp(1j*k*z)/(1j*get_wavelength(f)*z)*np.exp(1j*k/(2*z)*(xx**2+yy**2))

        E = np.fft.ifft2(np.fft.fft2(E) * np.fft.fft2(h))
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
