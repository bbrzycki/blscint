"""Coles-style multi-screen scattering simulation.

Primary reference
-----------------
Coles, W. A., Rickett, B. J., Gao, J. J., Hobbs, G., & Verbiest, J. P. W.
2010, "Scattering of pulsar radio emission by the interstellar plasma",
The Astrophysical Journal, 717, 1206-1221. doi:10.1088/0004-637X/717/2/1206

The 2010 paper points back to Coles et al. 1995a for implementation details;
the current filename is kept for compatibility with the original scratch work.

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
import scipy.special
from astropy import units as u
import setigen as stg

from . import hl07
from .base_classes import (
    get_k,
    get_rF,
    make_observer_dynamic_spectrum,
    BaseRadioSource,
    BasePhaseSpectrum,
    BaseScreen,
    BaseScatteringModel,
)


def f_l0(x):
    a1 = 1.4284
    a2 = 1.1987 
    a3 = 0.1414
    return (1 + a1*x + a2*x**2 + a3*x**3) * np.exp(-x)


class RadioSource(BaseRadioSource):
    def __init__(self, distance):
        self.distance = distance

    def emit(self):
        return 1


class PhaseSpectrum(BasePhaseSpectrum):
    def __init__(self, 
                 m_b2=None,
                #  C_n2=None,
                 l0=0,
                 alpha=5/3,
                 distance=None,
                 dz=None):
        self.dz = dz 
        self.distance = distance

        self.alpha = alpha
        self.l0 = stg.cast_value(l0, u.cm)
        self.K1 = 2**alpha * scipy.special.gamma(1 + alpha / 2) * np.cos(alpha * np.pi / 4)
        self.A = scipy.special.gamma(1 + alpha) * np.sin((alpha - 1) * np.pi / 2) / (4 * np.pi**2)
        # if m_b2 is None:
        #     if C_n2 is None:
        #         raise ValueError("Only set one of m_b2 or C_n2")
        #     self.C_n2 = C_n2
        # else:
        #     if m_b2 is None:
        #         raise ValueError("Only set one of m_b2 or C_n2")
        self.m_b2 = m_b2

    def _T_factor(self, f):
        """
        Factor K so that T = K * C_n2.
        """
        return 2 * np.pi * get_k(f)**2 * self.A * self.dz

    def C_n2(self, f):
        return self.m_b2 / (4 * np.pi * self._T_factor(f) 
                            * scipy.special.gamma(1 - self.alpha / 2) 
                            * np.cos(self.alpha * np.pi / 4) 
                            * get_rF(self.distance, f)**self.alpha / self.alpha)

    def T(self, f):
        return self._T_factor(f) * self.C_n2(f)

    def s0(self, f):
        """
        Eq. 15 & 17.
        """
        return (self.m_b2 / self.K1)**(-1/self.alpha) * get_rF(self.distance, f)

    def D_TS(self, s, f):
        return (s / self.s0(f))**(self.alpha)

    def Phi(self, q, f):
        return self.T(f) * q**(-self.alpha - 2) * f_l0(q * self.l0)


class Screen(BaseScreen):
    def __init__(self, 
                 distance, 
                 dx, 
                 dy, 
                 dz, 
                 shape=(16, 16), 
                 alpha=5/3,
                 m_b2=None,
                #  C_n2=None,
                 l0=0,
                 seed=None):
        self.rng = np.random.default_rng(seed)
        self.distance = distance 
        self.dz = dz
        self.shape = shape
        self.Ny, self.Nx = self.shape 
        self.dx, self.dy = dx, dy
        self.Ly, self.Lx = self.Ny * self.dy, self.Nx * self.dx 

        # wavenumber components in x, y directions
        self.q_x, self.q_y = np.meshgrid((np.arange(self.Nx) - self.Nx//2) * 2 * np.pi / self.Lx, 
                                         (np.arange(self.Ny) - self.Ny//2) * 2 * np.pi / self.Ly)
        self.q_mag = (self.q_x**2 + self.q_y**2)**0.5

        # Set up phase spectrum
        self.spectrum = PhaseSpectrum(m_b2=m_b2,
                                    #   C_n2=C_n2,
                                      l0=l0,
                                      alpha=alpha,
                                      distance=distance,
                                      dz=dz)

        self.noise = self.rng.standard_normal(size=self.shape) + 1j * self.rng.standard_normal(size=self.shape)

    def phases(self, f):
        with np.errstate(divide="ignore", invalid="ignore"):
            var_pq = (self.spectrum.Phi(self.q_mag, f)
                    * 4 * np.pi**2 * self.Nx * self.Ny / (self.dx * self.dy))
        var_pq[self.Ny//2, self.Nx//2] = 0

        var_pq = np.fft.fftshift(var_pq)

        phi_pq = self.noise * var_pq**0.5
        
        phi_mn = np.fft.ifft2(phi_pq).real
        return phi_mn

    def propagate_phase_screen(self, E, f):
        return E * np.exp(1j * self.phases(f))

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
