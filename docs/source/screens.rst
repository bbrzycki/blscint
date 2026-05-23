Screen Propagation
==================

The screen modules are the first step toward treating propagation as a modular
physical stage between a clean source signal and a receiver simulation.  The
near-term target is to produce observer-plane electric fields and dynamic
spectra that can be used either directly in voltage-space simulations or as
spectrogram-domain gain fields for setigen frames.

Reference roles
---------------

* Li, Peng, and Fu 2007 defines the FFT transfer-function machinery used for
  free-space diffraction checks.
* Cordes and Lazio 1991 is the narrowband SETI motivation: spectral broadening
  is usually small for Galactic lines of sight at GHz frequencies, but
  intensity scintillation materially changes detection probability and revisit
  strategy.
* Coles et al. 2010 is the main reference for multi-screen ISM propagation,
  diffractive and refractive spatial scales, and pulsar-style dynamic spectra.
* Ravi and Deshpande 2018 gives a thin-screen DISS simulation recipe and a
  dynamic-spectrum correlation use case.

Chosen methodology
------------------

The implementation strategy is not to maintain separate, competing numerical
screen engines for each paper.  Instead, ``power_law`` is the shared numerical
engine and the paper-specific modules are adapters that request a phase
spectrum using each paper's preferred variables.

This gives us one place to validate the numerics while preserving multiple
physical entry points:

* ``c95`` is the primary validation path for the shared engine.  It uses
  Coles-style ``m_b2``/``s0`` parameters, structure-function normalization, and
  transfer-function propagation.
* ``rd18`` is currently an RD18-facing adapter, not an independent
  line-by-line discrete reproduction of Ravi and Deshpande Appendix B.  It
  preserves the ``C_n2``/``dz`` normalization route into the same 2-D phase
  spectrum and keeps an RD18 default phase-sign convention.
* Exact C95/RD18 agreement is therefore an implementation consistency test:
  when the wrappers are given equivalent spectra and the same numerical
  conventions, they should match exactly.  It is not a claim that RD18's
  original discrete calculation has been independently replicated.

If we later need a stricter paper-replica test, the next step should be a
separate RD18 reference implementation that deliberately follows Appendix B's
discrete equations without sharing the production backend.  That reference can
then be compared against the shared engine.

Model boundaries
----------------

The current models are still ISM-focused.  They should not bake in pulsar-only
assumptions because the same propagation stage should eventually support
narrowband technosignatures, broadband signals, IPM/solar-wind screens, and
ionospheric or atmospheric effects.

Dynamic spectra use the array convention ``intensity[spatial_or_time,
frequency]``.  Screen propagation naturally creates a spatial observer-plane
pattern; a time axis is attached only after choosing an effective transverse
velocity.

Parameterizations
-----------------

The paper-specific modules now share the same power-law phase-screen backend in
``blscint.simulations.screens.power_law``.  The Coles-style path
parameterizes a screen with ``m_b2`` or ``s0``.  The Ravi and Deshpande-style
path can parameterize the same 2-D phase spectrum with a physical 3-D
``C_n2`` and screen thickness ``dz``.

For A/B implementation checks, the wrappers can be aligned by using the same
grid, random seed, propagation ordering, transfer-function kernel,
phase-sign convention, and equivalent phase-spectrum amplitude.  In that
configuration the two wrappers are expected to produce identical phase screens
and observer-plane intensities because they are exercising the same backend.

The conversion helpers ``C_n2_from_m_b2`` and ``m_b2_from_C_n2`` expose the
mapping through the intermediate 2-D phase-spectrum amplitude.  The
``ScatteringStrengthSpectrum.C_n2(f)`` method returns the equivalent physical
``C_n2`` for a Coles-style screen, and ``ThinScreenSpectrum.m_b2_at(f)``
returns the equivalent Coles-style ``m_b2`` for a physical thin screen.

API sketch
----------

The current shared result object is
``blscint.simulations.screens.DynamicSpectrum``.  Existing model methods named
``observer_dynamic_spectrum`` still return a raw intensity array for notebook
compatibility.  New code should prefer ``observer_dynamic_spectrum_result``,
which returns axes, metadata, optional mean normalization, and an optional time
axis.

For voltage-domain work, ``ElectricFieldSpectrum`` keeps the complex
observer-plane field before it is collapsed to intensity.  This preserves the
screen-induced phase offset and allows downstream receiver code to apply a
complex scalar envelope rather than only an intensity gain.

For intrinsically monochromatic signals, model methods named
``observer_narrowband_tone_field`` return the complex observer-plane field at
the tone frequency.  The field phase is the propagated carrier phase offset,
and ``abs(E)**2`` is the intensity gain for a unit-amplitude tone.
``observer_narrowband_tone_profile`` returns that same result collapsed to a
one-channel ``DynamicSpectrum``.  These helpers do not yet include finite
channel response, Doppler drift, or receiver noise.

Screen parameters currently include effective screen distance from the
observer, source distance through the model geometry, screen thickness ``dz``,
grid scale, spectral index ``alpha``, optional inner scale ``l0``, and
turbulence strength through either Coles-style ``m_b2``/``s0`` or physical
``C_n2``.  A mean electron density or absolute electron column is not currently
modeled; the physical path represents density fluctuations through ``C_n2``.

setigen interoperability
------------------------

The module ``blscint.simulations.screens.setigen_bridge`` provides the first
setigen-facing workflow layer.

For spectrogram-domain simulations:

* ``screen_gain_for_frame(model, frame, v_trans=...)`` samples a screen model
  onto a setigen ``Frame`` grid and returns a ``DynamicSpectrum`` whose shape
  matches ``frame.data``.
* ``scintillate_frame(frame, model=..., v_trans=...)`` multiplies an entire
  frame by that gain.  This is useful for clean synthetic signal frames, but
  it also scintillates any noise already in the frame.
* ``add_scintillated_signal(frame, clean_signal, model=..., v_trans=...)``
  applies the gain only to a clean signal array and then adds the result to
  the target frame.  This is the safer path for overlaying a propagated
  technosignature onto noise or a real observation.

For voltage-domain simulations:

* ``add_scintillated_voltage_signal(stream_or_antenna, model, f_start=...,
  v_trans=..., level=...)`` attaches a callable signal source to a setigen
  voltage ``DataStream``, ``Antenna``, or ``MultiAntennaArray``.
* The callable follows setigen's voltage convention by comparing ``f_start``
  to each stream's ``fch1``.  By default it returns a real voltage, with an
  ``analytic=True`` option for complex analytic samples.
* ``propagate_setigen_voltage(samples, model=..., sample_rate=..., fch1=...,
  v_trans=...)`` applies the broadband complex transfer to voltage arrays that
  were already generated by setigen.  It supports the usual sample-axis-last
  shapes, including ``(samples,)``, ``(pol, samples)``, and ``(antenna, pol,
  samples)``.
* ``get_propagated_setigen_voltage(source, num_samples, model=...,
  v_trans=...)`` asks a setigen voltage ``DataStream``, ``Antenna``, or
  ``MultiAntennaArray`` for samples and then propagates the resulting array.

The generic DSP layer lives in ``blscint.simulations.screens.voltage``.  Its
main entry point, ``propagate_iq(samples, transfer, sample_rate=...,
center_frequency=...)``, applies an ``ElectricFieldSpectrum`` to complex IQ
samples by FFT block propagation.  ``propagate_real_voltage`` first converts a
real voltage stream to its analytic representation, applies the same complex
operator, and can return either analytic or real samples.  GNU Radio-facing
code should call this pure NumPy/IQ layer rather than depending on setigen
objects.

The callable source-injection helper remains a narrowband approximation: it
samples the screen field at ``f_start`` and interpolates that complex scalar
envelope in time.  Broadband or heavily modulated voltage streams should use
``propagate_iq`` or ``propagate_setigen_voltage`` instead.

The current plasma screen is scalar and applies the same envelope to all
polarizations.  That is appropriate for the non-magnetoionic ISM scattering
model here.  Faraday rotation, birefringence, and other polarization-dependent
ionospheric/IPM effects should become separate propagation stages rather than
being hidden inside this scalar screen.

Electron-density links
----------------------

The screen phase is physically tied to electron column density.  For a cold
plasma phase convention,

``phi = - r_e * lambda * N_e``,

where ``N_e = integral n_e dz``.  The helpers
``phase_from_electron_column`` and ``electron_column_from_phase`` expose this
deterministic conversion.  If a screen is approximated as a uniform slab,
``phase_from_electron_density`` and ``electron_density_from_phase`` use ``dz``
to convert between column-density perturbation and slab-averaged
``delta_n_e``.

The turbulence parameter ``C_n2`` is different from mean ``n_e``.  It is the
amplitude of the 3-D electron-density fluctuation spectrum, and Cordes and
Lazio-style scattering measure is ``SM = integral C_n2 dz``.  The helpers
``scattering_measure_from_C_n2`` and ``C_n2_from_scattering_measure`` expose
the uniform-screen version of that relation.

To estimate an rms density fluctuation from ``C_n2``, the code needs assumed
inner and outer turbulence scales.  ``density_rms_from_C_n2`` integrates a
power-law fluctuation spectrum between those cutoffs.  A mean density enters
only if the caller wants a fractional fluctuation,
``delta_n_e,rms / mean_n_e``; there is no unique conversion from ``C_n2`` to
mean ``n_e`` without these extra assumptions.

Validation
----------

Reusable quantitative diagnostics live in
``blscint.simulations.screens.validation``.  The first validation notebooks are:

* ``jupyter-notebooks/screen_transfer_functions.ipynb``
* ``jupyter-notebooks/screen_dynamic_spectra.ipynb``
* ``jupyter-notebooks/screen_quantitative_validation.ipynb``
* ``jupyter-notebooks/screen_ab_validation.ipynb``
* ``jupyter-notebooks/screen_narrowband_tone.ipynb``
* ``jupyter-notebooks/screen_timescale_bridge.ipynb``

These notebooks are meant to separate four levels of confidence: numerical
kernel checks, qualitative screen behavior, quantitative paper-facing
diagnostics such as phase power-law slopes and intensity statistics, and A/B
implementation checks across paper-specific wrappers that request equivalent
phase spectra.  ``screen_timescale_bridge.ipynb`` is the current deeper
parameter study for converting physical screen outputs into calibrated
FFT/ARTA timescale inputs.  Other generated analysis notebooks from the current
screen-scintillation exploration are treated as local scratch until promoted.
