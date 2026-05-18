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
* Coles et al. 2010 is the main reference for multi-screen ISM propagation,
  diffractive and refractive spatial scales, and pulsar-style dynamic spectra.
* Ravi and Deshpande 2018 gives a thin-screen DISS simulation recipe and a
  dynamic-spectrum correlation use case.

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

API sketch
----------

The current shared result object is
``blscint.simulations.screens.DynamicSpectrum``.  Existing model methods named
``observer_dynamic_spectrum`` still return a raw intensity array for notebook
compatibility.  New code should prefer ``observer_dynamic_spectrum_result``,
which returns axes, metadata, optional mean normalization, and an optional time
axis.
