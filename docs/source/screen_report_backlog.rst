Screen Report Backlog
=====================

This page is a running list of screen-propagation and scintillation-analysis
items that look worth including in a later polished repo release, report, or
paper draft.  It is intentionally a working backlog, not final methodology.

Candidate figures
-----------------

These are generated analysis artifacts unless explicitly promoted into a
report/static-asset location.  They are useful waypoints for a later polished
writeup, but they should not all be committed by default.

* ``jupyter-notebooks/figures/acf_estimator_bias_analysis.png``:
  finite-sample ACF timescale bias.  This is a strong report figure because it
  separates estimator behavior from the FFT/ARTA generators themselves.
* ``jupyter-notebooks/figures/fast_synthesis_acf_benchmark.png``:
  FFT/ARTA ACF recovery and runtime comparison.  Useful for justifying when
  direct 1-D synthesis is appropriate.
* ``jupyter-notebooks/figures/gc_ne2001_timescale_comparison.png``:
  Galactic-center-style 600 s / 2.5 s narrowband tone comparison at 6 GHz and
  150 km/s.  Keep as an illustrative workflow until NE2001 is built and sampled
  directly.
* ``jupyter-notebooks/figures/screen_timescale_bridge.png``:
  first screen-parameter to effective-``t_d`` bridge figure.  This compares
  analytic ``s0 / v_eff`` estimates to row-averaged physical-screen ACF
  timescales and records a backed-out fast-synthesis handoff value.
* ``jupyter-notebooks/figures/screen_timescale_bridge_strength_diagnostics.png``:
  deeper bridge analysis over several screen strengths.  This is currently the
  clearest figure showing that ``s0 / v_eff`` is a useful coordinate but not a
  universal observed scintillation timescale.
* ``jupyter-notebooks/figures/screen_timescale_bridge_physical_variants.png``:
  fixed-``C_n2`` frequency and geometry variants.  Useful for documenting the
  distinction between code-facing ``m_b2`` sweeps and physically held
  turbulence amplitudes.
* ``jupyter-notebooks/figures/screen_timescale_bridge_fast_handoff.png``:
  backed-out physical-screen ``t_d`` passed into FFT/ARTA, showing that the
  fast generators still need finite-window calibration before their fitted
  timescales can be interpreted as observed screen timescales.
* ``jupyter-notebooks/figures/tone_method_comparison.png``:
  baseline side-by-side comparison of physical screen, setigen bridge, FFT,
  and ARTA for a fine channel-centered tone.
* ``jupyter-notebooks/figures/setigen_screen_workflows.png``:
  setigen interoperability overview for spectrogram and voltage-domain use.

Analysis conclusions to preserve
--------------------------------

* ACF-fit scintillation timescales from short observations are not unbiased
  point estimates.  For a 600 s observation sampled every 2.5 s with true
  ``t_d = 20 s``, the median fitted ``t_d`` is low even for an exact-target
  Gaussian process.  The fitted distribution is broad and skewed, so inference
  should use calibrated likelihoods rather than a universal correction factor.
* FFT synthesis is extremely fast and appears to recover the target ACF in the
  long-observation limit, but short observations need estimator calibration.
* The current ARTA implementation is fast but likely incomplete as a strict
  ARTA method: it uses the desired scintillation ACF as the latent Gaussian AR
  correlation before the exponential inverse-CDF transform.  A more rigorous
  implementation should solve the inverse correlation mapping for the
  transformed exponential process.
* The physical screen and setigen bridge are the same propagation calculation
  when both interpolate the complex electric field before squaring to intensity.
  The bridge is an interoperability layer for setigen ``Frame`` and voltage
  workflows, not a different scintillation model.

Calibration work before relying on fitted timescales
----------------------------------------------------

* Build estimator likelihood grids over true ``t_d``, observation duration,
  cadence, signal-to-noise ratio, noise/background normalization method, ACF
  estimator and fit window, and simulation route.
* Decide which estimator should be the default diagnostic for real detections:
  current sample-normalized ACF fit, e-fold crossing, likelihood rank over
  synthetic grids, or a Bayesian/forward-modeled inference layer.
* Record calibration products as versioned artifacts so that reported
  timescales are tied to the exact estimator and simulation assumptions used.

Physical screen and NE2001 work
-------------------------------

* Build or package the local NE2001 Fortran executable so
  ``query_ne2001`` can sample the Galactic-center sightline directly rather
  than using the current documented fallback.
* Replace the illustrative Galactic-center fallback with actual NE2001 rows
  for the target frequency and effective velocity, including the regime choice.
* Connect NE2001 scattering outputs to screen parameters more explicitly:
  ``SM``, ``C_n2``, screen thickness, screen placement, and the observed
  ``t_d``/decorrelation bandwidth.
* Keep distinguishing calibrated statistical synthesis from a physical
  screen-propagation realization.  They answer different questions and have
  very different runtime costs.

Parameter-to-timescale bridge
-----------------------------

* Build a bridge from physical screen inputs to fast-synthesis inputs.  The
  desired output is a calibrated ``t_d`` and decorrelation bandwidth that can
  feed FFT/ARTA for high-throughput signal injection, while preserving the full
  screen path for dedicated propagation experiments.
* Initial notebook: ``jupyter-notebooks/screen_timescale_bridge.ipynb``.
  This uses the physical screen itself as the reference by averaging ACFs over
  all horizontal observer-plane slices, which gives many scintle samples from
  one screen realization.
* Analytic first pass: compute the phase coherence scale ``s0`` from
  ``m_b2`` or ``C_n2``/``SM`` at the observing frequency, combine it with the
  Fresnel scale and thin-screen geometry to estimate the observer-plane
  diffractive scale, then use ``t_d = s_iss / v_eff``.
* Numerical calibration pass: sample a grid over frequency, screen distance,
  source distance, ``m_b2`` or ``C_n2``/``SM``, spectral index, inner scale,
  and effective transverse velocity; run physical screens for each point; fit
  the observer intensity ACF; store an interpolator from physical parameters to
  effective ``t_d``.
* Include uncertainty, not only a mean mapping.  Different screen realizations
  and short observing windows produce broad fitted-timescale distributions, so
  the bridge should return something like median, scatter, and estimator
  calibration metadata.
* Treat NE2001 as an upstream provider of scattering targets or turbulence
  amplitudes, not as a replacement for calibration.  NE2001 can supply
  ``SCINTIME``, ``SBW``, ``SM``, and related sightline quantities once the
  executable is built; the screen bridge should explain how those are converted
  into the simulation parameterization used here.
* Current empirical read from ``screen_timescale_bridge.ipynb``: for the
  representative ``6 GHz``, ``150 km/s``, ``m_b2=3000`` case, ``s0/v_eff`` is
  about ``46.4 s`` while the row-mean physical-screen e-fold is about
  ``26.7 s``.  At much stronger scattering and better domain coverage
  (``m_b2=30000``), these two scales nearly coincide.  The conversion is
  therefore regime- and domain-dependent.

Repo polish tasks
-----------------

* Move stable notebook-generation scripts into a more deliberate examples or
  validation location once the APIs settle.
* Add a compact public-facing example for applying a screen to a setigen
  ``Frame`` and another for applying a complex field transfer to voltage/IQ.
* Add regression tests for the physical screen versus setigen bridge
  equivalence at the center channel.
* Add a benchmark note with representative runtime numbers and hardware
  context.
* Decide which generated figures should be checked in, regenerated in CI, or
  left as notebook outputs only.
