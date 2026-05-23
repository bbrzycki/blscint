"""Build the screen-timescale bridge notebook.

The notebook is generated from this script so the long analysis cells can be
kept readable and regenerated without hand-editing JSON.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().with_name("screen_timescale_bridge.ipynb")


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
# Screen-Timescale Bridge Deep Dive

This notebook is meant to answer a practical question: when a physical phase-screen
simulation is too slow for bulk injection work, can we back out an effective
scintillation timescale that lets the faster FFT/ARTA synthesis paths stand in?

The key point tested here is not whether `s0 / v_eff` is dimensionally sensible.
It is whether the observer-plane intensity ACF actually behaves that way across
screen strengths, observing frequency, geometry, velocity, and finite grid size.
"""
    ),
    markdown(
        r"""
## Opinion Before Looking

The right workflow should probably be:

1. Use the physical screen as the calibration path.
2. Estimate the observer-plane intensity ACF over many horizontal slices, not one
   arbitrary cut.
3. Store both the analytic scales and the empirical `t_d` with validity metadata.
4. Feed the empirical `t_d` to FFT/ARTA only when we want cheap approximate
   injections.

The thing I do **not** trust yet is a universal scalar conversion from
`m_b2`, `C_n2`, or `s0` directly into an observed `t_d`. This notebook tests that
suspicion.
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from IPython.display import display


def find_project_dir():
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "blscint").is_dir() and (candidate / "jupyter-notebooks").is_dir():
            return candidate
    raise RuntimeError("Could not locate the blscint project root.")


PROJECT_DIR = find_project_dir()
NOTEBOOK_DIR = PROJECT_DIR / "jupyter-notebooks"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from blscint import diag_stats
from blscint.simulations.screens import c95, rd18, validation
from blscint.simulations.screens.base_classes import get_rF
from blscint.simulations.screens.power_law import scattering_measure_from_C_n2
from blscint.simulations.timeseries import get_ts_arta, get_ts_fft

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})
"""
    ),
    code(
        r"""
FREQUENCY = 6 * u.GHz
SOURCE_DISTANCE = 8.2 * u.kpc
SCREEN_DISTANCE = 4.0 * u.kpc
SCREEN_THICKNESS = 10 * u.pc
V_EFF = 150 * u.km / u.s
SHAPE = (512, 512)
DX = 1.7578125e7 * u.cm
SEED = 5
SUBHARMONIC_LEVELS = 1
ALPHA = 5 / 3


def dt_s(dx=DX, v_eff=V_EFF):
    return (dx / v_eff).to_value(u.s)


print(f"Project: {PROJECT_DIR}")
print(f"Grid: {SHAPE[0]} x {SHAPE[1]}, dx={DX.to_value(u.cm):.4e} cm")
print(f"Frequency baseline: {FREQUENCY.to_value(u.GHz):.2f} GHz")
print(f"Velocity baseline: {V_EFF.to_value(u.km / u.s):.1f} km/s")
print(f"Row time-equivalent sample spacing: {dt_s():.3f} s")
"""
    ),
    markdown(
        r"""
## Helper Functions

The important methodological choice is `row_acf_diagnostics`: every horizontal
observer-plane slice becomes one ACF estimate. That gives hundreds of samples
from a single physical screen realization and is much more defensible than
looking at one center row.
"""
    ),
    code(
        r"""
@dataclass(frozen=True)
class ScreenResult:
    label: str
    family: str
    frequency: u.Quantity
    v_eff: u.Quantity
    screen_distance: u.Quantity
    m_b2_request: float | None
    c_n2_request: u.Quantity | None
    intensity: np.ndarray
    diagnostics: dict
    scales: dict
    runtime_s: float


def c_n2_value(screen, frequency):
    candidate = getattr(screen.spectrum, "physical_C_n2", None)
    if callable(candidate):
        return candidate(frequency)
    candidate = getattr(screen.spectrum, "C_n2", None)
    if callable(candidate):
        return candidate(frequency)
    return candidate


def m_b2_value(screen, frequency):
    candidate = getattr(screen.spectrum, "m_b2_at", None)
    if callable(candidate):
        return float(candidate(frequency))
    return float(getattr(screen.spectrum, "m_b2"))


def analytic_scales(screen, frequency, v_eff, dx=DX, shape=SHAPE):
    r_f = get_rF(screen.distance, frequency).to(u.cm)
    s0 = screen.spectrum.s0(frequency).to(u.cm)
    c_n2 = c_n2_value(screen, frequency)
    sm = scattering_measure_from_C_n2(c_n2, screen.dz)
    refractive_scale = (r_f**2 / s0).to(u.cm)
    return {
        "r_f_cm": float(r_f.to_value(u.cm)),
        "s0_cm": float(s0.to_value(u.cm)),
        "c_n2_m20_over_3": float(c_n2.to_value(u.m ** (-20 / 3))),
        "sm_kpc_m20_over_3": float(sm.to_value(u.kpc * u.m ** (-20 / 3))),
        "m_b2_at_frequency": m_b2_value(screen, frequency),
        "refractive_scale_cm": float(refractive_scale.to_value(u.cm)),
        "t_s0_s": float((s0 / v_eff).to_value(u.s)),
        "t_fresnel_s": float((r_f / v_eff).to_value(u.s)),
        "t_refractive_s": float((refractive_scale / v_eff).to_value(u.s)),
        "domain_over_s0": float(shape[1] * dx.to_value(u.cm) / s0.to_value(u.cm)),
    }


def row_acf_diagnostics(intensity, dx=DX, v_eff=V_EFF):
    sample_dt = dt_s(dx, v_eff)
    row_acfs = []
    row_efold_px = []
    row_fit_px = []
    row_variance = []

    for row in intensity:
        normalized = row / np.mean(row)
        acf = diag_stats.autocorr(normalized)
        row_acfs.append(acf)
        row_variance.append(float(np.var(normalized)))

        lag = validation.decorrelation_lag(np.arange(len(acf)), acf)
        row_efold_px.append(float(lag) if np.isfinite(lag) else np.nan)

        try:
            fit = diag_stats.fit_acf(acf, pow=ALPHA, use_triangle=True)
            row_fit_px.append(float(fit[0]))
        except RuntimeError:
            row_fit_px.append(np.nan)

    row_acfs = np.asarray(row_acfs)
    mean_acf = np.nanmean(row_acfs, axis=0)
    p16_acf = np.nanpercentile(row_acfs, 16, axis=0)
    p84_acf = np.nanpercentile(row_acfs, 84, axis=0)

    mean_lag_px = validation.decorrelation_lag(np.arange(len(mean_acf)), mean_acf)
    if np.isfinite(mean_lag_px):
        mean_efold_s = float(mean_lag_px * sample_dt)
    else:
        mean_efold_s = np.nan

    try:
        mean_fit = diag_stats.fit_acf(mean_acf, pow=ALPHA, use_triangle=True)
        mean_fit_s = float(mean_fit[0] * sample_dt)
    except RuntimeError:
        mean_fit_s = np.nan

    normalized_intensity = intensity / np.mean(intensity)
    exp_diag = validation.exponential_gain_diagnostic(normalized_intensity)

    return {
        "row_acfs": row_acfs,
        "mean_acf": mean_acf,
        "p16_acf": p16_acf,
        "p84_acf": p84_acf,
        "row_efold_px": np.asarray(row_efold_px, dtype=float),
        "row_fit_px": np.asarray(row_fit_px, dtype=float),
        "row_variance": np.asarray(row_variance, dtype=float),
        "mean_efold_px": float(mean_lag_px) if np.isfinite(mean_lag_px) else np.nan,
        "mean_efold_s": mean_efold_s,
        "mean_fit_s": mean_fit_s,
        "intensity_exponential": exp_diag,
    }


def analyze_c95_case(
    m_b2,
    *,
    label=None,
    frequency=FREQUENCY,
    v_eff=V_EFF,
    source_distance=SOURCE_DISTANCE,
    screen_distance=SCREEN_DISTANCE,
    screen_thickness=SCREEN_THICKNESS,
    shape=SHAPE,
    dx=DX,
    seed=SEED,
):
    source = c95.RadioSource(source_distance)
    screen = c95.Screen(
        distance=screen_distance,
        dx=dx,
        dy=dx,
        dz=screen_thickness,
        shape=shape,
        m_b2=m_b2,
        seed=seed,
        subharmonic_levels=SUBHARMONIC_LEVELS,
    )
    model = c95.ScatteringModel(source, [screen], dx=dx, dy=dx, shape=shape)
    start = time.perf_counter()
    field = model.observer_electric_field(frequency)
    runtime_s = time.perf_counter() - start
    intensity = np.abs(field) ** 2
    diagnostics = row_acf_diagnostics(intensity, dx=dx, v_eff=v_eff)
    scales = analytic_scales(screen, frequency, v_eff, dx=dx, shape=shape)
    diagnostics["eta_efold"] = diagnostics["mean_efold_s"] / scales["t_s0_s"]
    diagnostics["eta_fit"] = diagnostics["mean_fit_s"] / scales["t_s0_s"]
    return ScreenResult(
        label=label or f"c95 m_b2={m_b2:g}",
        family="c95/m_b2",
        frequency=frequency,
        v_eff=v_eff,
        screen_distance=screen_distance,
        m_b2_request=float(m_b2),
        c_n2_request=None,
        intensity=intensity / np.mean(intensity),
        diagnostics=diagnostics,
        scales=scales,
        runtime_s=runtime_s,
    )


def analyze_rd18_case(
    c_n2,
    *,
    label,
    frequency=FREQUENCY,
    v_eff=V_EFF,
    source_distance=SOURCE_DISTANCE,
    screen_distance=SCREEN_DISTANCE,
    screen_thickness=SCREEN_THICKNESS,
    shape=SHAPE,
    dx=DX,
    seed=SEED,
):
    source = rd18.RadioSource(source_distance)
    screen = rd18.Screen(
        distance=screen_distance,
        N=shape[0],
        dr=dx,
        dz=screen_thickness,
        C_n2=c_n2,
        seed=seed,
        subharmonic_levels=SUBHARMONIC_LEVELS,
    )
    model = rd18.ScatteringModel(source, [screen], N=shape[0], dr=dx)
    start = time.perf_counter()
    field = model.observer_electric_field(frequency)
    runtime_s = time.perf_counter() - start
    intensity = np.abs(field) ** 2
    diagnostics = row_acf_diagnostics(intensity, dx=dx, v_eff=v_eff)
    scales = analytic_scales(screen, frequency, v_eff, dx=dx, shape=shape)
    diagnostics["eta_efold"] = diagnostics["mean_efold_s"] / scales["t_s0_s"]
    diagnostics["eta_fit"] = diagnostics["mean_fit_s"] / scales["t_s0_s"]
    return ScreenResult(
        label=label,
        family="rd18/C_n2",
        frequency=frequency,
        v_eff=v_eff,
        screen_distance=screen_distance,
        m_b2_request=None,
        c_n2_request=c_n2,
        intensity=intensity / np.mean(intensity),
        diagnostics=diagnostics,
        scales=scales,
        runtime_s=runtime_s,
    )


def summary_row(result):
    d = result.diagnostics
    s = result.scales
    return {
        "label": result.label,
        "family": result.family,
        "nu_GHz": result.frequency.to_value(u.GHz),
        "D_screen_kpc": result.screen_distance.to_value(u.kpc),
        "m_b2": s["m_b2_at_frequency"],
        "C_n2_m^-20/3": s["c_n2_m20_over_3"],
        "SM_kpc_m^-20/3": s["sm_kpc_m20_over_3"],
        "s0_cm": s["s0_cm"],
        "domain/s0": s["domain_over_s0"],
        "s0/v_s": s["t_s0_s"],
        "screen_efold_s": d["mean_efold_s"],
        "screen_fit_s": d["mean_fit_s"],
        "row_median_s": np.nanmedian(d["row_efold_px"]) * dt_s(DX, result.v_eff),
        "row_p16_s": np.nanpercentile(d["row_efold_px"], 16) * dt_s(DX, result.v_eff),
        "row_p84_s": np.nanpercentile(d["row_efold_px"], 84) * dt_s(DX, result.v_eff),
        "eta": d["eta_efold"],
        "intensity_var": d["intensity_exponential"]["variance"],
        "runtime_s": result.runtime_s,
    }


def summary_table(results):
    return pd.DataFrame([summary_row(result) for result in results])


def save_table_json(path, tables):
    payload = {}
    for name, table in tables.items():
        payload[name] = json.loads(table.to_json(orient="records"))
    Path(path).write_text(json.dumps(payload, indent=2))
"""
    ),
    markdown(
        r"""
## 1. Strength Sweep: Does `m_b2` Map Cleanly to `t_d`?

This sweep keeps the GC-ish geometry, `6 GHz`, and `150 km/s` fixed, then varies
the Coles-style scattering strength. The table reports the analytic
`s0 / v_eff`, the empirical row-mean ACF e-folding time, the fit-based ACF
timescale, and `eta = t_empirical / (s0 / v_eff)`.
"""
    ),
    code(
        r"""
M_B2_VALUES = [300, 1000, 3000, 10000, 30000]
strength_results = [analyze_c95_case(m_b2) for m_b2 in M_B2_VALUES]
strength = summary_table(strength_results)

cols = [
    "label",
    "m_b2",
    "C_n2_m^-20/3",
    "domain/s0",
    "s0/v_s",
    "screen_efold_s",
    "screen_fit_s",
    "row_p16_s",
    "row_p84_s",
    "eta",
    "intensity_var",
    "runtime_s",
]
display(strength[cols].style.format({
    "m_b2": "{:.0f}",
    "C_n2_m^-20/3": "{:.3e}",
    "domain/s0": "{:.2f}",
    "s0/v_s": "{:.2f}",
    "screen_efold_s": "{:.2f}",
    "screen_fit_s": "{:.2f}",
    "row_p16_s": "{:.2f}",
    "row_p84_s": "{:.2f}",
    "eta": "{:.3f}",
    "intensity_var": "{:.3f}",
    "runtime_s": "{:.3f}",
}))
"""
    ),
    code(
        r"""
def plot_strength_diagnostics(results):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    selected = [results[1], results[2], results[-1]]
    extent = [0, SHAPE[1] * dt_s(), SHAPE[0], 0]

    for ax, result in zip(axes[0], selected):
        image = result.intensity
        im = ax.imshow(
            image,
            aspect="auto",
            extent=extent,
            vmin=0,
            vmax=np.nanpercentile(image, 99),
        )
        ax.set_title(f"{result.label}\nvar={result.diagnostics['intensity_exponential']['variance']:.2f}")
        ax.set_xlabel("Horizontal cut time equivalent [s]")
        ax.set_ylabel("slice")
        fig.colorbar(im, ax=ax, label="unit-mean I")

    lags_s = np.arange(SHAPE[1]) * dt_s()
    keep = lags_s <= 170
    for ax, result in zip(axes[1], selected):
        d = result.diagnostics
        ax.plot(lags_s[keep], d["mean_acf"][keep], lw=2, label="row-mean ACF")
        ax.fill_between(
            lags_s[keep],
            d["p16_acf"][keep],
            d["p84_acf"][keep],
            alpha=0.18,
            linewidth=0,
            label="row 16-84%",
        )
        ax.plot(
            lags_s[keep],
            diag_stats.scint_acf(lags_s[keep], d["mean_efold_s"], pow=ALPHA),
            "k--",
            label="Kolmogorov at empirical e-fold",
        )
        ax.axhline(np.exp(-1), color="0.5", ls=":")
        ax.axvline(d["mean_efold_s"], color="tab:blue", ls=":")
        ax.axvline(result.scales["t_s0_s"], color="tab:orange", ls=":")
        ax.set_title(f"{result.label}: {d['mean_efold_s']:.1f}s vs analytic {result.scales['t_s0_s']:.1f}s")
        ax.set_xlabel("lag [s]")
        ax.set_ylabel("ACF")
        ax.legend(frameon=False, fontsize=8)

    m = strength["m_b2"].to_numpy()
    axes[2, 0].loglog(m, strength["s0/v_s"], "o-", label="analytic s0/v")
    axes[2, 0].loglog(m, strength["screen_efold_s"], "s-", label="screen e-fold")
    axes[2, 0].loglog(m, strength["screen_fit_s"], "d-", label="screen fit")
    axes[2, 0].invert_xaxis()
    axes[2, 0].set_xlabel("m_b2")
    axes[2, 0].set_ylabel("timescale [s]")
    axes[2, 0].set_title("Timescale vs strength")
    axes[2, 0].legend(frameon=False)

    axes[2, 1].semilogx(m, strength["eta"], "o-")
    axes[2, 1].invert_xaxis()
    axes[2, 1].set_xlabel("m_b2")
    axes[2, 1].set_ylabel("eta = empirical / analytic")
    axes[2, 1].set_ylim(0, 1.1)
    axes[2, 1].set_title("Eta depends on regime")
    for _, row in strength.iterrows():
        axes[2, 1].annotate(
            f"{row['domain/s0']:.1f} s0",
            (row["m_b2"], row["eta"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    axes[2, 2].semilogx(strength["domain/s0"], strength["eta"], "o-")
    axes[2, 2].axvspan(0, 10, color="tab:red", alpha=0.08, label="under-sampled caution")
    axes[2, 2].set_xlabel("screen width / s0")
    axes[2, 2].set_ylabel("eta")
    axes[2, 2].set_title("Domain coverage matters")
    axes[2, 2].legend(frameon=False)

    path = FIGURE_DIR / "screen_timescale_bridge_strength_diagnostics.png"
    fig.savefig(path, dpi=180)
    return fig, path


fig_strength, strength_path = plot_strength_diagnostics(strength_results)
strength_path
"""
    ),
    markdown(
        r"""
![strength diagnostics](figures/screen_timescale_bridge_strength_diagnostics.png)

Current run, high-signal numbers:

- `m_b2=300`: `s0/v = 184.7 s`, screen e-fold `72.6 s`, `eta = 0.393`,
  domain coverage `3.25 s0`. I do not trust this as a clean calibration point;
  the screen is too small in units of the coherence scale.
- `m_b2=3000`: `s0/v = 46.4 s`, screen e-fold `26.7 s`, `eta = 0.576`,
  domain coverage `12.93 s0`. This is the representative GC-ish point in this
  notebook.
- `m_b2=30000`: `s0/v = 11.65 s`, screen e-fold `11.15 s`, `eta = 0.957`,
  domain coverage `51.48 s0`. In this stronger/better-sampled regime, the
  analytic scale and intensity ACF scale nearly agree.
"""
    ),
    markdown(
        r"""
### Strength-Sweep Read

This is the first genuinely useful result: the analytic scale is not wrong, but
it is not the answer by itself. The empirical ratio `eta` changes across the
sweep. Weak screens are also poorly covered by this finite grid, so their
screen-wide ACF is less trustworthy. Stronger cases have better domain coverage
and move closer to `eta ~ 1`.

My current opinion: expose `s0 / v_eff` as a diagnostic, but do not use it as the
fast-injection `t_d` unless it has been calibrated for the screen regime and
grid/domain coverage.
"""
    ),
    markdown(
        r"""
## 2. Fixed Physical Turbulence: Frequency and Geometry Variants

The previous sweep holds `m_b2` fixed at each frequency, which is a code-facing
parameterization. For a physical propagation story we also need a fixed
`C_n2` case. Here I take the equivalent `C_n2` from the representative
`m_b2=3000`, `6 GHz` screen and feed that through the RD18-facing wrapper.

That gives two useful tests:

- Frequency sweep: same screen turbulence, different observing frequency.
- Geometry sweep: same turbulence and frequency, different screen distance.

The geometry sweep should be read with a caveat: this current implementation's
analytic `r_F`/`m_b2` bookkeeping uses the screen-observer distance in the same
way as the current C95/RD18 wrappers. A more formal finite-source thin-screen
analysis may want an effective Fresnel distance.
"""
    ),
    code(
        r"""
representative = strength_results[M_B2_VALUES.index(3000)]
base_c_n2 = representative.scales["c_n2_m20_over_3"] * u.m ** (-20 / 3)

frequency_results = [
    analyze_rd18_case(base_c_n2, label=f"fixed C_n2, {freq.to_value(u.GHz):.0f} GHz", frequency=freq)
    for freq in [4 * u.GHz, 6 * u.GHz, 8 * u.GHz]
]

geometry_results = [
    analyze_rd18_case(
        base_c_n2,
        label=f"fixed C_n2, D_screen={dist.to_value(u.kpc):.1f} kpc",
        frequency=FREQUENCY,
        screen_distance=dist,
    )
    for dist in [2.0 * u.kpc, 4.0 * u.kpc, 6.0 * u.kpc]
]

frequency_table = summary_table(frequency_results)
geometry_table = summary_table(geometry_results)

display(frequency_table[[
    "label", "nu_GHz", "m_b2", "domain/s0", "s0/v_s", "screen_efold_s", "eta", "intensity_var"
]].style.format({
    "nu_GHz": "{:.1f}",
    "m_b2": "{:.1f}",
    "domain/s0": "{:.2f}",
    "s0/v_s": "{:.2f}",
    "screen_efold_s": "{:.2f}",
    "eta": "{:.3f}",
    "intensity_var": "{:.3f}",
}))

display(geometry_table[[
    "label", "D_screen_kpc", "m_b2", "domain/s0", "s0/v_s", "screen_efold_s", "eta", "intensity_var"
]].style.format({
    "D_screen_kpc": "{:.1f}",
    "m_b2": "{:.1f}",
    "domain/s0": "{:.2f}",
    "s0/v_s": "{:.2f}",
    "screen_efold_s": "{:.2f}",
    "eta": "{:.3f}",
    "intensity_var": "{:.3f}",
}))
"""
    ),
    code(
        r"""
def plot_physical_variants(frequency_table, geometry_table, frequency_results, geometry_results):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    axes[0, 0].plot(frequency_table["nu_GHz"], frequency_table["s0/v_s"], "o-", label="s0/v")
    axes[0, 0].plot(frequency_table["nu_GHz"], frequency_table["screen_efold_s"], "s-", label="screen e-fold")
    axes[0, 0].set_xlabel("frequency [GHz]")
    axes[0, 0].set_ylabel("timescale [s]")
    axes[0, 0].set_title("Fixed C_n2 frequency sweep")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(frequency_table["nu_GHz"], frequency_table["m_b2"], "o-", color="tab:purple")
    axes[0, 1].set_xlabel("frequency [GHz]")
    axes[0, 1].set_ylabel("equivalent m_b2")
    axes[0, 1].set_title("Same C_n2 is stronger at lower frequency")

    axes[0, 2].plot(frequency_table["domain/s0"], frequency_table["eta"], "o-")
    axes[0, 2].set_xlabel("domain/s0")
    axes[0, 2].set_ylabel("eta")
    axes[0, 2].set_title("Frequency changes both strength and sampling")

    axes[1, 0].plot(geometry_table["D_screen_kpc"], geometry_table["s0/v_s"], "o-", label="s0/v")
    axes[1, 0].plot(geometry_table["D_screen_kpc"], geometry_table["screen_efold_s"], "s-", label="screen e-fold")
    axes[1, 0].set_xlabel("screen distance from observer [kpc]")
    axes[1, 0].set_ylabel("timescale [s]")
    axes[1, 0].set_title("Fixed C_n2 geometry sweep")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(geometry_table["D_screen_kpc"], geometry_table["m_b2"], "o-", color="tab:purple")
    axes[1, 1].set_xlabel("screen distance from observer [kpc]")
    axes[1, 1].set_ylabel("equivalent m_b2")
    axes[1, 1].set_title("Current wrapper strength vs screen distance")

    lags_s = np.arange(SHAPE[1]) * dt_s()
    keep = lags_s <= 140
    for result in frequency_results:
        axes[1, 2].plot(
            lags_s[keep],
            result.diagnostics["mean_acf"][keep],
            label=f"{result.frequency.to_value(u.GHz):.0f} GHz",
        )
    axes[1, 2].axhline(np.exp(-1), color="0.5", ls=":")
    axes[1, 2].set_xlabel("lag at 150 km/s [s]")
    axes[1, 2].set_ylabel("row-mean ACF")
    axes[1, 2].set_title("Frequency ACF shapes")
    axes[1, 2].legend(frameon=False)

    path = FIGURE_DIR / "screen_timescale_bridge_physical_variants.png"
    fig.savefig(path, dpi=180)
    return fig, path


fig_variants, variants_path = plot_physical_variants(
    frequency_table,
    geometry_table,
    frequency_results,
    geometry_results,
)
variants_path
"""
    ),
    markdown(
        r"""
![physical variants](figures/screen_timescale_bridge_physical_variants.png)

Current fixed-`C_n2` read:

- Frequency sweep at fixed turbulence is physically more useful than fixed
  `m_b2`: `4 GHz` maps to equivalent `m_b2 ~ 9463` and `t_d ~ 26.1 s`, while
  `8 GHz` maps to `m_b2 ~ 1328` and `t_d ~ 41.7 s`.
- The frequency trend is qualitatively right: lower frequency is effectively
  stronger scattering and faster scintillation in this setup.
- Geometry is not a trivial scalar correction in the current wrapper: with the
  same `C_n2`, `6 GHz`, and `150 km/s`, the measured e-folds are about
  `36.2 s`, `37.3 s`, and `28.2 s` for screen distances of `2`, `4`, and
  `6 kpc`. I would not build a public analytic conversion around this until the
  effective-distance convention is made explicit.
"""
    ),
    markdown(
        r"""
### Physical-Variant Read

Holding `C_n2` fixed is the more physical bridge than holding `m_b2` fixed. In
that mode, frequency changes the effective scattering strength substantially.
This is good: it gives us a path to a physical API where the caller specifies
screen turbulence once, then the code derives the frequency-dependent
scintillation behavior.

The geometry behavior is useful but should not be over-interpreted until we
decide whether the analytic scale should use the current wrapper's screen
distance convention or a finite-source effective distance. I would keep the
current numerical result, but label the analytic comparison as implementation
specific.
"""
    ),
    markdown(
        r"""
## 3. Velocity Is the Clean Part

Once the observer-plane spatial pattern exists, changing `v_eff` is just a
conversion from spatial lag to time lag. This should scale nearly exactly as
`1 / v_eff`. If it does not, the bug is likely in the bookkeeping rather than
in the screen propagation.
"""
    ),
    code(
        r"""
rep = representative
velocity_rows = []
for velocity in [75, 100, 150, 250, 300] * (u.km / u.s):
    scale = dt_s(DX, velocity)
    velocity_rows.append({
        "v_eff_km_s": velocity.to_value(u.km / u.s),
        "empirical_t_d_s": rep.diagnostics["mean_efold_px"] * scale,
        "fit_t_d_s": np.nanmedian(rep.diagnostics["row_fit_px"]) * scale,
        "s0_over_v_s": rep.scales["s0_cm"] / (velocity.to_value(u.cm / u.s)),
    })
velocity_table = pd.DataFrame(velocity_rows)
display(velocity_table.style.format({
    "v_eff_km_s": "{:.0f}",
    "empirical_t_d_s": "{:.2f}",
    "fit_t_d_s": "{:.2f}",
    "s0_over_v_s": "{:.2f}",
}))

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(velocity_table["v_eff_km_s"], velocity_table["empirical_t_d_s"], "o-", label="screen empirical")
ax.plot(velocity_table["v_eff_km_s"], velocity_table["s0_over_v_s"], "s--", label="s0/v")
ax.set_xlabel("v_eff [km/s]")
ax.set_ylabel("timescale [s]")
ax.set_title("Velocity scaling is a clean inverse rescaling")
ax.legend(frameon=False)
velocity_path = FIGURE_DIR / "screen_timescale_bridge_velocity_scaling.png"
fig.savefig(velocity_path, dpi=180)
velocity_path
"""
    ),
    markdown(
        r"""
![velocity scaling](figures/screen_timescale_bridge_velocity_scaling.png)

Velocity is the clean axis: the representative screen's empirical e-fold moves
from `53.4 s` at `75 km/s` to `26.7 s` at `150 km/s` and `13.4 s` at
`300 km/s`, matching the expected inverse scaling.
"""
    ),
    markdown(
        r"""
## 4. Backed-Out `t_d` Into FFT/ARTA

For fast synthesis, the thing to pass to FFT/ARTA should be the empirical screen
timescale, not the naive analytic one. Here I use the representative
`m_b2=3000` screen's row-mean e-folding time as the input target and compare
the average synthesized ACFs back to the physical screen ACF.
"""
    ),
    code(
        r"""
def fast_synthesis_acfs(target_t_d_s, sample_dt_s, samples, trials=250):
    acfs = {"FFT": [], "ARTA": []}
    fitted = {"FFT": [], "ARTA": []}
    for idx in range(trials):
        fft_gain = get_ts_fft(target_t_d_s, sample_dt_s, samples, pow=ALPHA, seed=idx)
        arta_gain = get_ts_arta(target_t_d_s, sample_dt_s, samples, p=16, pow=ALPHA, seed=idx)
        for name, gain in [("FFT", fft_gain), ("ARTA", arta_gain)]:
            acf = diag_stats.autocorr(gain)
            acfs[name].append(acf)
            try:
                fitted[name].append(diag_stats.fit_acf(acf, pow=ALPHA, use_triangle=True)[0] * sample_dt_s)
            except RuntimeError:
                fitted[name].append(np.nan)
    return {name: np.asarray(values) for name, values in acfs.items()}, {
        name: np.asarray(values, dtype=float) for name, values in fitted.items()
    }


target_t_d_s = representative.diagnostics["mean_efold_s"]
fast_acfs, fast_fits = fast_synthesis_acfs(target_t_d_s, dt_s(), SHAPE[1])
fast_table = pd.DataFrame([
    {
        "method": name,
        "input_t_d_s": target_t_d_s,
        "median_fit_s": np.nanmedian(values),
        "p16_fit_s": np.nanpercentile(values, 16),
        "p84_fit_s": np.nanpercentile(values, 84),
        "median_fit/input": np.nanmedian(values) / target_t_d_s,
    }
    for name, values in fast_fits.items()
])
display(fast_table.style.format({
    "input_t_d_s": "{:.2f}",
    "median_fit_s": "{:.2f}",
    "p16_fit_s": "{:.2f}",
    "p84_fit_s": "{:.2f}",
    "median_fit/input": "{:.3f}",
}))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
lags_s = np.arange(SHAPE[1]) * dt_s()
keep = lags_s <= 140

axes[0].plot(lags_s[keep], representative.diagnostics["mean_acf"][keep], lw=2, label="physical screen")
axes[0].plot(
    lags_s[keep],
    diag_stats.scint_acf(lags_s[keep], target_t_d_s, pow=ALPHA),
    "k--",
    label="target analytic ACF",
)
for name, color in [("FFT", "tab:green"), ("ARTA", "tab:red")]:
    mean = np.nanmean(fast_acfs[name], axis=0)
    p16, p84 = np.nanpercentile(fast_acfs[name], [16, 84], axis=0)
    axes[0].plot(lags_s[keep], mean[keep], color=color, label=f"{name} mean")
    axes[0].fill_between(lags_s[keep], p16[keep], p84[keep], color=color, alpha=0.12, linewidth=0)
axes[0].axhline(np.exp(-1), color="0.5", ls=":")
axes[0].set_xlabel("lag [s]")
axes[0].set_ylabel("ACF")
axes[0].set_title("Fast synthesis after physical t_d calibration")
axes[0].legend(frameon=False)

bins = np.linspace(0, 60, 35)
for name, color in [("FFT", "tab:green"), ("ARTA", "tab:red")]:
    axes[1].hist(fast_fits[name], bins=bins, histtype="step", lw=2, color=color, label=name)
axes[1].axvline(target_t_d_s, color="k", ls="--", label="input physical t_d")
axes[1].set_xlabel("fit t_d recovered from synthetic realization [s]")
axes[1].set_ylabel("count")
axes[1].set_title("Finite realizations remain biased/scattered")
axes[1].legend(frameon=False)

fast_path = FIGURE_DIR / "screen_timescale_bridge_fast_handoff.png"
fig.savefig(fast_path, dpi=180)
fast_path
"""
    ),
    markdown(
        r"""
![fast handoff](figures/screen_timescale_bridge_fast_handoff.png)

Current fast-handoff read:

- Physical-screen input target from the representative case: `t_d = 26.72 s`.
- FFT realizations recover median fitted `t_d = 21.86 s`, or `0.818` of the
  input, with broad finite-sample scatter.
- ARTA realizations recover median fitted `t_d = 21.00 s`, or `0.786` of the
  input.

That does not make FFT/ARTA useless. It means the fast path needs its own
finite-window calibration layer if we care about observed/fitted timescale
rather than generator input timescale.
"""
    ),
    markdown(
        r"""
### Fast-Handoff Read

The fast paths are useful, but they are not literal screen simulators. Even when
given the empirical physical-screen `t_d`, finite-length FFT/ARTA realizations
have their own estimator bias and scatter. That is acceptable for bulk
injection experiments if the metadata says "fast statistical surrogate", not
"physical screen realization".

My recommended API split:

- `PhysicalScreenScintillator`: slow, complex field, phase-preserving,
  calibration-grade.
- `ScreenTimescaleBridge`: derives empirical `t_d`, analytic scales, and
  validity diagnostics from physical screens.
- `FastScintillationGain`: FFT/ARTA, takes a calibrated `t_d` plus provenance.
"""
    ),
    markdown(
        r"""
## 5. Save Tables for Report/Regression Use
"""
    ),
    code(
        r"""
metrics_path = FIGURE_DIR / "screen_timescale_bridge_deep_metrics.json"
save_table_json(
    metrics_path,
    {
        "strength": strength,
        "frequency_fixed_C_n2": frequency_table,
        "geometry_fixed_C_n2": geometry_table,
        "velocity": velocity_table,
        "fast_handoff": fast_table,
    },
)

print(metrics_path)
print(strength_path)
print(variants_path)
print(velocity_path)
print(fast_path)
"""
    ),
    markdown(
        r"""
## Bottom Line

The physical screen can absolutely be used to back out a fast-synthesis
timescale, but the bridge is empirical, not a one-line analytic identity.

The cleanest result is that the full observer-plane screen gives many
horizontal ACF samples, so we can estimate a robust effective `t_d` from one
screen realization. The least clean result is that `s0 / v_eff` does not map to
that empirical `t_d` with a universal factor. The factor depends on scattering
strength, domain coverage, frequency/physical normalization, and probably the
finite-source geometry convention.

For real blscint usage, I would implement the fast path as calibrated-by-screen:
run physical screen simulations over a small parameter grid, store
`t_d_empirical`, `s0/v_eff`, `eta`, `domain/s0`, seed scatter, and screen
metadata, then let FFT/ARTA draw cheap gains from that calibrated effective
timescale.
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
print(NOTEBOOK_PATH)
