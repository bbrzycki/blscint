"""Generate c95-vs-rd18 A/B equivalence figures."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from astropy import units as u

from blscint.simulations.screens import c95, rd18


FIG_DIR = Path(__file__).resolve().parent / "figures"


def matched_models(shape=(96, 96), dx=5e5 * u.cm, m_b2=20, seed=11):
    source_distance = 2e13 * u.cm
    screen_distance = 1e13 * u.cm
    dz = 1e12 * u.cm

    c95_screen = c95.Screen(
        distance=screen_distance,
        dx=dx,
        dy=dx,
        dz=dz,
        shape=shape,
        m_b2=m_b2,
        seed=seed,
    )
    rd18_screen = rd18.Screen(
        distance=screen_distance,
        N=shape[0],
        dr=dx,
        dz=dz,
        m_b2=m_b2,
        phase_sign=1,
        seed=seed,
    )
    c95_model = c95.ScatteringModel(
        c95.RadioSource(source_distance),
        [c95_screen],
        dx=dx,
        dy=dx,
        shape=shape,
    )
    rd18_model = rd18.ScatteringModel(
        rd18.RadioSource(source_distance),
        [rd18_screen],
        N=shape[0],
        dr=dx,
    )
    return c95_screen, rd18_screen, c95_model, rd18_model


def plot_ab_equivalence():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    c95_screen, rd18_screen, c95_model, rd18_model = matched_models()

    f0 = 1 * u.GHz
    phi_c95 = c95_screen.phases(f0)
    phi_rd18 = rd18_screen.phases(f0)
    phase_delta = phi_c95 - phi_rd18

    ds_c95 = c95_model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        2 * u.MHz,
        64,
        normalize="mean",
        progress=False,
    )
    ds_rd18 = rd18_model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        2 * u.MHz,
        64,
        normalize="mean",
        progress=False,
    )
    intensity_delta = ds_c95.intensity - ds_rd18.intensity

    phase_corr = np.corrcoef(phi_c95.ravel(), phi_rd18.ravel())[0, 1]
    intensity_corr = np.corrcoef(ds_c95.intensity.ravel(), ds_rd18.intensity.ravel())[0, 1]
    max_abs_delta = np.max(np.abs(intensity_delta))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.3), constrained_layout=True)
    im00 = axes[0, 0].imshow(phi_c95, origin="lower", cmap="RdBu_r")
    axes[0, 0].set_title("c95 phase screen")
    axes[0, 0].set_xlabel("x pixel")
    axes[0, 0].set_ylabel("y pixel")
    fig.colorbar(im00, ax=axes[0, 0], label="phase [rad]")

    im01 = axes[0, 1].imshow(phi_rd18, origin="lower", cmap="RdBu_r")
    axes[0, 1].set_title("rd18 phase screen, matched mode")
    axes[0, 1].set_xlabel("x pixel")
    axes[0, 1].set_ylabel("y pixel")
    fig.colorbar(im01, ax=axes[0, 1], label="phase [rad]")

    im02 = axes[0, 2].imshow(phase_delta, origin="lower", cmap="coolwarm")
    axes[0, 2].set_title(f"phase residual\ncorr={phase_corr:.6f}")
    axes[0, 2].set_xlabel("x pixel")
    axes[0, 2].set_ylabel("y pixel")
    fig.colorbar(im02, ax=axes[0, 2], label="phase difference [rad]")

    freq = ds_c95.frequencies.to_value(u.MHz)
    extent = [freq[0], freq[-1], 0, ds_c95.intensity.shape[0] - 1]
    im10 = axes[1, 0].imshow(ds_c95.intensity, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[1, 0].set_title("c95 dynamic spectrum")
    axes[1, 0].set_xlabel("frequency [MHz]")
    axes[1, 0].set_ylabel("observer cut sample")
    fig.colorbar(im10, ax=axes[1, 0], label="gain")

    im11 = axes[1, 1].imshow(ds_rd18.intensity, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[1, 1].set_title("rd18 dynamic spectrum, matched mode")
    axes[1, 1].set_xlabel("frequency [MHz]")
    axes[1, 1].set_ylabel("observer cut sample")
    fig.colorbar(im11, ax=axes[1, 1], label="gain")

    im12 = axes[1, 2].imshow(intensity_delta, origin="lower", aspect="auto", extent=extent, cmap="coolwarm")
    axes[1, 2].set_title(f"intensity residual\ncorr={intensity_corr:.6f}, max |d|={max_abs_delta:.2e}")
    axes[1, 2].set_xlabel("frequency [MHz]")
    axes[1, 2].set_ylabel("observer cut sample")
    fig.colorbar(im12, ax=axes[1, 2], label="gain difference")

    fig.suptitle("c95 vs rd18 A/B check with shared physical screen", y=1.02, fontsize=13)
    fig.savefig(FIG_DIR / "validation_c95_rd18_ab.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "phase_corr": phase_corr,
        "intensity_corr": intensity_corr,
        "max_abs_intensity_delta": max_abs_delta,
        "mean_abs_intensity_delta": float(np.mean(np.abs(intensity_delta))),
    }


def main():
    summary = plot_ab_equivalence()
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(FIG_DIR / "validation_c95_rd18_ab.png")


if __name__ == "__main__":
    main()
