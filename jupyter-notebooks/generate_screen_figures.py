"""Generate figures used by the screen propagation notebooks."""

from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from astropy import units as u
from scipy import signal

from blscint.simulations.screens import c95, hl07, rd18
from blscint.simulations.screens.base_classes import get_k


FIG_DIR = Path(__file__).resolve().parent / "figures"


def configure_plots():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 170,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def structure_function_x(phi, max_lag=None):
    if max_lag is None:
        max_lag = phi.shape[1] // 3
    lags = np.arange(1, max_lag + 1)
    values = np.array([
        np.mean((phi[:, lag:] - phi[:, :-lag]) ** 2)
        for lag in lags
    ])
    return lags, values


def normalized_corr2d(a, b):
    da = a - np.mean(a)
    db = b - np.mean(b)
    corr = signal.correlate2d(db, da, mode="same", boundary="fill")
    denom = np.sqrt(np.sum(da**2) * np.sum(db**2))
    return corr / denom


def eta_hat(on, off):
    don = on - np.mean(on)
    doff = off - np.mean(off)
    return float(np.sum(don * doff) / np.sum(don**2))


def plot_transfer_functions():
    f = 1 * u.GHz
    k = get_k(f).to(1 / u.cm)
    eta = np.linspace(0.0, 0.85, 700)
    q = eta * k
    z = 5e4 * u.cm

    fresnel_phase_norm = -0.5 * eta**2
    angular_phase_norm = np.real(np.sqrt(1 - eta**2 + 0j) - 1)
    phase_error_norm = angular_phase_norm - fresnel_phase_norm
    fresnel_kernel = hl07.fresnel_transfer_function(q, z, f)
    angular_kernel = hl07.angular_spectrum_transfer_function(q, z, f)
    complex_mismatch = np.abs(angular_kernel - fresnel_kernel)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(eta, angular_phase_norm, lw=2.2, label="Angular spectrum")
    axes[0].plot(eta, fresnel_phase_norm, lw=2.2, ls="--", label="Fresnel/paraxial")
    axes[0].plot(eta, phase_error_norm, lw=1.8, c="0.25", label="Difference")
    axes[0].set_xlabel(r"Transverse wavenumber ratio $q/k$")
    axes[0].set_ylabel(r"Carrier-removed phase / $kz$")
    axes[0].set_title("Free-space kernel phase")
    axes[0].legend(frameon=False)

    axes[1].plot(eta, complex_mismatch, lw=2.2, c="tab:red")
    axes[1].axvline(0.25, color="0.35", lw=1.2, ls=":", label="q/k = 0.25")
    axes[1].axvline(0.5, color="0.55", lw=1.2, ls=":", label="q/k = 0.50")
    axes[1].set_xlabel(r"Transverse wavenumber ratio $q/k$")
    axes[1].set_ylabel(r"$|H_{AS} - H_F|$")
    axes[1].set_title("Complex transfer-function mismatch")
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Li 2007 kernel check: exact angular spectrum vs Fresnel approximation",
        y=1.04,
        fontsize=13,
    )
    fig.savefig(FIG_DIR / "screen_transfer_functions.png", bbox_inches="tight")
    plt.close(fig)


def plot_c95_behavior():
    source = c95.RadioSource(2e13 * u.cm)
    screen = c95.Screen(
        distance=1e13 * u.cm,
        dx=4e8 * u.cm,
        dy=4e8 * u.cm,
        dz=1e12 * u.cm,
        shape=(128, 128),
        alpha=5 / 3,
        m_b2=10.0,
        seed=4,
    )
    model = c95.ScatteringModel(
        source,
        [screen],
        dx=4e8 * u.cm,
        dy=4e8 * u.cm,
        shape=(128, 128),
    )

    phi = screen.phases(1 * u.GHz)
    lags, dphi = structure_function_x(phi)
    valid = dphi > 0
    mid = len(dphi[valid]) // 2
    ref = dphi[valid][mid] * (lags / lags[valid][mid]) ** (5 / 3)
    ds = model.observer_dynamic_spectrum_result(
        1.0 * u.GHz,
        2.0 * u.MHz,
        96,
        v_trans=8e6 * u.cm / u.s,
        normalize="mean",
        progress=False,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    im0 = axes[0].imshow(phi, origin="lower", cmap="RdBu_r")
    axes[0].set_title("Phase screen")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")
    fig.colorbar(im0, ax=axes[0], label="phase [rad]")

    axes[1].loglog(lags, dphi, lw=2.0, label="Measured x-lag structure")
    axes[1].loglog(lags, ref, ls="--", lw=1.8, label=r"$5/3$ reference")
    axes[1].set_xlabel("lag [pixels]")
    axes[1].set_ylabel(r"$D_\phi$ [rad$^2$]")
    axes[1].set_title("Structure-function sanity check")
    axes[1].legend(frameon=False)

    time = ds.time_axis.to_value(u.s)
    freq = ds.frequencies.to_value(u.MHz)
    extent = [freq[0], freq[-1], time[0], time[-1]]
    c95_fluctuation = ds.intensity - 1
    c95_limit = np.nanpercentile(np.abs(c95_fluctuation), 99)
    im2 = axes[2].imshow(
        c95_fluctuation,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-c95_limit,
        vmax=c95_limit,
    )
    axes[2].set_title("Observer dynamic spectrum")
    axes[2].set_xlabel("frequency [MHz]")
    axes[2].set_ylabel("time from spatial cut [s]")
    fig.colorbar(im2, ax=axes[2], label="fractional intensity fluctuation")

    fig.suptitle("Coles-style screen behavior (c95)", y=1.04, fontsize=13)
    fig.savefig(FIG_DIR / "c95_phase_dynamic_spectrum.png", bbox_inches="tight")
    plt.close(fig)


def plot_rd18_behavior():
    source = rd18.RadioSource(2e10 * u.m)
    screen = rd18.Screen(
        distance=1e10 * u.m,
        N=128,
        dr=5e5 * u.m,
        dz=1e10 * u.m,
        alpha=5 / 3,
        # Deliberately boosted for a compact visual demo; Appendix B of
        # Ravi & Deshpande also uses artificial screen parameters to keep the
        # dynamic range computationally feasible.
        C_n2=1e21 * u.m ** (-20 / 3),
        seed=2,
    )
    model = rd18.ScatteringModel(source, [screen], N=128, dr=5e5 * u.m)

    phi = screen.phases(270 * u.MHz)
    ds = model.observer_dynamic_spectrum_result(
        270 * u.MHz,
        1.0 * u.MHz,
        96,
        v_trans=1e5 * u.m / u.s,
        normalize="mean",
        progress=False,
    )
    on = ds.intensity
    rng = np.random.default_rng(8)
    eta_true = 0.08
    noise = rng.normal(0, 0.08 * np.std(on), size=on.shape)
    off_matched = eta_true * on + noise
    off_shifted = eta_true * np.roll(on, shift=(14, 9), axis=(0, 1)) + noise
    corr_matched = normalized_corr2d(on, off_matched)
    corr_shifted = normalized_corr2d(on, off_shifted)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), constrained_layout=True)
    time = ds.time_axis.to_value(u.s)
    freq = ds.frequencies.to_value(u.MHz)
    extent = [freq[0], freq[-1], time[0], time[-1]]

    im00 = axes[0, 0].imshow(phi, origin="lower", cmap="RdBu_r")
    axes[0, 0].set_title("RD18 phase screen")
    axes[0, 0].set_xlabel("x pixel")
    axes[0, 0].set_ylabel("y pixel")
    fig.colorbar(im00, ax=axes[0, 0], label="phase [rad]")

    log_on = np.log10(np.clip(on, 1e-12, None))
    im01 = axes[0, 1].imshow(log_on, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[0, 1].set_title("On-source dynamic spectrum")
    axes[0, 1].set_xlabel("frequency [MHz]")
    axes[0, 1].set_ylabel("time [s]")
    fig.colorbar(im01, ax=axes[0, 1], label="log10 gain")

    im02 = axes[0, 2].imshow(
        off_matched,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[0, 2].set_title(f"Matched off-source toy model\neta_hat={eta_hat(on, off_matched):.3f}")
    axes[0, 2].set_xlabel("frequency [MHz]")
    axes[0, 2].set_ylabel("time [s]")
    fig.colorbar(im02, ax=axes[0, 2], label="relative intensity")

    cc_extent = [
        -corr_matched.shape[1] // 2,
        corr_matched.shape[1] // 2,
        -corr_matched.shape[0] // 2,
        corr_matched.shape[0] // 2,
    ]
    im10 = axes[1, 0].imshow(
        corr_matched,
        aspect="auto",
        origin="lower",
        extent=cc_extent,
        cmap="coolwarm",
    )
    axes[1, 0].set_title("Matched cross-correlation")
    axes[1, 0].set_xlabel("frequency lag [channels]")
    axes[1, 0].set_ylabel("time lag [samples]")
    fig.colorbar(im10, ax=axes[1, 0], label="normalized correlation")

    im11 = axes[1, 1].imshow(
        off_shifted,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    axes[1, 1].set_title(
        f"Shifted/aliased toy model\nzero-lag eta_hat={eta_hat(on, off_shifted):.3f}"
    )
    axes[1, 1].set_xlabel("frequency [MHz]")
    axes[1, 1].set_ylabel("time [s]")
    fig.colorbar(im11, ax=axes[1, 1], label="relative intensity")

    im12 = axes[1, 2].imshow(
        corr_shifted,
        aspect="auto",
        origin="lower",
        extent=cc_extent,
        cmap="coolwarm",
    )
    axes[1, 2].set_title("Shifted cross-correlation")
    axes[1, 2].set_xlabel("frequency lag [channels]")
    axes[1, 2].set_ylabel("time lag [samples]")
    fig.colorbar(im12, ax=axes[1, 2], label="normalized correlation")

    fig.suptitle(
        "Ravi & Deshpande-style dynamic spectrum and correlation behavior",
        y=1.02,
        fontsize=13,
    )
    fig.savefig(FIG_DIR / "rd18_dynamic_correlation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    configure_plots()
    plot_transfer_functions()
    plot_c95_behavior()
    plot_rd18_behavior()
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
