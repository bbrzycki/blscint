"""Generate quantitative validation figures for screen simulations."""

from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from astropy import units as u
from scipy import stats

from blscint.simulations.screens import c95, rd18, validation


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


def build_c95_screen(shape=(256, 256), dx=5e5 * u.cm, m_b2=10, seed=1):
    return c95.Screen(
        distance=1e13 * u.cm,
        dx=dx,
        dy=dx,
        dz=1e12 * u.cm,
        shape=shape,
        alpha=5 / 3,
        m_b2=m_b2,
        seed=seed,
    )


def build_c95_dynamic_spectrum():
    dx = 5e5 * u.cm
    shape = (128, 128)
    source = c95.RadioSource(2e13 * u.cm)
    screen = build_c95_screen(shape=shape, dx=dx, m_b2=30, seed=3)
    model = c95.ScatteringModel(source, [screen], dx=dx, dy=dx, shape=shape)
    return model.observer_dynamic_spectrum_result(
        1 * u.GHz,
        2 * u.MHz,
        128,
        v_trans=8e6 * u.cm / u.s,
        normalize="mean",
        progress=False,
    )


def plot_phase_statistics():
    alpha = 5 / 3
    beta = alpha + 2
    f = 1 * u.GHz
    dx = 5e5 * u.cm
    screen = build_c95_screen(dx=dx)
    phi = screen.phases(f)

    q, power, counts = validation.radial_power_spectrum_2d(
        phi,
        dx=dx.to_value(u.cm),
        dy=dx.to_value(u.cm),
        bins=42,
    )
    power_fit = validation.fit_power_law(q, power, xmin=q[4], xmax=q[-8])
    s, dphi = validation.phase_structure_function(
        phi,
        spacing=dx.to_value(u.cm),
        max_lag=80,
    )
    sf_fit = validation.fit_power_law(s, dphi, xmin=s[1], xmax=s[20])
    target_dphi = screen.spectrum.D_TS(s * u.cm, f).decompose().value

    power_ref = power_fit["amplitude"] * q**(-beta)
    sf_ref = sf_fit["amplitude"] * s**alpha

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    im0 = axes[0].imshow(phi, origin="lower", cmap="RdBu_r")
    axes[0].set_title("c95 phase screen")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("y pixel")
    fig.colorbar(im0, ax=axes[0], label="phase [rad]")

    axes[1].loglog(q, power, marker="o", ms=3, lw=1.3, label="radial FFT power")
    axes[1].loglog(
        q,
        power_ref,
        ls="--",
        lw=1.8,
        label=rf"target slope $-\beta=-{beta:.2f}$",
    )
    axes[1].loglog(
        power_fit["x"],
        power_fit["amplitude"] * power_fit["x"] ** power_fit["slope"],
        lw=2.0,
        label=rf"fit slope {power_fit['slope']:.2f}",
    )
    axes[1].set_title("Phase power spectrum")
    axes[1].set_xlabel(r"$q$ [cm$^{-1}$]")
    axes[1].set_ylabel("azimuthal mean power")
    axes[1].legend(frameon=False)

    axes[2].loglog(s / dx.to_value(u.cm), dphi, marker="o", ms=3, lw=1.3, label="measured")
    axes[2].loglog(s / dx.to_value(u.cm), target_dphi, lw=2.0, label="screen target")
    axes[2].loglog(
        s / dx.to_value(u.cm),
        sf_ref,
        ls="--",
        lw=1.8,
        label=rf"target slope $\alpha={alpha:.2f}$",
    )
    axes[2].loglog(
        sf_fit["x"] / dx.to_value(u.cm),
        sf_fit["amplitude"] * sf_fit["x"] ** sf_fit["slope"],
        lw=2.0,
        label=rf"fit slope {sf_fit['slope']:.2f}",
    )
    axes[2].axvline(
        screen.spectrum.s0(f).to_value(u.cm) / dx.to_value(u.cm),
        color="0.35",
        ls=":",
        lw=1.4,
        label=r"$s_0$",
    )
    axes[2].set_title("Phase structure function")
    axes[2].set_xlabel("lag [pixels]")
    axes[2].set_ylabel(r"$D_\phi$ [rad$^2$]")
    axes[2].legend(frameon=False)

    fig.suptitle("Quantitative c95 phase-screen validation", y=1.03, fontsize=13)
    fig.savefig(FIG_DIR / "validation_c95_phase_statistics.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "power_slope": power_fit["slope"],
        "target_power_slope": -beta,
        "structure_slope": sf_fit["slope"],
        "target_structure_slope": alpha,
    }


def plot_intensity_statistics():
    ds = build_c95_dynamic_spectrum()
    intensity = ds.intensity
    gain_diag = validation.exponential_gain_diagnostic(intensity)
    time_lags, time_acf = validation.mean_axis_autocorrelation(intensity, axis=0)
    freq_lags, freq_acf = validation.mean_axis_autocorrelation(intensity, axis=1)
    time_decorr = validation.decorrelation_lag(time_lags, time_acf)
    freq_decorr = validation.decorrelation_lag(freq_lags, freq_acf)

    time = ds.time_axis.to_value(u.s)
    freq = ds.frequencies.to_value(u.MHz)
    extent = [freq[0], freq[-1], time[0], time[-1]]

    x = np.linspace(0, np.nanpercentile(intensity, 99.5), 400)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    im0 = axes[0].imshow(intensity, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    axes[0].set_title("Mean-normalized dynamic spectrum")
    axes[0].set_xlabel("frequency [MHz]")
    axes[0].set_ylabel("time [s]")
    fig.colorbar(im0, ax=axes[0], label="gain")

    axes[1].hist(intensity.ravel(), bins=70, density=True, alpha=0.75, label="simulation")
    axes[1].plot(x, stats.expon.pdf(x), lw=2.2, label="unit-mean exponential")
    axes[1].set_title(
        "Intensity PDF\n"
        f"var={gain_diag['variance']:.2f}, KS={gain_diag['ks_statistic']:.3f}"
    )
    axes[1].set_xlabel("mean-normalized intensity")
    axes[1].set_ylabel("density")
    axes[1].legend(frameon=False)

    axes[2].plot(time_lags, time_acf, lw=2.0, label=rf"time ACF, $t_d={time_decorr:.2f}$ bins")
    axes[2].plot(freq_lags, freq_acf, lw=2.0, label=rf"frequency ACF, $\nu_d={freq_decorr:.2f}$ bins")
    axes[2].axhline(np.exp(-1), color="0.3", ls=":", lw=1.4, label=r"$e^{-1}$")
    axes[2].set_xlim(0, 20)
    axes[2].set_ylim(-0.2, 1.05)
    axes[2].set_title("Scintle correlation scales")
    axes[2].set_xlabel("lag [bins]")
    axes[2].set_ylabel("normalized ACF")
    axes[2].legend(frameon=False)

    fig.suptitle("Quantitative c95 intensity and decorrelation validation", y=1.03, fontsize=13)
    fig.savefig(FIG_DIR / "validation_c95_intensity_statistics.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "intensity_variance": gain_diag["variance"],
        "ks_statistic": gain_diag["ks_statistic"],
        "time_decorrelation_bins": time_decorr,
        "frequency_decorrelation_bins": freq_decorr,
    }


def plot_ravi_correlation_validation():
    source = rd18.RadioSource(2e10 * u.m)
    screen = rd18.Screen(
        distance=1e10 * u.m,
        N=128,
        dr=5e5 * u.m,
        dz=1e10 * u.m,
        alpha=5 / 3,
        C_n2=1e21 * u.m ** (-20 / 3),
        seed=2,
    )
    model = rd18.ScatteringModel(source, [screen], N=128, dr=5e5 * u.m)
    ds = model.observer_dynamic_spectrum_result(
        270 * u.MHz,
        1 * u.MHz,
        128,
        v_trans=1e5 * u.m / u.s,
        normalize="mean",
        progress=False,
    )

    on = ds.intensity
    rng = np.random.default_rng(8)
    eta_true = 0.08
    noise = rng.normal(0, 0.08 * np.std(on), size=on.shape)
    off_matched = eta_true * on + noise
    shift = (14, 9)
    off_shifted = eta_true * np.roll(on, shift=shift, axis=(0, 1)) + noise
    corr_matched = validation.normalized_cross_correlation_2d(on, off_matched)
    corr_shifted = validation.normalized_cross_correlation_2d(on, off_shifted)
    matched_peak = validation.peak_correlation_lag(corr_matched)
    shifted_peak = validation.peak_correlation_lag(corr_shifted)

    def eta_hat(off):
        don = on - np.mean(on)
        doff = off - np.mean(off)
        return float(np.sum(don * doff) / np.sum(don**2))

    cc_extent = [
        -corr_matched.shape[1] // 2,
        corr_matched.shape[1] // 2,
        -corr_matched.shape[0] // 2,
        corr_matched.shape[0] // 2,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    im0 = axes[0].imshow(on, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title("RD18 on-source gain")
    axes[0].set_xlabel("frequency bin")
    axes[0].set_ylabel("time bin")
    fig.colorbar(im0, ax=axes[0], label="gain")

    im1 = axes[1].imshow(corr_matched, origin="lower", aspect="auto", extent=cc_extent, cmap="coolwarm")
    axes[1].set_title(
        "Matched off-source\n"
        f"eta_hat={eta_hat(off_matched):.3f}, peak lag={matched_peak[:2]}"
    )
    axes[1].set_xlabel("frequency lag [bins]")
    axes[1].set_ylabel("time lag [bins]")
    fig.colorbar(im1, ax=axes[1], label="normalized correlation")

    im2 = axes[2].imshow(corr_shifted, origin="lower", aspect="auto", extent=cc_extent, cmap="coolwarm")
    axes[2].set_title(
        "Shifted/aliased off-source\n"
        f"zero-lag eta_hat={eta_hat(off_shifted):.3f}, peak lag={shifted_peak[:2]}"
    )
    axes[2].set_xlabel("frequency lag [bins]")
    axes[2].set_ylabel("time lag [bins]")
    fig.colorbar(im2, ax=axes[2], label="normalized correlation")

    fig.suptitle("Ravi & Deshpande correlation validation toy model", y=1.04, fontsize=13)
    fig.savefig(FIG_DIR / "validation_rd18_correlation.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "eta_true": eta_true,
        "eta_matched": eta_hat(off_matched),
        "eta_shifted_zero_lag": eta_hat(off_shifted),
        "matched_peak_lag": matched_peak[:2],
        "shifted_peak_lag": shifted_peak[:2],
        "applied_shift": shift,
    }


def main():
    configure_plots()
    summaries = {
        "phase": plot_phase_statistics(),
        "intensity": plot_intensity_statistics(),
        "ravi_correlation": plot_ravi_correlation_validation(),
    }
    for name, summary in summaries.items():
        print(name)
        for key, value in summary.items():
            print(f"  {key}: {value}")
    for path in sorted(FIG_DIR.glob("validation_*.png")):
        print(path)


if __name__ == "__main__":
    main()
