"""Generate figures for a perfect narrowband tone through a screen model."""

from pathlib import Path
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from astropy import units as u
from scipy import stats

from blscint.simulations.screens import c95


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


def build_model(shape=(160, 160), dx=5e5 * u.cm, m_b2=30, seed=5):
    source = c95.RadioSource(2e13 * u.cm)
    screen = c95.Screen(
        distance=1e13 * u.cm,
        dx=dx,
        dy=dx,
        dz=1e12 * u.cm,
        shape=shape,
        alpha=5 / 3,
        m_b2=m_b2,
        seed=seed,
    )
    model = c95.ScatteringModel(source, [screen], dx=dx, dy=dx, shape=shape)
    return screen, model


def delta_channel_spectrogram(profile, fchans=33, channel_index=None):
    """Place the one-channel tone profile into a toy channelized spectrogram."""
    if channel_index is None:
        channel_index = fchans // 2
    data = np.zeros((profile.intensity.shape[0], fchans))
    data[:, channel_index] = profile.intensity[:, 0]
    return data, channel_index


def plot_narrowband_tone_profile():
    configure_plots()
    f0 = 1 * u.GHz
    df = 250 * u.kHz
    v_trans = 8e6 * u.cm / u.s

    screen, model = build_model()
    tone_field = model.observer_narrowband_tone_field(
        f0,
        v_trans=v_trans,
    )
    profile = tone_field.as_dynamic_spectrum(
        intrinsic_intensity=1.0,
        normalize="mean",
    )
    context = model.observer_dynamic_spectrum_result(
        f0 - 16 * df,
        df,
        33,
        v_trans=v_trans,
        normalize="mean",
        progress=False,
    )
    tone_spec, tone_idx = delta_channel_spectrogram(profile, fchans=33)

    time = profile.time_axis.to_value(u.s)
    freq_offset = (context.frequencies - f0).to_value(u.MHz)
    extent = [freq_offset[0], freq_offset[-1], time[0], time[-1]]
    gain = profile.intensity[:, 0]

    fig = plt.figure(figsize=(14, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_context = fig.add_subplot(gs[0, 0])
    ax_tone = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_voltage = fig.add_subplot(gs[1, 2])

    context_fluct = context.intensity - 1
    fluct_lim = np.nanpercentile(np.abs(context_fluct), 99)
    im_context = ax_context.imshow(
        context_fluct,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-fluct_lim,
        vmax=fluct_lim,
    )
    ax_context.axvline(0, color="k", lw=1.1, ls=":")
    ax_context.set_title("Screen gain around the tone")
    ax_context.set_xlabel("frequency offset [MHz]")
    ax_context.set_ylabel("time from spatial cut [s]")
    fig.colorbar(im_context, ax=ax_context, label="fractional gain")

    im_tone = ax_tone.imshow(
        tone_spec,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=np.nanpercentile(gain, 99),
    )
    ax_tone.axvline(freq_offset[tone_idx], color="w", lw=1.1, ls=":")
    ax_tone.set_title("Perfect delta-channel tone")
    ax_tone.set_xlabel("frequency offset [MHz]")
    ax_tone.set_ylabel("time from spatial cut [s]")
    fig.colorbar(im_tone, ax=ax_tone, label="observed intensity")

    ax_profile.plot(time, gain, lw=1.8, color="tab:blue")
    ax_profile.axhline(1, color="0.3", lw=1.1, ls=":")
    ax_profile.set_title("Tone intensity profile")
    ax_profile.set_xlabel("time from spatial cut [s]")
    ax_profile.set_ylabel("mean-normalized intensity")

    x = np.linspace(0, np.nanpercentile(gain, 99.5), 400)
    ax_hist.hist(gain, bins=36, density=True, alpha=0.78, label="tone samples")
    ax_hist.plot(x, stats.expon.pdf(x), lw=2.1, label="unit-mean exponential")
    ax_hist.set_title("Single-frequency gain PDF")
    ax_hist.set_xlabel("mean-normalized intensity")
    ax_hist.set_ylabel("density")
    ax_hist.legend(frameon=False)

    phi = screen.phases(f0)
    im_phase = ax_phase.imshow(phi, origin="lower", cmap="RdBu_r")
    ax_phase.axhline(screen.Ny // 2, color="k", lw=1.0, ls=":")
    ax_phase.set_title("Phase screen and sampled row")
    ax_phase.set_xlabel("x pixel")
    ax_phase.set_ylabel("y pixel")
    fig.colorbar(im_phase, ax=ax_phase, label="phase [rad]")

    center_index = len(tone_field.electric_field) // 2
    voltage_time = np.linspace(0, 5, 700) * u.ns
    voltage = tone_field.sample_voltage(voltage_time, spatial_index=center_index)
    ax_voltage.plot(voltage_time.to_value(u.ns), voltage, lw=1.6, color="tab:purple")
    ax_voltage.axhline(0, color="0.3", lw=1.1, ls=":")
    ax_voltage.set_title("Passband voltage at one observer point")
    ax_voltage.set_xlabel("local voltage time [ns]")
    ax_voltage.set_ylabel("voltage [arb.]")

    fig.suptitle(
        "Perfect narrowband sine tone through a Coles-style scattering screen",
        y=1.02,
        fontsize=13,
    )
    output = FIG_DIR / "narrowband_tone_intensity_profile.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure": str(output),
        "frequency_MHz": f0.to_value(u.MHz),
        "samples": int(profile.intensity.shape[0]),
        "mean_gain": float(np.mean(gain)),
        "variance": float(np.var(gain)),
        "min_gain": float(np.min(gain)),
        "max_gain": float(np.max(gain)),
        "center_field_phase_rad": float(tone_field.phase[center_index]),
        "center_field_amplitude": float(np.abs(tone_field.electric_field[center_index])),
    }


def main():
    summary = plot_narrowband_tone_profile()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
