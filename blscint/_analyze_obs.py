import os
import sys
import glob
import numpy as np
import pandas as pd
import setigen as stg
import blimpy as bl
import matplotlib.pyplot as plt
import tqdm
import collections

from astropy import units as u
from astropy.stats import sigma_clip
import scipy.stats

from turbo_seti.find_doppler.find_doppler import FindDoppler

from . import bounds
from . import dataframe
from . import frame_processing
from . import diag_stats
from . import gen_arta

import jort


def as_file_list(fns, node_excludes=[], str_excludes=[]):
    """
    Expand files, using glob pattern matching, into a full list.
    In addition, user can specify strings to exclude in any filenames.
    
    Parameters
    ----------
    fns : list
        List of files or patterns (i.e. for use with glob)
    node_excludes : list, optional
        List which nodes should be excluded from analysis, particularly
        for overlapped spectrum
    str_excludes : list, optional
        List of strings that shouldn't appear in filenames
        
    Returns
    -------
    fns : list
        Returned list of all suitable filenames
    """
    if not isinstance(fns, list):
        fns = [fns]
    fns = [fn for exp_fns in fns for fn in glob.glob(exp_fns)]
    fns.sort()
    for exclude_str in node_excludes:
        exclude_str = f"{int(exclude_str):02d}"
        fns = [fn for fn in fns if f"blc{exclude_str}" not in fn]
    for exclude_str in str_excludes:
        fns = [fn for fn in fns if exclude_str not in fn]
    return fns


def run_turboseti(obs_fns, min_drift=0.00001, max_drift=5, snr=10, out_dir='.', gpu_id=0, replace_existing=False):
    """
    Run TurboSETI on all observation files. 
    Accept observation as input, return and save csv as output (via pandas).
    
    Parameters
    ----------
    obs_fns : list
        List of files or patterns for analysis
    min_drift : float, optional
        Minimum drift rate, absolute value
    max_drift : float, optional
        Maximum drift rate, absolute value
    snr : float, optional
        SNR threshold
    out_dir : str, optional
        Output directory for .dat files
    gpu_id : int, optional
        ID of GPU used for analysis
    replace_existing : bool, optional
        Option to overwrite existing .dat files
        
    Returns
    -------
    turbo_dat_list : list
        List of all turboseti .dat files created
    """
    tr = jort.Tracker()
    turbo_dat_list = []
    for data_fn in as_file_list(obs_fns):
       
        # First, check if equivalent h5 data file exists in either source or target directory
        h5_fn_old = f"{os.path.splitext(data_fn)[0]}.h5"
        h5_fn_new = f"{out_dir}/{os.path.splitext(os.path.basename(data_fn))[0]}.h5"
        if os.path.exists(h5_fn_old):
            data_fn = h5_fn_old
        elif os.path.exists(h5_fn_new):
            print("Using H5 file in target directory")
            data_fn = h5_fn_new
        if gpu_id == 5:
            gpu_backend=False
            gpu_id=0
        else:
            gpu_backend=True
        turbo_dat_fn = f"{out_dir}/{os.path.splitext(os.path.basename(data_fn))[0]}.dat"
        if not os.path.exists(turbo_dat_fn) or replace_existing:
            tr.start('turboseti')
            find_seti_event = FindDoppler(data_fn,
                                          min_drift=min_drift,
                                          max_drift=max_drift,
                                          snr=snr,
                                          out_dir=out_dir,
                                          gpu_backend=gpu_backend,
                                          gpu_id=gpu_id,
                                          precision=1)
            find_seti_event.search()
            turbo_dat_list.append(turbo_dat_fn)
            tr.stop('turboseti')
    tr.report()
    return turbo_dat_list


def get_bbox_frame(index, df):
    """
    Return dedrifted frame for a given index, if all relevant statistics
    are in the dataframe. 

    Parameters
    ----------
    index : int
        Signal index
    df : DataFrame
        Pandas dataframe with TurboSETI parameters
    """
    row = df.loc[index]
    param_dict = dataframe.get_frame_params(row['fn'])
    frame = dataframe.turbo_centered_frame(index, df, row['fn'], row['fchans'], **param_dict)
    frame = stg.dedrift(frame)
    return frame


def empty_ts_stats(fchans):
    """
    Produce dictionary with empty values for all time series statistics.

    Parameters
    ----------
    fchans : int
        Number of frequency channels 

    Returns
    -------
    stats : dict
        Dictionary of statistics
    """
    ts_stats = {
        'std': None,
        'min': None,
        'ks': None,
        'anderson': None,
        'lag1': None,
        'lag2': None,
        'fchans': fchans,
        'l': None,
        'r': None,
        # 'acf_t_d': None,
        # 'acf_A': None,
        # 'acf_W': None,
    }
    for label, pow in [('sq', 2), ('k', 5/3)]:
        for use_triangle in [True, False]:
            ts_stats[f'acf_t_d.{label}.{use_triangle}'] = None
            ts_stats[f'acf_A.{label}.{use_triangle}'] = None
            ts_stats[f'acf_W.{label}.{use_triangle}'] = None
    return ts_stats


def run_bbox_stats(turbo_dat_fns, 
                   data_dir='.', 
                   data_ext='.fil', 
                   data_res_ext='.0005', 
                   replace_existing=False,
                   bound_type='threshold',
                   divide_std=False):
    """
    Accept TurboSETI .dat files as input, return and save csv as output (via pandas).
    Boundary box statistics.

    Parameters
    ----------
    turbo_dat_fns : list
        List of files or patterns for analysis
    data_dir : str, optional
        Location of data
    data_ext : str, optional
        File extension of data
    data_res_ext : str, optional
        Resolution code of data. Changing this gives you the option of using TurboSETI
        results on one resolution with data of another.
    replace_existing : bool, optional
        Option to overwrite existing .csv files
    bound_type : str, optional
        Type of frequency bounding to use, between 'snr' and 'threshold'
    divide_std : bool, optional
        Normalize each spectrum by dividing by its standard deviation 
        
    Returns
    -------
    csv_list : list
        List of all .csv files created
    """
    tr = jort.Tracker()
    csv_list = []
    for turbo_dat_fn in as_file_list(turbo_dat_fns):
        print(f"Working on {turbo_dat_fn}")
        data_fn = f"{data_dir}/{os.path.splitext(os.path.basename(turbo_dat_fn))[0][:-5]}{data_res_ext}{data_ext}"
        csv_fn = f"{os.path.splitext(turbo_dat_fn)[0][:-5]}{data_res_ext}_bbox_{bound_type}.csv"
        
        # Skip if csv already exists
        if not os.path.exists(csv_fn) or replace_existing:
            df = dataframe.make_dataframe(turbo_dat_fn)
            param_dict = dataframe.get_frame_params(data_fn)

            ts_stats_dict = collections.defaultdict(list)
            for index, row in tqdm.tqdm(df.iterrows()):
                found_peak = False
                fchans = 256
                while not found_peak:
                    try:
                        tr.start('frame_init')
                        frame = dataframe.turbo_centered_frame(index, df, data_fn, fchans, **param_dict)
                        frame = stg.dedrift(frame)
                        tr.stop('frame_init')
                                
                        spec = frame.integrate()

                        tr.start('polyfit')
                        l, r, metadata = bounds.polyfit_bounds(spec, deg=1, snr_threshold=10)
                        tr.stop('polyfit')

                        found_peak = True
                    except ValueError:
                        # If no fit found, or out of bounds
                        fchans *= 2
                        tr.remove('polyfit')
                    except IndexError:
                        # Broadband interferer
                        l, r, metadata = None, None, None
                        ts_stats = empty_ts_stats(fchans)
                        tr.remove('polyfit')
                        break

                # If IndexError... was probably not narrowband signal,
                # so just skip adding it in
                if l is not None:
                    try:
                        tr.start('bounds')
                        if bound_type == 'snr':
                            l, r, metadata = bounds.snr_bounds(spec, snr=5)
                        else:
                            l, r, metadata = bounds.threshold_baseline_bounds(spec)
                        # print(l,r)
                        tr.stop('bounds')

                        n_frame = frame_processing.tnorm(frame, divide_std=divide_std)
                        tr_frame = n_frame.get_slice(l, r)

                        # Get time series and normalize
                        ts = tr_frame.integrate('f')
                        ts = ts / np.mean(ts)

                        ts_stats = diag_stats.get_stats(ts)
                        ts_stats['fchans'] = fchans
                        ts_stats['l'] = l
                        ts_stats['r'] = r

                    except IndexError:
                        tr.remove('bounds')
                        ts_stats = empty_ts_stats(fchans)
                for key in ts_stats:
                    ts_stats_dict[f"{key}"].append(ts_stats[key])

            # Set statistic columns
            for key in ts_stats_dict:
                df[key] = ts_stats_dict[key]

            df['fn'] = data_fn
            df['node'] = os.path.basename(data_fn)[:5]

            df.to_csv(csv_fn, index=False)
        csv_list.append(csv_fn)
    tr.report()
    return csv_list


def plot_snapshot(index, df, divide_std=False):
    """
    Plot a single signal, with time series, normalized spectrogram, 
    and autocorrelation.

    Parameters
    ----------
    index : int
        Signal index
    df : DataFrame
        Pandas dataframe with TurboSETI parameters
    divide_std : bool, optional
        Normalize each spectrum by dividing by its standard deviation 
    """
    row = df.loc[index]
    
    param_dict = dataframe.get_frame_params(row['fn'])
    frame = dataframe.turbo_centered_frame(index, df, row['fn'], row['fchans'], **param_dict)
    dd_frame = stg.dedrift(frame)

    spec = dd_frame.integrate()

    l, r, metadata = bounds.threshold_baseline_bounds(spec)

    n_frame = frame_processing.tnorm(dd_frame, divide_std=divide_std)
    tr_frame = n_frame.get_slice(l, r)

    # Get time series and normalize
    ts = tr_frame.integrate('f')
    ts = ts / np.mean(ts)

    ts_stats = diag_stats.get_stats(ts)
    
    print(f"SNR : {row['SNR']:.3}")
    for stat in ts_stats:
        print(f"{stat:<4}: {ts_stats[stat]:.3}")
    print(f"l, r: {l}, {r}")
    
    plt.figure(figsize=(20, 3))
    plt.subplot(1, 4, 1)
    frame.bl_plot()
    plt.title(f'Index {index}')
    
    plt.subplot(1, 4, 2)
    bounds.plot_bounds(n_frame, l, r)
    plt.title(f"Drift rate: {row['DriftRate']:.3} Hz/s")
    
    plt.subplot(1, 4, 3)
    plt.plot(ts, c='k')
    plt.axhline(0, ls='--')
    plt.axhline(1, ls='-')
    plt.xlabel('Time sample')
    plt.ylabel('Normalized Intensity')
    plt.title('Time series')
    
    plt.subplot(1, 4, 4)
    acf = diag_stats.autocorr(ts)
    plt.plot(acf, c='k')
    plt.axhline(0, ls='--')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f"ACF: ks={row['ks']:.3}")
    plt.show()
    

def plot_bounded_frame(index, df):
    """
    Plot bounded, dedrifted signal as a frame.

    Parameters
    ----------
    index : int
        Signal index
    df : DataFrame
        Pandas dataframe with TurboSETI parameters
    """
    row = df.loc[index]
    
    param_dict = dataframe.get_frame_params(row['fn'])
    frame = dataframe.turbo_centered_frame(index, df, row['fn'], row['fchans'], **param_dict)
    dd_frame = stg.dedrift(frame)

    spec = dd_frame.integrate()

    l, r, metadata = bounds.threshold_baseline_bounds(spec)

    tr_frame = dd_frame.get_slice(l, r)
    tr_frame.plot()
    plt.show()
    
    
def plot_random_snapshots(df, n=1):
    """
    Plot n signals from a dataframe.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe with TurboSETI parameters
    """
    df_sampled = df.sample(n=n)
    for i in df_sampled.index:
        plot_snapshot(i, df_sampled)
        
        
def plot_all_snapshots(df):
    """
    Plot all signals from a dataframe.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe with TurboSETI parameters
    """
    for i in df.index:
        plot_snapshot(i, df)
    


def get_bbox_df(csv_fns):
    """
    Read in csvs with bbox statistics calculated and compile into
    Pandas dataframe.

    Parameters
    ----------
    csv_fns : list
        List of files or patterns for analysis
        
    Returns
    -------
    data_df : DataFrame
        Compiled dataframe
    """
    df_list = [pd.read_csv(fn) for fn in as_file_list(csv_fns)]
    data_df = pd.concat(df_list, ignore_index=True)
    
    # Exclude DC bin (value depends on rawspec fftlength)
    # print('Before DC bins (may be excluded by TurboSETI):', data_df.shape)
    data_df = data_df[data_df['ChanIndx'] != 524288]
    # print('After removing:', data_df.shape)
    
    # # Exclude first compute node
    # data_df = data_df[data_df['fn'].apply(lambda x: x.split('/')[-1][3:5] != '00')]
    
    # Remove non-fit signals (which are replaced with NaN)
    data_df = data_df[data_df['ks'].notna()]
    return data_df
        
        
def plot_bbox_stats(csv_fns, 
                    pow=5/3, 
                    use_triangle=True, 
                    bound_type='threshold',
                    divide_std=False, 
                    plot_fn_prefix='bbox_stats'):
    """
    Make stats plots with RFI and synthetic signals, and save result as a pdf.

    Parameters
    ----------
    csv_fns : list
        List of files or patterns for analysis
    pow : float, optional
        Exponent for ACF fit, either 5/3 or 2 (arising from phase structure function) 
    use_triangle : bool, optional
        Option to use triangle function to modulate modeled ACF (default: True)
    bound_type : str, optional
        Type of frequency bounding to use, between 'snr' and 'threshold'
    divide_std : bool, optional
        Normalize each spectrum by dividing by its standard deviation 
    plot_fn_prefix : str, optional
        Filename prefix for plot
    """
    data_df = get_bbox_df(csv_fns)
    
    # Simulate signals
    tr = jort.Tracker()
    n_samples = 1000

    synth_stats_dicts = {}
    sample_frame = stg.Frame.from_backend_params(
                                fchans=256,
                                obs_length=600, 
                                sample_rate=3e9, 
                                num_branches=1024,
                                fftlength=1048576,
                                int_factor=13,
                                fch1=8*u.GHz,
                                ascending=False)
    for t_d in [10, 30, 100]:
        tr.start('synthesize_bbox')
        ts_stats_dict = collections.defaultdict(list)

        for _ in range(n_samples):
            ts = gen_arta.get_ts_arta(t_d, sample_frame.dt, sample_frame.tchans, p=32, pow=pow)
            frame = stg.Frame(**sample_frame.get_params())
            frame.add_noise_from_obs()
            signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(128), 
                                                        drift_rate=0),
                                      ts * frame.get_intensity(snr=10),
                                      stg.sinc2_f_profile(width=3*frame.df*u.Hz),
                                      stg.constant_bp_profile(level=1))
             
            if bound_type == 'snr':
                l, r, _ = bounds.snr_bounds(frame.integrate())
            else:
                l, r, _ = bounds.threshold_baseline_bounds(frame.integrate())

            n_frame = frame_processing.tnorm(frame, divide_std=divide_std)
            tr_frame = n_frame.get_slice(l, r)
            tr_ts = tr_frame.integrate('f')
            tr_ts /= tr_ts.mean()

            # Just get the stats for the detected signal
            ts_stats = diag_stats.get_stats(tr_ts)

            for key in ts_stats:
                ts_stats_dict[f"{key}"].append(ts_stats[key])

        synth_stats_dicts[t_d] = ts_stats_dict
        tr.stop('synthesize_bbox')
    
    
    
    if pow == 2:
        label = 'sq'
    else:
        label = 'k'
    keys = ['std', 'min', 'ks', f'acf_t_d.{label}.{use_triangle}']
    t_ds = [10, 30, 100]
    titles = ['Standard Deviation', 'Minimum', 'Kolmogorov-Smirnoff Statistic', 'Scintillation Timescale Fit (s)']

    fig, axs = plt.subplots(1, len(keys), figsize=(20, 4), sharey='col')

    for j, key in enumerate(keys):
        key = f"{key}"
        bins=np.histogram(np.hstack([synth_stats_dicts[t_d][key] for t_d in t_ds] + [data_df[key]]), bins=40)[1]
        for i, t_d in enumerate(t_ds):
            axs[j].hist(synth_stats_dicts[t_d][key], bins=bins, histtype='step', label=f'{t_d} s')
            axs[j].set_title(f'{key.upper()}')
            axs[j].xaxis.set_tick_params(labelbottom=True)
    #         axs[j].legend()

        axs[j].hist(data_df[key], bins=bins, histtype='step', color='k', lw=2, label='Non-DC RFI')
        axs[j].set_title(f'{key.upper()}')
        # axs[j].legend(loc=[1, 1, 1, 2][j])
    plt.savefig(f"{plot_fn_prefix}.pdf", bbox_inches='tight')