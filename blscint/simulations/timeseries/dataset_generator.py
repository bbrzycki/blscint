from pathlib import Path
import click 
import collections
import numpy as np
import pandas as pd
import tqdm
import setigen as stg
import multiprocessing

from blscint import simulations
from blscint import frame_processing
from blscint import diag_stats


def _single_trial(generator, 
                    t_d=10, 
                    snr=25,
                    bw=2,
                    injected=False,
                    bound='threshold',
                    gen_method='arta',
                    pow=5/3, 
                    divide_std=True,
                    file_stem=None, 
                    save_ts=False,
                    p=None):    
    if gen_method == 'arta':
        if p is None:
            p = generator.frame_metadata['tchans'] // 4
        ts = simulations.get_ts_arta(t_d, 
                                        generator.frame_metadata['dt'],
                                        generator.frame_metadata['tchans'],
                                        p=p,
                                        pow=pow,
                                        seed=generator.rng)
    elif gen_method == 'fft':
        ts = simulations.get_ts_fft(t_d,
                                    generator.frame_metadata['dt'],
                                    generator.frame_metadata['tchans'],
                                    pow=pow,
                                    seed=generator.rng)
    else:
        raise ValueError("Generation method must be either 'arta' or 'fft'")
    l = r = fchans = None

    if injected:
        frame = stg.Frame(fchans=256,
                            tchans=generator.frame_metadata['tchans'],
                            df=generator.frame_metadata['df'],
                            dt=generator.frame_metadata['dt'],
                            seed=generator.rng)
        frame.add_noise_from_obs()
        signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(128), 
                                                    drift_rate=0),
                                    ts * frame.get_intensity(snr=snr),
                                    stg.sinc2_f_profile(width=bw*frame.df, 
                                                        width_mode="crossing"),
                                    stg.constant_bp_profile(level=1))

        try:
            ts, (l, r) = frame_processing.extract_ts(frame,
                                                        bound=bound,
                                                        divide_std=divide_std,
                                                        as_data=frame.get_slice(0, frame.fchans//2-bw//2))
        except IndexError:
            # Signal not bound by bounding algorithm 
            ts = None 

        fchans = frame.fchans
        
    if ts is not None:
        ts_stats = diag_stats.get_diag_stats(ts)
        ts_stats.update({
            'fchans': fchans,
            't_d': t_d,
            'l': l,
            'r': r,
            'SNR': snr,
            'DriftRate': 0,
        })
    else:
        ts_stats = None
    if not save_ts:
        ts = None
    return ts_stats, ts
    



class SignalGenerator(object):
    """
    Class to synthesize scintillated signals for use in thresholding.
    """
    def __init__(self, dt, df, tchans, seed=None, **kwargs):
        self.rng = np.random.default_rng(seed)

        self.frame_metadata = {
            'dt': dt,
            'df': df,
            'tchans': tchans
        }
        self.df = None
    
    def _single_trial(self, 
                     t_d, 
                     snr=25,
                     bw=2,
                     injected=False,
                     bound='threshold',
                     gen_method='arta',
                     pow=5/3, 
                     divide_std=True,
                     file_stem=None, 
                     save_ts=False,
                     p=None):
        if gen_method == 'arta':
            if p is None:
                p = self.frame_metadata['tchans'] // 4
            ts = simulations.get_ts_arta(t_d, 
                                            self.frame_metadata['dt'],
                                            self.frame_metadata['tchans'],
                                            p=p,
                                            pow=pow,
                                            seed=self.rng)
        elif gen_method == 'fft':
            ts = simulations.get_ts_fft(t_d,
                                        self.frame_metadata['dt'],
                                        self.frame_metadata['tchans'],
                                        pow=pow,
                                        seed=self.rng)
        else:
            raise ValueError("Generation method must be either 'arta' or 'fft'")
        l = r = fchans = None

        if injected:
            frame = stg.Frame(fchans=256,
                                tchans=self.frame_metadata['tchans'],
                                df=self.frame_metadata['df'],
                                dt=self.frame_metadata['dt'],
                                seed=self.rng)
            frame.add_noise_from_obs()
            signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(128), 
                                                        drift_rate=0),
                                        ts * frame.get_intensity(snr=snr),
                                        stg.sinc2_f_profile(width=bw*frame.df, 
                                                            width_mode="crossing"),
                                        stg.constant_bp_profile(level=1))

            try:
                ts, (l, r) = frame_processing.extract_ts(frame,
                                                            bound=bound,
                                                            divide_std=divide_std,
                                                            as_data=frame.get_slice(0, frame.fchans//2-bw//2))
            except IndexError:
                # Signal not bound by bounding algorithm 
                ts = None 
        
            fchans = frame.fchans
            
        if ts is not None:
            ts_stats = diag_stats.get_diag_stats(ts)
            ts_stats.update({
                'fchans': fchans,
                't_d': t_d,
                'l': l,
                'r': r,
                'SNR': snr,
                'DriftRate': 0,
            })
        else:
            ts_stats = None
        if not save_ts:
            ts = None
        return ts_stats, ts

    def make_dataset(self, 
                     t_d, 
                     n=1000, 
                     snr=25,
                     bw=2,
                     injected=False,
                     bound='threshold',
                     gen_method='arta',
                     pow=5/3, 
                     divide_std=True,
                     file_stem=None, 
                     save_ts=False,
                     p=None,
                     processes=1):
        """
        Create dataset of synthetic scintillated signals, and save
        statistic details to csv. 

        gen_method is either 'arta' or 'fft'.
        """
        if file_stem is not None:
            stem_path = Path(file_stem)
            csv_path = stem_path.parent / f"{stem_path.name}.diagstat.csv"
            tsdump_path = stem_path.parent / f"{stem_path.name}.tsdump.npy"
        
        stats_df = pd.DataFrame()
        ts_stats_dict = collections.defaultdict(list)
        if save_ts:
            tsdump = np.full((n, self.frame_metadata['tchans']), 
                             np.nan)
        if processes == 1:
            for idx in tqdm.trange(n):
                ts_stats, ts = self._single_trial(t_d, snr, bw, injected, bound, gen_method,
                                                pow, divide_std, file_stem, save_ts, p)
                if save_ts and ts is not None:
                    tsdump[idx, :] = ts.array()

                if ts_stats is not None:
                    for stat in ts_stats:
                        ts_stats_dict[stat].append(ts_stats[stat])
        else:
            from itertools import repeat
            with multiprocessing.Pool(processes=processes) as pool:
                ts_stats_list = pool.starmap(_single_trial, zip(repeat(self), repeat(t_d), repeat(snr),
                 repeat(bw), repeat(injected), repeat(bound), repeat(gen_method),
                 repeat(pow), repeat(divide_std), repeat(file_stem), repeat(save_ts), repeat(p)))
                
                # np.tile(np.array([self, t_d, snr, bw, injected, bound, gen_method,
                #                          pow, divide_std, file_stem, save_ts, p]).reshape([-1, 1]), n))
            
            for idx in tqdm.trange(n):
                ts_stats, ts = ts_stats_list[idx]
                if save_ts and ts is not None:
                    tsdump[idx, :] = ts.array()

                if ts_stats is not None:
                    for stat in ts_stats:
                        ts_stats_dict[stat].append(ts_stats[stat])
                

        # Set statistic columns
        for stat in ts_stats_dict:
            stats_df[stat] = ts_stats_dict[stat]

        stats_df['real'] = False

        if self.df is None:
            self.df = stats_df 
        else:
            self.df = pd.concat([self.df, stats_df], ignore_index=True)

        if file_stem is not None:
            # Save time series intensities for all signals if option enabled
            if save_ts:
                stats_df['tsdump_fn'] = tsdump_path.resolve()
                np.save(tsdump_path, tsdump)

            stats_df.to_csv(csv_path, index=False)


@click.command(name='synthesize',
               short_help='Make datasets of sythetic scintillated signals',
               no_args_is_help=True,)
# @click.argument('filename')
@click.option('-d', '--save-dir', 
              help='Directory to save output files')
@click.option('-t', '--tscint', multiple=True, type=float,
              help='Scintillation timescales to synthesize')
@click.option('-n', '--sample-number', type=int, default=1000, show_default=True,
              help='Number of samples per scintillation timescale')
@click.option('--dt', type=float,
              help='Time resolution of observations')
@click.option('--tchans', type=int,
              help='Number of time channels in each observation')
@click.option('--df', type=float, 
              help='Frequency resolution of observations')
@click.option('--rfi-csv', 
              help='Diagstat csv for RFI observations')
@click.option('--data-csv', 
              help='Diagstat csv for data observations')
@click.option('-i', '--injected', is_flag=True,
              help='Whether to inject synthetic signals and extract noisy intensity time series')
@click.option('--store-idp', is_flag=True,
              help='Store intermediate data products')
def synthesize_dataset(save_dir, tscint, sample_number, dt, tchans, df,
                       rfi_csv, data_csv, injected, store_idp):
    """
    Make datasets of sythetic scintillated signals for use in thresholding
    """
    generator = SignalGenerator(dt=dt, df=df, tchans=tchans)
    for t_d in tscint:
        generator.make_dataset(t_d=t_d,
                               n=sample_number,
                               injected=injected,
                               file_stem=Path(save_dir) / f"synthetic_{t_d}s")