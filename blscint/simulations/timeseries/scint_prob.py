import numpy as np 
import pandas as pd
import seaborn as sns 
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.patches as mpatches
from astropy.stats import sigma_clip


class BaseSyntheticDistRanker():
    def __init__(self, t_ds, dfs, stats=['std', 'min', 'ks', 'fit_t_d']):
        self.t_ds = t_ds 
        self.dfs = dfs 
        self.stats = stats
        self.d = len(self.stats)
        assert len(self.t_ds) == len(self.dfs)

        self.N = len(self.dfs[0])
        for df in self.dfs:
            assert len(df) == self.N 

        self.ps = None

    def p(self, hit, t_d):
        raise NotImplementedError

    def rank(self, hit):
        """
        Return rank, MLE t_d
        """
        ranks = [] 
        for t_d in self.t_ds:
            ranks.append(self.p(hit, t_d))
        t_d_idx = np.argmax(ranks)
        return ranks[t_d_idx], self.t_ds[t_d_idx]

    def rank_all_synthetic(self):
        """ 
        Iterate over all class data and rank synthetic signals
        """
        for df in self.dfs:
            df[['mle_rank', 'mle_t_d']] = df.apply(lambda row: self.rank(row), 
                                                   axis=1).apply(pd.Series)
    
    @property
    def all_dfs(self):
        return pd.concat(self.dfs, ignore_index=True)

    def plot_ranking_vs_frequency(self, df_events, bins=100):
        gs = grd.GridSpec(1, 2, 
                          # height_ratios=[10, 10, 10, 6, 6], 
                          width_ratios=[12, 2], 
                          wspace=0.02, 
                          # hspace=0.
                          )
        
        data_ranks = df_events.apply(lambda row: self.rank(row)[0], axis=1)
                    
        ax = plt.subplot(gs[0])
        plt.scatter(df_events['Uncorrected_Frequency'], 
                    data_ranks, 
                    marker='.', 
                    s=1, 
                    c='k')

        plt.yscale('log')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Ranking (Arbitrary Units)')
        plt.grid(ls='--')

        ax = plt.subplot(gs[1], sharey=ax)
        ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(axis='x', labelbottom=False)
        log_vals = np.log10(self.all_dfs['mle_rank'])
        log_bins = np.histogram(log_vals[np.isfinite(log_vals)],
                                bins)[1]
        # log_bins = np.histogram(np.log10(self.all_dfs['mle_rank']),
        #                         100)[1]
            
        for i in range(len(self.dfs)):
            plt.hist(self.dfs[i]['mle_rank'], 
                    histtype='step', #density=True, 
                    bins=10**log_bins, 
                    orientation='horizontal',
                    label=f'{self.t_ds[i]} s')
            
        plt.yscale('log')
        plt.ylim(10**(log_bins[0]-3), 10**(log_bins[-1]+1))
        plt.xlabel('Density')
        plt.grid(ls='--', axis='y')
        plt.legend()

    def plot_confusion_matrix(self, title=False):
        import sklearn.metrics

        y_true = np.empty(0)
        y_pred = np.empty(0)
        for i in range(len(self.t_ds)):
            y_true = np.concatenate([
                y_true, 
                np.tile(self.t_ds[i], self.N)
            ])
            y_pred = np.concatenate([
                y_pred, 
                self.dfs[i]['mle_t_d']
            ])
        label_t_ds = [f'{t_d} s' for t_d in self.t_ds]
        temp_df = pd.DataFrame(sklearn.metrics.confusion_matrix(y_true, 
                                                                y_pred, 
                                                                labels=self.t_ds), 
                               label_t_ds, 
                               label_t_ds)
        sns.heatmap(temp_df, annot=True,
                    fmt='d', cmap='Blues') 
        plt.ylabel('True Timescale', fontsize=13)
        if title:
            plt.title('Confusion Matrix', fontsize=17, pad=20)
        plt.gca().xaxis.set_label_position('top') 
        plt.xlabel('Predicted Timescale', fontsize=13)
        plt.gca().xaxis.tick_top()

    def plot_diagstat_corner_plot(self, bw_adjust=1):
        g = sns.PairGrid(self.all_dfs[['t_d'] + self.stats].sort_values(by=['t_d'], ascending=False),
                        hue='t_d',
                        # hue_order=[100, 30, 10],
                        palette=sns.color_palette(),
                        diag_sharey=False,
                        despine=False, 
                        corner=True)

        def kdeplot_by_t_d(t_d):
            def plot_func(x, y, **kwargs):
                if kwargs['label'] == t_d:
                    sns.kdeplot(x=x, y=y, **kwargs) 
            return plot_func

        for t_d in [100, 30, 10]:
            g.map_lower(kdeplot_by_t_d(t_d), levels=5, fill=False, kde_kws={'bw_method': bw_adjust})
            g.map_lower(kdeplot_by_t_d(t_d), levels=5, fill=True, alpha=0.8, kde_kws={'bw_method': bw_adjust})
        g.map_diag(sns.histplot, stat='density', element='step', bins=30)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        def get_handle(color):
            return mpatches.Patch(color=color)

        plt.legend(handles = [get_handle(colors[i]) for i in range(3)],
                labels=[rf'$\Delta t_d = {t_d}$ s' for t_d in self.t_ds],
                bbox_to_anchor=(0.9, 2))

        g.axes[0, 0].set_ylabel('Standard Deviation')
        g.axes[1, 0].set_ylabel('Minimum')
        g.axes[2, 0].set_ylabel('Kolmogorov-Smirnoff Statistic')
        g.axes[3, 0].set_ylabel('Scintillation Timescale Fit (s)')
        g.axes[3, 0].set_xlabel('Standard Deviation')
        g.axes[3, 1].set_xlabel('Minimum')
        g.axes[3, 2].set_xlabel('Kolmogorov-Smirnoff Statistic')
        g.axes[3, 3].set_xlabel('Scintillation Timescale Fit (s)')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
    def plot_corner_plot(self):
        g = sns.PairGrid(self.all_dfs[['t_d'] + self.stats],
                         hue='t_d',
                         # hue_order=[100, 30, 10],
                         palette=sns.color_palette(),
                         diag_sharey=False,
                         corner=True)
        # g.map_upper(sns.scatterplot)
        g.map_lower(sns.kdeplot, fill=False, kde_kws={'bw_adjust': 0.25})
        g.map_diag(sns.histplot, kde=True, stat='density', kde_kws={'bw_adjust': 1})
        g.add_legend()

    def plot_synthetic_ranks(self):
        try:
            sns.histplot(self.all_dfs, x='mle_rank', hue='t_d', palette=sns.color_palette(),
                        log_scale=True)
        except ValueError:
            log_vals = np.log10(self.all_dfs['mle_rank'])
            log_bins = np.histogram(log_vals[np.isfinite(log_vals)],
                                    100)[1]
            sns.histplot(self.all_dfs, x='mle_rank', hue='t_d', palette=sns.color_palette(),
                        log_scale=True, bins=log_bins)


class HistogramRanker(BaseSyntheticDistRanker):
    def __init__(self, t_ds, dfs, 
                 stats=['std', 'min', 'ks', 'fit_t_d'],
                 bins=[np.linspace(0, 2, 50), 
                       np.linspace(-2, 1, 50), 
                       np.linspace(0, 0.7, 50), 
                       np.linspace(0, 150, 50)]):
        super().__init__(t_ds, dfs, stats)
        self.bins = bins
        self.ps = {t_d: {stat: self._fit_p(t_d, stat) 
                         for stat in self.stats} 
                   for t_d in self.t_ds} 
        self.rank_all_synthetic()

    def _fit_p(self, t_d, stat):
        return np.histogram(self.dfs[self.t_ds.index(t_d)][stat],
                            bins=self.bins[self.stats.index(stat)],
                            density=True)

    def p_stat(self, hit_stat, t_d, stat):
        idx = np.digitize(hit_stat, self.ps[t_d][stat][1])
        if idx == 0 or idx == len(self.ps[t_d][stat][1]):
            return 0
        else:
            return self.ps[t_d][stat][0][idx - 1]

    def p(self, hit, t_d):
        val = 1 
        for stat in self.stats:
            val *= self.p_stat(hit[stat], t_d, stat)
        return val

    def plot_stat_hist(self, statistic, **kwargs):
        sns.histplot(data=self.all_dfs, 
                     x=statistic, 
                     hue='t_d', 
                     palette=sns.color_palette(), **kwargs);


class KDERanker(BaseSyntheticDistRanker):
    def __init__(self, t_ds, dfs, 
                 stats=['std', 'min', 'ks', 'fit_t_d'],
                 factors=np.array([1, 1, 1, 1]),
                 bw_adjust=1):
        super().__init__(t_ds, dfs, stats)
        scotts_factor = self.N**(-1./(self.d + 4))
        self.factors = factors
        self.ps = {t_d: self._fit_p(t_d, bw_adjust*scotts_factor) 
                   for t_d in self.t_ds} 
        # self.ps = {t_d: self._fit_p(t_d, None) 
        #            for t_d in self.t_ds} 
        self.rank_all_synthetic()

    def _fit_p(self, t_d, bw_method=None):
        X = self.dfs[self.t_ds.index(t_d)][self.stats].to_numpy(np.float64).T
        return scipy.stats.gaussian_kde(X / self.factors.T[:,np.newaxis], bw_method=bw_method)

    def p(self, hit, t_d):
        value_vec = hit[self.stats].to_numpy(np.float64).reshape((4, -1))
        if np.any(np.isnan(value_vec)):
            return 0
        return self.ps[t_d](value_vec / self.factors.T[:,np.newaxis])[0]
