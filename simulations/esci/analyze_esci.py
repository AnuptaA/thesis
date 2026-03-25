#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, json, re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi':      150,
    'font.size':       13,
    'axes.titlesize':  15,
    'axes.labelsize':  14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

#-------------------------------------------------------------------------------

ALGO_ORDER = [
    'lemma1', 'lemma1_no_union',
    'lemma2', 'lemma2_no_union',
    'combined', 'combined_no_union',
]

ALGO_LABELS = {
    'lemma1':            'CIG',
    'lemma1_no_union':   'CIG (no union)',
    'lemma2':            'HGG',
    'lemma2_no_union':   'HGG (no union)',
    'combined':          'Combined',
    'combined_no_union': 'Combined (no union)',
}

NO_UNION_ALGOS: Set[str] = {'lemma1_no_union', 'lemma2_no_union', 'combined_no_union'}

#-------------------------------------------------------------------------------
# benchmark name parsing

_BENCH_RE = re.compile(r'esci_c(\d+)k(\d+)_t(\d+)n(\d+)_s(\d+)')

def _parse_name(name: str) -> Dict:
    """Parse ESCI benchmark name into component integers."""
    m = _BENCH_RE.search(name)
    if not m:
        return {}
    return dict(
        num_cache=int(m.group(1)), cache_K=int(m.group(2)),
        num_test=int(m.group(3)),  test_N=int(m.group(4)),
        seed=int(m.group(5)),
    )

def _config_key(name: str) -> str:
    """Strip seed suffix to get a config-level key for seed aggregation."""
    return re.sub(r'_s\d+$', '', name)

def _classify_set(num_test: int) -> str:
    """Classify a benchmark into set2 (cache scaling) or set3 (K/N ratios)."""
    return 'set3' if num_test == 256 else 'set2'

#-------------------------------------------------------------------------------
# style / ordering helpers

def ordered_algos(algos, include_no_union: bool = False) -> List[str]:
    algos = set(algos)
    if not include_no_union:
        algos -= NO_UNION_ALGOS
    return [a for a in ALGO_ORDER if a in algos] + sorted(algos - set(ALGO_ORDER))

def ordered_metrics(metrics) -> List[str]:
    preferred = ['euclidean', 'angular', 'cosine']
    metrics = set(metrics)
    return [m for m in preferred if m in metrics] + sorted(metrics - set(preferred))

def algo_palette(algos) -> Dict[str, tuple]:
    ordered = ordered_algos(algos, include_no_union=True)
    colors = sns.color_palette('tab10', len(ordered))
    return {a: c for a, c in zip(ordered, colors)}

def lbl(algo: str) -> str:
    return ALGO_LABELS.get(algo, algo)

def lbls(algos) -> List[str]:
    return [lbl(a) for a in algos]

def cap_metric(m: str) -> str:
    return m.title()

def cap_metrics(metrics) -> List[str]:
    return [cap_metric(m) for m in metrics]

def save_fig(fig: plt.Figure, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")

def pct_fmt(ax, axis='y'):
    formatter = mticker.FuncFormatter(lambda v, _: f'{v:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

def _set_cache_size_xaxis(ax, cache_sizes):
    """Log2 x-axis for cache size plots - ticks show log2 exponents."""
    ax.set_xscale('log', base=2)
    ax.set_xticks(cache_sizes)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(np.log2(x)))))
    ax.tick_params(axis='x', labelrotation=0)

def _add_footnote(fig: plt.Figure, text: str):
    fig.text(0.01, -0.01, text, fontsize=7.5, color='#555555',
             va='top', ha='left', transform=fig.transFigure)

def _left_cbar_fig(n_rows, n_cols, cmap, vmin, vmax, cbar_label,
                   fig_w, fig_h, wspace=0.05):
    fig = plt.figure(figsize=(fig_w + 1.2, fig_h))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[0.05] + [1] * n_cols,
                  hspace=0.4, wspace=wspace)
    cbar_ax = fig.add_subplot(gs[:, 0])
    sm = mpl.cm.ScalarMappable(cmap=cmap,
                                norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.tick_left()
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)]
            for r in range(n_rows)]
    for row in axes:
        for ax in row:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
    return fig, axes

#-------------------------------------------------------------------------------
# data loading

def load_summaries(raw_dir: str) -> List[Dict]:
    raw_path = Path(raw_dir)
    summaries = []
    for p in sorted(raw_path.glob('*/summary.json')):
        with open(p) as f:
            summaries.append(json.load(f))
    print(f"Loaded {len(summaries)} benchmark summaries")
    return summaries

def build_summary_df(summaries: List[Dict]) -> pd.DataFrame:
    rows = []
    for ds in summaries:
        name = ds['dataset']
        cfg = ds['config']
        pars = _parse_name(name)
        num_test = pars.get('num_test', cfg.get('num_test_queries', 0))
        for res in ds['results']:
            rows.append({
                'dataset':            name,
                'config_key':         _config_key(name),
                'set_id':             _classify_set(num_test),
                'seed':               pars.get('seed'),
                'algorithm':          res['algorithm'],
                'metric':             res['metric'],
                'hit_rate':           res['hit_rate'],
                'total_queries':      res['total_queries'],
                'cache_hits':         res['cache_hits'],
                'lemma1_hits':        res.get('lemma1_hits', 0),
                'lemma2_hits':        res.get('lemma2_hits', 0),
                'avg_distance_calcs': res.get('avg_distance_calcs'),
                'avg_time_us':        res.get('avg_time_us'),
                'cache_K':            cfg.get('cache_K'),
                'test_N':             cfg.get('test_N'),
                'num_cache_queries':  cfg.get('num_cache_queries'),
                'num_base_vectors':   cfg.get('num_base_vectors'),
                'dimension':          cfg.get('dimension'),
            })
    return pd.DataFrame(rows)

def load_per_query_df(raw_dir: str, summaries: List[Dict]) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    cfg_lookup = {ds['dataset']: ds['config'] for ds in summaries}
    dfs = []
    for p in sorted(raw_path.glob('*/per_query.csv')):
        name = p.parent.name
        try:
            df   = pd.read_csv(p)
            cfg  = cfg_lookup.get(name, {})
            pars = _parse_name(name)
            num_test = pars.get('num_test', cfg.get('num_test_queries', 0))
            df['dataset']           = name
            df['config_key']        = _config_key(name)
            df['set_id']            = _classify_set(num_test)
            df['num_cache_queries'] = cfg.get('num_cache_queries')
            df['cache_K']           = cfg.get('cache_K')
            df['test_N']            = cfg.get('test_N')
            dfs.append(df)
        except Exception as e:
            print(f"  Could not load {p}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

#-------------------------------------------------------------------------------
# aggregation helper

def _get_agg(a_sub: pd.DataFrame, value_col: str,
             group_vals: list, group_col: str):
    """Average value_col per config_key (seed isolation), then mean ± std across seeds.

    Returns (ys, errs) aligned to group_vals; NaN for missing values, 0 for missing std.
    """
    a_sub = a_sub.copy()
    per_seed = a_sub.groupby([group_col, 'config_key'])[value_col].mean().reset_index()
    agg = per_seed.groupby(group_col)[value_col].agg(['mean', 'std'])
    ys = [float(agg.loc[v, 'mean']) if v in agg.index else np.nan for v in group_vals]
    errs = [float(agg.loc[v, 'std'])  if v in agg.index else 0.0    for v in group_vals]
    errs = [0.0 if np.isnan(e) else e for e in errs]
    return ys, errs

#-------------------------------------------------------------------------------
# coverage computation

def compute_esci_coverage(
    dataset_name: str,
    K: int,
    num_cache_queries: int,
    num_base_vectors: int,
    datasets_base_dir: Path,
) -> Optional[float]:
    npz = datasets_base_dir / dataset_name / f'cache_gt_K{K}.npz'
    if not npz.exists():
        print(f'  [coverage] missing: {npz}')
        return None
    data = np.load(npz)
    all_idx = np.concatenate([data[f'indices_{i}'] for i in range(num_cache_queries)])
    return float(len(np.unique(all_idx)) / num_base_vectors)


def compute_set2_coverage_df(sub: pd.DataFrame, datasets_base_dir: Path) -> pd.DataFrame:
    unique = (
        sub[['dataset', 'num_cache_queries', 'cache_K', 'num_base_vectors']]
        .drop_duplicates()
        .copy()
    )
    rows = []
    for _, row in unique.iterrows():
        name = row['dataset']
        C = int(row['num_cache_queries'])
        K = int(row['cache_K'])
        B = int(row['num_base_vectors'])
        ac = compute_esci_coverage(name, K, C, B, datasets_base_dir)
        ec = 1.0 - (1.0 - K / B) ** C
        rows.append({'dataset': name, 'cache_size': C,
                     'coverage_actual': ac, 'coverage_estimated': ec})
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------------
# coverage plots

def plot_set2_coverage_and_hit_rate(
    sub: pd.DataFrame,
    cov_df: pd.DataFrame,
    outdir: Path,
):
    algos = ordered_algos(
        (a for a in sub['algorithm'].unique()),
        include_no_union=False,
    )
    cache_sizes = sorted(cov_df['cache_size'].unique().astype(int))
    pal = algo_palette(algos)

    cov_agg = (
        cov_df.dropna(subset=['coverage_actual'])
        .groupby('cache_size')['coverage_actual']
        .mean()
        .reindex(cache_sizes)
    )
    cov_mean = cov_agg.values
    cs_arr = np.array(cache_sizes, dtype=float)

    n_rows = len(algos)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]

    cov_handle = None
    for row_idx, algo in enumerate(algos):
        ax = axes[row_idx]
        h_cov, = ax.plot(cs_arr, cov_mean, color='steelblue', linewidth=2.5,
                         label='Coverage (actual)')
        if cov_handle is None:
            cov_handle = h_cov
        a_sub = sub[sub['algorithm'] == algo].copy()
        if not a_sub.empty:
            ys, _ = _get_agg(a_sub, 'hit_rate', cache_sizes, 'num_cache_queries')
            ax.plot(cs_arr, ys, color=pal.get(algo), linewidth=1.8, marker='o', markersize=4)
        pct_fmt(ax)
        ax.set_ylim(0, 1.05)
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_ylabel(lbl(algo))
        if row_idx == n_rows - 1:
            ax.set_xlabel('Cache Size (log2)')

    if cov_handle:
        fig.legend([cov_handle], ['Coverage (actual)'],
                   loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle('Set 2 - Coverage and Hit Rate vs Cache Size (K=99, N=20, B=1,820,000)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set2_coverage_and_hit_rate.png')


def plot_set2_coverage_estimated_vs_actual(cov_df: pd.DataFrame, outdir: Path):
    cache_sizes = sorted(cov_df['cache_size'].unique().astype(int))
    cs_arr = np.array(cache_sizes, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    cov_agg = (
        cov_df.dropna(subset=['coverage_actual'])
        .groupby('cache_size')['coverage_actual']
        .mean()
        .reindex(cache_sizes)
    )
    h_mean, = ax.plot(cs_arr, cov_agg.values, color='steelblue', linewidth=2.5,
                      label='Actual coverage (mean)')
    est = cov_df.groupby('cache_size')['coverage_estimated'].first().reindex(cache_sizes).values
    h_est, = ax.plot(cs_arr, est, color='darkorange', linewidth=2,
                     linestyle='--', label='Estimated: 1-(1-K/B)^C')
    pct_fmt(ax)
    ax.set_ylim(0, 1.05)
    _set_cache_size_xaxis(ax, cache_sizes)
    ax.set_xlabel('Cache Size (log2)')
    ax.set_ylabel('Database Coverage')
    ax.legend([h_mean, h_est], ['Actual coverage (mean)', 'Estimated: 1-(1-K/B)^C'], fontsize=9)
    fig.suptitle('Set 2 - Estimated vs Actual Coverage (K=99, B=1,820,000)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set2_coverage_estimated_vs_actual.png')

#-------------------------------------------------------------------------------
# global plots

def plot_global_hit_rate_heatmap(df: pd.DataFrame, outdir: Path):
    """Hit rate heatmap averaged across all benchmarks: algo (rows) × metric (cols)."""
    sub = df[~df['algorithm'].isin(NO_UNION_ALGOS)]
    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())

    mat = np.full((len(algos), len(metrics)), np.nan)
    for i, a in enumerate(algos):
        for j, m in enumerate(metrics):
            vals = sub[(sub['algorithm'] == a) & (sub['metric'] == m)]['hit_rate']
            if len(vals):
                mat[i, j] = vals.mean()

    fig_w = max(5, len(metrics) * 2.5)
    fig_h = max(4, len(algos) * 0.9)
    fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
    ax = ax_grid[0][0]
    sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=cap_metrics(metrics), yticklabels=lbls(algos),
                ax=ax, cbar=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Hit Rate by Algorithm — All Benchmarks', pad=12)
    save_fig(fig, outdir, 'hit_rate_heatmap_global.png')

#-------------------------------------------------------------------------------

def plot_lemma_breakdown_global(df: pd.DataFrame, outdir: Path):
    """Stacked bar: CIG / HGG share within Combined algorithm hits, grouped by set.

    When combined cache hits are zero for a group the bars show 0% with a footnote.
    """
    comb = df[df['algorithm'] == 'combined'].copy()
    if comb.empty:
        print("  No combined algorithm data, skipping lemma breakdown.")
        return

    comb['l1_share'] = np.where(comb['cache_hits'] > 0,
                                comb['lemma1_hits'] / comb['cache_hits'], 0.0)
    comb['l2_share'] = np.where(comb['cache_hits'] > 0,
                                comb['lemma2_hits'] / comb['cache_hits'], 0.0)
    has_zero_hits = (comb['cache_hits'] == 0).any()

    metrics = ordered_metrics(comb['metric'].unique())
    set_ids = sorted(comb['set_id'].unique())
    pal = sns.color_palette('tab10', 2)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(5, len(set_ids) * 1.8 + 2) * len(metrics), 5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = comb[comb['metric'] == metric]
        agg = sub.groupby('set_id')[['l1_share', 'l2_share']].mean()
        agg = agg.reindex([s for s in set_ids if s in agg.index])
        x = np.arange(len(agg))
        ax.bar(x, agg['l1_share'], label='CIG', color=pal[0])
        ax.bar(x, agg['l2_share'], bottom=agg['l1_share'], label='HGG', color=pal[1])
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in agg.index], rotation=0)
        ax.set_title(cap_metric(metric))
        pct_fmt(ax)

    axes[0].set_ylabel('Share of Combined Hits')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pal[:2]]
    axes[-1].legend(handles, ['CIG', 'HGG'],
                    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    if has_zero_hits:
        _add_footnote(fig, '* Bars at 0% indicate zero combined cache hits for that group.')
    fig.suptitle('CIG vs HGG Contribution Within Combined Algorithm (by Set)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'lemma_breakdown_global.png')

#-------------------------------------------------------------------------------

def plot_avg_time(df: pd.DataFrame, outdir: Path):
    """Log-scale bar chart of average query time per algorithm, grouped by set."""
    sub = df[~df['algorithm'].isin(NO_UNION_ALGOS)].copy()
    if sub.empty or sub['avg_time_us'].isna().all():
        print("  No timing data, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    set_ids = sorted(sub['set_id'].unique())
    pal = algo_palette(algos)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(6, len(set_ids) * 2.5 + 2) * len(metrics), 5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        n_grps = len(algos)
        width = 0.8 / n_grps
        x = np.arange(len(set_ids))
        for i, algo in enumerate(algos):
            ys, errs = _get_agg(m_sub[m_sub['algorithm'] == algo],
                                'avg_time_us', set_ids, 'set_id')
            ax.bar(x + (i - n_grps / 2 + 0.5) * width, ys, width,
                   label=lbl(algo), color=pal.get(algo),
                   yerr=errs, capsize=3, error_kw=dict(lw=1))
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in set_ids])
        ax.set_title(cap_metric(metric))

    axes[0].set_ylabel('Avg Query Time (\u03bcs, log scale)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    _add_footnote(fig, 'Error bars: \u00b11 std across seeds.')
    fig.suptitle('Average Query Time by Algorithm and Set', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'avg_time_by_algorithm.png')

#-------------------------------------------------------------------------------
# Set 1: Baseline (c=1024, K=99/100, N=20, 3 seeds)
# Subset of Set 2 filtered to num_cache=1024
#-------------------------------------------------------------------------------

def plot_set1_baseline(df: pd.DataFrame, outdir: Path):
    """Grouped bar: hit rate and distance calcs at baseline config, ±1σ across 3 seeds."""
    sub = df[
        (df['set_id'] == 'set2') &
        (df['num_cache_queries'] == 1024) &
        (~df['algorithm'].isin(NO_UNION_ALGOS))
    ]
    if sub.empty:
        print("  No Set 1 (baseline) data, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)

    k_vals = sorted(sub['cache_K'].dropna().unique().astype(int))
    k_str = '/'.join(str(k) for k in k_vals)
    n_vals = sorted(sub['test_N'].dropna().unique().astype(int))
    n_str = '/'.join(str(n) for n in n_vals)

    for value_col, ylabel, yscale, fname in [
        ('hit_rate', 'Hit Rate', 'linear', 'baseline_hit_rate.png'),
        ('avg_distance_calcs', 'Avg Distance Calcs/Query (log)', 'log', 'baseline_distance_calcs.png'),
    ]:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5),
                                 sharey=True)
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            m_sub = sub[sub['metric'] == metric]
            per_seed = m_sub.groupby(['algorithm', 'config_key'])[value_col].mean().reset_index()
            stats = per_seed.groupby('algorithm')[value_col].agg(['mean', 'std']).reindex(algos)
            for i, algo in enumerate(algos):
                if algo not in stats.index or pd.isna(stats.loc[algo, 'mean']):
                    continue
                mean = stats.loc[algo, 'mean']
                std = 0.0 if pd.isna(stats.loc[algo, 'std']) else stats.loc[algo, 'std']
                ax.bar(i, mean, 0.6, color=pal[algo],
                       yerr=std, capsize=4, error_kw=dict(lw=1.2))
            ax.set_xticks(range(len(algos)))
            ax.set_xticklabels(lbls(algos), rotation=30, ha='right')
            ax.set_yscale(yscale)
            if yscale == 'linear':
                pct_fmt(ax)
                ax.set_ylim(0, 1.05)
            ax.set_title(cap_metric(metric))

        axes[0].set_ylabel(ylabel)
        _add_footnote(fig, 'Error bars: \u00b11 std across 3 seeds.')
        fig.suptitle(
            f'Set 1 Baseline \u2014 {ylabel} (K={k_str}, N={n_str}, Cache=1024)',
            y=1.02,
        )
        plt.tight_layout()
        save_fig(fig, outdir, fname)

#-------------------------------------------------------------------------------
# Set 2: Cache size scaling (K=99/100, N=20, 4 sizes × 3 seeds)
#-------------------------------------------------------------------------------

def plot_set2_cache_size(df: pd.DataFrame, outdir: Path, datasets_base_dir: Path = None):
    """Line plots (hit rate, distance calcs) and heatmap vs cache size, seeds aggregated."""
    sub = df[(df['set_id'] == 'set2') & (~df['algorithm'].isin(NO_UNION_ALGOS))]
    if sub.empty:
        print("  No Set 2 data, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    cache_sizes = sorted(sub['num_cache_queries'].dropna().unique().astype(int))
    pal = algo_palette(algos)

    if len(cache_sizes) < 2:
        print("  Only one cache size in Set 2, skipping line plots.")
        return

    k_vals = sorted(sub['cache_K'].dropna().unique().astype(int))
    k_str = '/'.join(str(k) for k in k_vals)
    n_vals = sorted(sub['test_N'].dropna().unique().astype(int))
    n_str = '/'.join(str(n) for n in n_vals)

    # --- hit rate vs cache size ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in algos:
            ys, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                             'hit_rate', cache_sizes, 'num_cache_queries')
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal[algo], linestyle=ls)
        pct_fmt(ax)
        ax.set_ylim(0, 1.05)
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_xlabel('Cache Size (log2)')
        ax.set_title(cap_metric(metric))
    axes[0].set_ylabel('Hit Rate')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle(f'Set 2 - Hit Rate vs Cache Size (K={k_str}, N={n_str})', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'hit_rate_vs_cache_size.png')

    # --- distance calcs vs cache size ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in algos:
            ys, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                             'avg_distance_calcs', cache_sizes, 'num_cache_queries')
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal[algo], linestyle=ls)
        ax.set_yscale('log')
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_xlabel('Cache Size (log2)')
        ax.set_title(cap_metric(metric))
    axes[0].set_ylabel('Avg Distance Calcs/Query (log scale)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle(
        f'Set 2 - Avg Distance Calcs vs Cache Size (K={k_str}, N={n_str})', y=1.02
    )
    plt.tight_layout()
    save_fig(fig, outdir, 'distance_calcs_vs_cache_size.png')

    # --- coverage plots ---
    if datasets_base_dir is not None:
        cov_sub = sub.copy()
        cov_df = compute_set2_coverage_df(cov_sub, datasets_base_dir)
        if not cov_df.empty:
            plot_set2_coverage_and_hit_rate(cov_sub, cov_df, outdir)
            plot_set2_coverage_estimated_vs_actual(cov_df, outdir)

    # --- heatmap: algo × cache size, one per metric ---
    for metric in metrics:
        m_sub = sub[sub['metric'] == metric]
        algos_m = ordered_algos(m_sub['algorithm'].unique())
        mat = np.full((len(algos_m), len(cache_sizes)), np.nan)
        for i, algo in enumerate(algos_m):
            ys, _ = _get_agg(m_sub[m_sub['algorithm'] == algo],
                             'hit_rate', cache_sizes, 'num_cache_queries')
            for j, y in enumerate(ys):
                mat[i, j] = y

        fig_w = max(7, len(cache_sizes) * 1.5)
        fig_h = max(4, len(algos_m) * 0.9)
        fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
        ax = ax_grid[0][0]
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=[int(c) for c in cache_sizes],
                    yticklabels=lbls(algos_m), ax=ax, cbar=False)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('Cache Size (queries)')
        ax.set_title(
            f'Set 2 \u2014 Hit Rate: Algorithm \u00d7 Cache Size '
            f'({cap_metric(metric)}, K={k_str}, N={n_str})',
            pad=12,
        )
        save_fig(fig, outdir, f'hit_rate_heatmap_{metric}.png')

#-------------------------------------------------------------------------------
# Set 3: K/N ratios (num_cache=1024, num_test=256, 2 seeds)
#-------------------------------------------------------------------------------

def plot_set3_kn(df: pd.DataFrame, outdir: Path):
    """K×N hit-rate heatmap (Combined only) and K/N ratio matrix (all algorithms)."""
    sub = df[df['set_id'] == 'set3']
    if sub.empty:
        print("  No Set 3 data, skipping.")
        return

    metrics = ordered_metrics(sub['metric'].unique())
    non_union_algos = ordered_algos(sub['algorithm'].unique(), include_no_union=False)

    def _fmt_ratio(r):
        return str(int(r)) if r == int(r) else f'{r:.2f}'.rstrip('0').rstrip('.')

    for metric in metrics:
        m_sub = sub[sub['metric'] == metric]

        # --- K × N heatmap (Combined only, seeds averaged) ---
        comb = m_sub[m_sub['algorithm'] == 'combined']
        # average per config_key first (seed isolation), then across seeds
        agg = (comb.groupby(['cache_K', 'test_N', 'config_key'])['hit_rate']
                   .mean()
                   .reset_index()
                   .groupby(['cache_K', 'test_N'])['hit_rate']
                   .mean()
                   .reset_index())
        K_vals = sorted(agg['cache_K'].dropna().unique().astype(int))
        N_vals = sorted(agg['test_N'].dropna().unique().astype(int))

        if K_vals and N_vals:
            mat = np.full((len(K_vals), len(N_vals)), np.nan)
            for _, row in agg.iterrows():
                i = K_vals.index(int(row['cache_K']))
                j = N_vals.index(int(row['test_N']))
                mat[i, j] = row['hit_rate']

            fig_w = max(7, len(N_vals) * 1.5)
            fig_h = max(5, len(K_vals) * 0.9)
            fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
            ax = ax_grid[0][0]
            sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                        xticklabels=N_vals, yticklabels=K_vals,
                        ax=ax, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel('N (test neighbors)')
            ax.set_ylabel('K (cache neighbors)')
            ax.set_title(
                f'Set 3 \u2014 Hit Rate vs K and N (Combined, {cap_metric(metric)})',
                pad=12,
            )
            save_fig(fig, outdir, f'kn_heatmap_combined_{metric}.png')

        # --- K/N ratio matrix: ratio (rows) × algorithm (cols), seeds averaged ---
        ratio_sub = m_sub[m_sub['algorithm'].isin(set(non_union_algos))].copy()
        ratio_sub['kn_ratio'] = (ratio_sub['cache_K'] / ratio_sub['test_N']).round(4)
        agg2 = (ratio_sub.groupby(['kn_ratio', 'algorithm', 'config_key'])['hit_rate']
                         .mean()
                         .reset_index()
                         .groupby(['kn_ratio', 'algorithm'])['hit_rate']
                         .mean()
                         .reset_index())
        kn_ratios = sorted(agg2['kn_ratio'].unique())

        if kn_ratios and non_union_algos:
            mat2 = np.full((len(kn_ratios), len(non_union_algos)), np.nan)
            for i, ratio in enumerate(kn_ratios):
                for j, algo in enumerate(non_union_algos):
                    row = agg2[(agg2['kn_ratio'] == ratio) & (agg2['algorithm'] == algo)]
                    if len(row):
                        mat2[i, j] = row.iloc[0]['hit_rate']

            fig_w = max(7, len(non_union_algos) * 2.0)
            fig_h = max(5, len(kn_ratios) * 0.6)
            fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
            ax = ax_grid[0][0]
            sns.heatmap(mat2, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                        xticklabels=lbls(non_union_algos),
                        yticklabels=[_fmt_ratio(r) for r in kn_ratios],
                        ax=ax, cbar=False)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('K/N Ratio')
            ax.set_title(
                f'Set 3 \u2014 Hit Rate by K/N Ratio and Algorithm ({cap_metric(metric)})',
                pad=12,
            )
            save_fig(fig, outdir, f'kn_ratio_matrix_{metric}.png')

#-------------------------------------------------------------------------------
# analysis entry point

def analyze(raw_dir: str, output_dir: str):
    """Load ESCI simulation results, export CSVs, and produce all plots."""
    raw_path    = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dirs = {
        'global': output_path / 'global',
        'set1':   output_path / 'set1',
        'set2':   output_path / 'set2',
        'set3':   output_path / 'set3',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"ESCI Simulation Analysis")
    print(f"{'='*80}")
    print(f"Raw dir:    {raw_path}")
    print(f"Output dir: {output_path}")
    print()

    summaries = load_summaries(str(raw_path))
    if not summaries:
        print("No summaries found. Exiting.")
        return

    df = build_summary_df(summaries)
    per_query_df = load_per_query_df(str(raw_path), summaries)

    # data exports at output root
    df.to_csv(output_path / "summary.csv", index=False)
    print(f"Exported summary.csv ({len(df)} rows)")

    if not per_query_df.empty:
        per_query_df.to_csv(output_path / "per_query_all.csv", index=False)
        print(f"Exported per_query_all.csv ({len(per_query_df)} rows)")

    with open(output_path / "summaries.json", 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Exported summaries.json ({len(summaries)} benchmarks)")

    print("\n[Global]")
    plot_global_hit_rate_heatmap(df, dirs['global'])
    plot_lemma_breakdown_global(df, dirs['global'])
    plot_avg_time(df, dirs['global'])

    print("\n[Set 1: Baseline (c=1024)]")
    plot_set1_baseline(df, dirs['set1'])

    datasets_base = Path(__file__).parent.parent.parent / 'datasets' / 'esci'

    print("\n[Set 2: Cache size scaling]")
    plot_set2_cache_size(df, dirs['set2'], datasets_base_dir=datasets_base)

    print("\n[Set 3: K/N ratios]")
    plot_set3_kn(df, dirs['set3'])

    print(f"\n{'='*80}")
    print(f"Analysis complete. Results saved to {output_path}")
    print(f"{'='*80}\n")

#-------------------------------------------------------------------------------

def _default_raw_dir() -> str:
    """Auto-detect the raw dir: picks the latest timestamped subdir if present."""
    base = Path(__file__).parent / 'raw'
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    ) if base.exists() else []
    if stamped:
        chosen = stamped[-1]
        print(f"Auto-selected run: {chosen}")
        return str(chosen)
    return str(base)


def main():
    parser = argparse.ArgumentParser(description="Analyze ESCI simulation results")
    parser.add_argument(
        "--raw_dir", type=str,
        default=None,
        help="Directory containing raw simulation results (default: latest timestamped run)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: ./processed/<timestamp>)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir or _default_raw_dir()

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(__file__).parent / "processed" / ts)
    else:
        output_dir = args.output_dir

    analyze(raw_dir, output_dir)

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
