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
    'brute',
]

ALGO_LABELS = {
    'lemma1':            'CIG',
    'lemma1_no_union':   'CIG (no union)',
    'lemma2':            'HGG',
    'lemma2_no_union':   'HGG (no union)',
    'combined':          'Combined',
    'combined_no_union': 'Combined (no union)',
    'brute':             'Brute Force',
}

NO_UNION_ALGOS: Set[str] = {'lemma1_no_union', 'lemma2_no_union', 'combined_no_union'}

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
        for res in ds['results']:
            rows.append({
                'dataset':            name,
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
        dataset_name = p.parent.name
        try:
            df  = pd.read_csv(p)
            cfg = cfg_lookup.get(dataset_name, {})
            df['dataset']           = dataset_name
            df['num_cache_queries'] = cfg.get('num_cache_queries')
            df['cache_K']           = cfg.get('cache_K')
            df['test_N']            = cfg.get('test_N')
            dfs.append(df)
        except Exception as e:
            print(f"  Could not load {p}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=[
        'dataset', 'algorithm', 'metric', 'query_id',
        'cache_hit', 'hit_source', 'distance_calcs', 'time_us',
        'num_cache_queries', 'cache_K', 'test_N',
    ])

#-------------------------------------------------------------------------------
# coverage computation

def compute_sift_coverage(
    dataset_name: str,
    K: int,
    num_cache_queries: int,
    num_base_vectors: int,
    datasets_base_dir: Path,
) -> Optional[float]:
    npz = datasets_base_dir / dataset_name / f'cache_gt_angular_K{K}.npz'
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
        ac = compute_sift_coverage(name, K, C, B, datasets_base_dir)
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
        (a for a in sub['algorithm'].unique() if a != 'brute'),
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
            a_sub['num_cache_queries'] = a_sub['num_cache_queries'].astype(int)
            per_ds = a_sub.groupby(['num_cache_queries', 'dataset'])['hit_rate'].mean().reset_index()
            lvl_agg = per_ds.groupby('num_cache_queries')['hit_rate'].mean()
            ys = [float(lvl_agg.loc[c]) if c in lvl_agg.index else np.nan for c in cache_sizes]
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
    fig.suptitle('Set 2 - Coverage and Hit Rate vs Cache Size (K=100, N=20, B=50,000)', y=1.02)
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
    fig.suptitle('Set 2 - Estimated vs Actual Coverage (K=100, B=50,000)', y=1.02)
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
    ax.set_title('Hit Rate by Algorithm \u2014 All Benchmarks', pad=12)
    save_fig(fig, outdir, 'hit_rate_heatmap_global.png')

#-------------------------------------------------------------------------------

def plot_lemma_breakdown_global(df: pd.DataFrame, outdir: Path):
    """Stacked bar: CIG / HGG share within Combined algorithm hits, averaged over benchmarks.

    When combined cache hits are zero for a cache size the bars show 0% with a footnote.
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
    cache_sizes = sorted(comb['num_cache_queries'].dropna().unique().astype(int))
    pal = sns.color_palette('tab10', 2)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(5, len(cache_sizes) * 1.5 + 2) * len(metrics), 5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = comb[comb['metric'] == metric].copy()
        sub['num_cache_queries'] = sub['num_cache_queries'].astype(int)
        agg = sub.groupby('num_cache_queries')[['l1_share', 'l2_share']].mean()
        agg = agg.reindex([c for c in cache_sizes if c in agg.index])
        x   = np.arange(len(agg))
        ax.bar(x, agg['l1_share'], label='CIG', color=pal[0])
        ax.bar(x, agg['l2_share'], bottom=agg['l1_share'], label='HGG', color=pal[1])
        ax.set_xticks(x)
        ax.set_xticklabels([int(c) for c in agg.index], rotation=0)
        ax.set_xlabel('Cache Size (queries)')
        ax.set_title(cap_metric(metric))
        pct_fmt(ax)

    axes[0].set_ylabel('Share of Combined Hits')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pal[:2]]
    axes[-1].legend(handles, ['CIG', 'HGG'],
                    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    if has_zero_hits:
        _add_footnote(fig, '* Bars at 0% indicate zero combined cache hits for that cache size.')
    fig.suptitle('CIG vs HGG Contribution Within Combined Algorithm', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'lemma_breakdown_global.png')

#-------------------------------------------------------------------------------

def plot_avg_time(df: pd.DataFrame, outdir: Path):
    """Log-scale bar chart of average query time per algorithm, grouped by cache size."""
    sub = df[~df['algorithm'].isin(NO_UNION_ALGOS)].copy()
    if sub.empty or sub['avg_time_us'].isna().all():
        print("  No timing data, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    cache_sizes = sorted(sub['num_cache_queries'].dropna().unique().astype(int))
    pal = algo_palette(algos)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(max(8, len(cache_sizes) * 2 + 2) * len(metrics), 5),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub  = sub[sub['metric'] == metric].copy()
        m_sub['num_cache_queries'] = m_sub['num_cache_queries'].astype(int)
        n_grps = len(algos)
        width  = 0.8 / n_grps
        x = np.arange(len(cache_sizes))
        for i, algo in enumerate(algos):
            ys = []
            for cs in cache_sizes:
                vals = m_sub[
                    (m_sub['algorithm'] == algo) & (m_sub['num_cache_queries'] == cs)
                ]['avg_time_us'].dropna()
                ys.append(vals.mean() if len(vals) else np.nan)
            ax.bar(x + (i - n_grps / 2 + 0.5) * width, ys, width,
                   label=lbl(algo), color=pal.get(algo))
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([int(c) for c in cache_sizes])
        ax.set_xlabel('Cache Size (queries)')
        ax.set_title(cap_metric(metric))

    axes[0].set_ylabel('Avg Query Time (\u03bcs, log scale)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle('Average Query Time by Algorithm and Cache Size', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'avg_time_by_algorithm.png')

#-------------------------------------------------------------------------------
# cache scaling plots

def plot_cache_scaling(df: pd.DataFrame, outdir: Path, datasets_base_dir: Path = None):
    """Line plots (hit rate, distance calcs) and heatmap vs cache size."""
    sub = df[~df['algorithm'].isin(NO_UNION_ALGOS)].copy()
    sub['num_cache_queries'] = sub['num_cache_queries'].astype(int)

    cache_sizes = sorted(sub['num_cache_queries'].dropna().unique().astype(int))
    if len(cache_sizes) < 2:
        print("  Only one cache size found, skipping cache scaling plots.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)

    k_vals = sorted(sub['cache_K'].dropna().unique().astype(int))
    k_str  = '/'.join(str(k) for k in k_vals)
    n_vals = sorted(sub['test_N'].dropna().unique().astype(int))
    n_str  = '/'.join(str(n) for n in n_vals)

    # --- hit rate vs cache size ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in algos:
            a_sub = m_sub[m_sub['algorithm'] == algo]
            ys = [a_sub[a_sub['num_cache_queries'] == cs]['hit_rate'].mean()
                  if not a_sub[a_sub['num_cache_queries'] == cs].empty else np.nan
                  for cs in cache_sizes]
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
    fig.suptitle(f'Hit Rate vs Cache Size (K={k_str}, N={n_str})', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'hit_rate_vs_cache_size.png')

    # --- distance calcs vs cache size ---
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in algos:
            a_sub = m_sub[m_sub['algorithm'] == algo]
            ys = [a_sub[a_sub['num_cache_queries'] == cs]['avg_distance_calcs'].mean()
                  if not a_sub[a_sub['num_cache_queries'] == cs].empty else np.nan
                  for cs in cache_sizes]
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal.get(algo), linestyle=ls)
        ax.set_yscale('log')
        _set_cache_size_xaxis(ax, cache_sizes)
        ax.set_xlabel('Cache Size (log2)')
        ax.set_title(cap_metric(metric))
    axes[0].set_ylabel('Avg Distance Calcs/Query (log scale)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle(f'Avg Distance Calcs vs Cache Size (K={k_str}, N={n_str})', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'distance_calcs_vs_cache_size.png')

    # --- coverage plots ---
    if datasets_base_dir is not None:
        cov_df = compute_set2_coverage_df(sub, datasets_base_dir)
        if not cov_df.empty:
            plot_set2_coverage_and_hit_rate(sub, cov_df, outdir)
            plot_set2_coverage_estimated_vs_actual(cov_df, outdir)

    # --- heatmap: algo × cache size ---
    for metric in metrics:
        m_sub = sub[sub['metric'] == metric]
        algos_m = ordered_algos(m_sub['algorithm'].unique())
        mat = np.full((len(algos_m), len(cache_sizes)), np.nan)
        for i, algo in enumerate(algos_m):
            for j, cs in enumerate(cache_sizes):
                vals = m_sub[
                    (m_sub['algorithm'] == algo) & (m_sub['num_cache_queries'] == cs)
                ]['hit_rate']
                if len(vals):
                    mat[i, j] = vals.mean()

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
            f'Hit Rate: Algorithm \u00d7 Cache Size ({cap_metric(metric)}, K={k_str}, N={n_str})',
            pad=12,
        )
        save_fig(fig, outdir, f'hit_rate_heatmap_{metric}.png')

#-------------------------------------------------------------------------------
# analysis entry point

def analyze(raw_dir: str, output_dir: str):
    """Load SIFT simulation results, export CSVs, and produce all plots."""
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dirs = {
        'global':         output_path / 'global',
        'cache_scaling':  output_path / 'cache_scaling',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"SIFT Simulation Analysis")
    print(f"{'='*80}")
    print(f"Raw dir:    {raw_path}")
    print(f"Output dir: {output_path}")
    print()

    summaries = load_summaries(str(raw_path))
    if not summaries:
        print("No summaries found. Exiting.")
        return

    df           = build_summary_df(summaries)
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

    datasets_base = Path(__file__).parent.parent.parent / 'datasets' / 'sift'

    print("\n[Cache scaling]")
    plot_cache_scaling(df, dirs['cache_scaling'], datasets_base_dir=datasets_base)

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
    parser = argparse.ArgumentParser(description="Analyze SIFT simulation results")
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
