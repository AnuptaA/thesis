#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, re, json
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
    'figure.dpi':    150,
    'font.size':     13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
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
    'lemma1': 'CIG',
    'lemma1_no_union': 'CIG (no union)',
    'lemma2': 'HGG',
    'lemma2_no_union': 'HGG (no union)',
    'combined': 'Combined',
    'combined_no_union': 'Combined (no union)',
    'brute': 'Brute Force',
}

# we exclude union algorithms from all plots except Set 5
NO_UNION_ALGOS: Set[str] = {'lemma1_no_union', 'lemma2_no_union', 'combined_no_union'}

PERT_ORDER = ['small', 'medium', 'large']

PERT_LABELS = {
    'small': 'Small\n(0-5 deg.)',
    'medium': 'Medium\n(5-30 deg.)',
    'large': 'Large\n(30-180 deg.)',
}

VARIABILITY_ORDER = ['low', 'medium', 'high']

EXCLUDED_METRICS: set = {'cosine'}

# footnote strings
_NOTE_NO_HITS = ('* Gray "no hits" cells indicate zero cache hits for that condition, '
                 'not a data error.')
_NOTE_BANDS = 'Shaded bands: +/-1 std across seeds.'

#-------------------------------------------------------------------------------

# naming and grouping helpers

def set_prefix(name: str) -> str:
    return name.split('_')[0]

def config_key(name: str) -> str:
    return re.sub(r'_seed\d+$', '', name)

def pert_variant(name: str) -> Optional[str]:
    m = re.search(r'pert_(small|medium|large)', name)
    return m.group(1) if m else None

def n_variability(name: str) -> Optional[str]:
    m = re.search(r'varn_(low|medium|high)', name)
    return m.group(1) if m else None

#-------------------------------------------------------------------------------

# data loading

def load_summaries(raw_dir: str) -> List[Dict]:
    """Load all summary.json files from a raw results directory.

    Args:
        raw_dir: path to directory containing per-dataset subdirs, each with a summary.json

    Returns:
        list of summary dicts, each with keys 'dataset', 'config', 'results', 'validations'
    """
    raw_path = Path(raw_dir)
    summaries = []
    for p in sorted(raw_path.glob('*/summary.json')):
        with open(p) as f:
            summaries.append(json.load(f))
    print(f"Loaded {len(summaries)} dataset summaries")
    return summaries

#-------------------------------------------------------------------------------

def build_summary_df(summaries: List[Dict]) -> pd.DataFrame:
    """Flatten per-dataset summary dicts into a single long-form DataFrame.

    Args:
        summaries: list of summary dicts as returned by load_summaries

    Returns:
        DataFrame with one row per (dataset, algorithm, metric) triple; columns include
        hit_rate, accuracy, avg_distance_calcs, avg_time_us, K, N, num_cache_queries, etc.
    """
    rows = []
    for ds in summaries:
        name = ds['dataset']
        cfg = ds['config']
        for res in ds['results']:
            rows.append({
                'dataset': name,
                'set_prefix': set_prefix(name),
                'config_key': config_key(name),
                'pert_variant': pert_variant(name),
                'n_variability': n_variability(name),
                'algorithm': res['algorithm'],
                'metric': res['metric'],
                'hit_rate': res['hit_rate'],
                'accuracy': res['accuracy'],
                'total_queries': res['total_queries'],
                'cache_hits': res['cache_hits'],
                'lemma1_hits': res.get('lemma1_hits', 0),
                'lemma2_hits': res.get('lemma2_hits', 0),
                'avg_distance_calcs': res['avg_distance_calcs'],
                'avg_time_us': res['avg_time_us'],
                'K': cfg.get('K'),
                'N': cfg.get('N'),
                'num_cache_queries': cfg.get('num_cache_queries'),
                'num_base_vectors': cfg.get('num_base_vectors'),
                'dimension': cfg.get('dimension'),
                'perturbation_level': cfg.get('perturbation_level'),
                'num_clusters': cfg.get('num_clusters'),
            })
    return pd.DataFrame(rows)

#-------------------------------------------------------------------------------

def load_per_query_df(raw_dir: str, summaries: List[Dict]) -> pd.DataFrame:
    """Load per_query.csv files and enrich rows with per-query angle and N metadata.

    angle_deg is indexed from config test_perturbation_angles_deg by query_id;
    n_value from test_N_values. Both are NaN when not present in the config.

    Args:
        raw_dir: path to raw results directory
        summaries: output of load_summaries, used for config lookup

    Returns:
        concatenated DataFrame with one row per query; empty DataFrame if no csvs found
    """
    raw_path = Path(raw_dir)

    # pre-build per-dataset lookup tables from configs before reading csvs
    angle_lookup: Dict[str, list] = {}
    n_val_lookup: Dict[str, list] = {}
    cfg_lookup: Dict[str, Dict] = {}
    for ds in summaries:
        cfg_lookup[ds['dataset']] = ds['config']
        angles = ds['config'].get('test_perturbation_angles_deg')
        if angles:
            angle_lookup[ds['dataset']] = angles
        n_vals = ds['config'].get('test_N_values')
        if n_vals:
            n_val_lookup[ds['dataset']] = n_vals

    dfs = []
    for p in sorted(raw_path.glob('*/per_query.csv')):
        dataset_name = p.parent.name
        try:
            df = pd.read_csv(p)
            df['dataset'] = dataset_name
            df['set_prefix'] = set_prefix(dataset_name)
            df['config_key'] = config_key(dataset_name)
            df['pert_variant'] = pert_variant(dataset_name)
            df['n_variability'] = n_variability(dataset_name)

            cfg = cfg_lookup.get(dataset_name, {})
            df['num_cache_queries'] = cfg.get('num_cache_queries')
            df['num_clusters'] = cfg.get('num_clusters')

            angles = angle_lookup.get(dataset_name)
            if angles is not None:
                df['angle_deg'] = df['query_id'].map({i: a for i, a in enumerate(angles)})
            else:
                df['angle_deg'] = np.nan

            n_vals = n_val_lookup.get(dataset_name)
            if n_vals is not None:
                df['n_value'] = df['query_id'].map({i: v for i, v in enumerate(n_vals)})
            else:
                df['n_value'] = np.nan

            dfs.append(df)
        except Exception as e:
            print(f"Could not load {p}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame(columns=[
        'dataset', 'algorithm', 'metric', 'query_id',
        'cache_hit', 'hit_source', 'distance_calcs', 'time_us',
        'angle_deg', 'n_value', 'set_prefix', 'config_key', 'pert_variant',
        'n_variability', 'num_cache_queries', 'num_clusters',
    ])


#-------------------------------------------------------------------------------

# style/ordering helpers

def ordered_algos(algos, include_no_union: bool = False,
                  include_brute: bool = True) -> List[str]:
    """Return algorithms sorted by canonical ALGO_ORDER, optionally filtering variants.

    Args:
        algos: iterable of algorithm code names
        include_no_union: if False, removes *_no_union variants
        include_brute: if False, removes 'brute'

    Returns:
        ordered list; any unknown algos appended alphabetically at the end
    """
    algos = set(algos)
    if not include_no_union:
        algos -= NO_UNION_ALGOS
    if not include_brute:
        algos -= {'brute'}
    return [a for a in ALGO_ORDER if a in algos] + sorted(algos - set(ALGO_ORDER))

def ordered_metrics(metrics, exclude=None) -> List[str]:
    """Return metrics sorted by preferred order [euclidean, angular, cosine, ...], applying exclusions.

    Args:
        metrics: iterable of metric strings
        exclude: set of metrics to drop; defaults to EXCLUDED_METRICS ({'cosine'})

    Returns:
        filtered and ordered list
    """
    if exclude is None:
        exclude = EXCLUDED_METRICS
    preferred = ['euclidean', 'angular', 'cosine']
    metrics = set(metrics) - exclude
    return [m for m in preferred if m in metrics] + sorted(metrics - set(preferred))

def algo_palette(algos) -> Dict[str, tuple]:
    """Build a consistent tab10 color palette keyed by algorithm code name.

    Args:
        algos: iterable of algorithm code names

    Returns:
        dict mapping algo code name -> RGB tuple, ordered by ALGO_ORDER
    """
    ordered = ordered_algos(algos, include_no_union=True, include_brute=True)
    colors = sns.color_palette('tab10', len(ordered))
    return {a: c for a, c in zip(ordered, colors)}

def lbl(algo: str) -> str:
    """Return the human-readable label for an algorithm code name."""
    return ALGO_LABELS.get(algo, algo)

def lbls(algos) -> List[str]:
    """Return human-readable labels for a sequence of algorithm code names.

    Args:
        algos: iterable of algorithm code names

    Returns:
        list of display strings in the same order
    """
    return [lbl(a) for a in algos]

def cap_metric(m: str) -> str:
    return m.title()

def cap_metrics(metrics) -> List[str]:
    return [cap_metric(m) for m in metrics]

def save_fig(fig: plt.Figure, path: Path, name: str):
    """Save a figure to disk at 200 dpi and close it.

    Args:
        fig: matplotlib Figure to save
        path: output directory (created if missing)
        name: filename including extension
    """
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"{name}")

def pct_fmt(ax, axis='y'):
    """Apply percentage tick label formatting to an axis.

    Args:
        ax: matplotlib Axes
        axis: 'y' or 'x' (default 'y')
    """
    formatter = mticker.FuncFormatter(lambda v, _: f'{v:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

def _add_footnote(fig: plt.Figure, text: str):
    """Add a small gray footnote at the bottom-left of the figure.

    Args:
        fig: matplotlib Figure
        text: footnote content
    """
    fig.text(0.01, -0.01, text, fontsize=7.5, color='#555555',
             va='top', ha='left', transform=fig.transFigure)

def _left_cbar_fig(n_rows, n_cols, cmap, vmin, vmax, cbar_label, fig_w, fig_h, wspace=0.05):
    """Create a figure with a narrow left colorbar column and heatmap axes via GridSpec.

    Y-axis ticks and labels on all returned heatmap axes are set to the right side so they
    don't overlap with the colorbar.

    Args:
        n_rows: number of heatmap rows
        n_cols: number of heatmap columns
        cmap: matplotlib colormap name
        vmin: colorbar minimum value
        vmax: colorbar maximum value
        cbar_label: colorbar axis label string
        fig_w: heatmap area width in inches (colorbar column adds ~1.2 in on top)
        fig_h: figure height in inches
        wspace: horizontal space between GridSpec columns (default 0.05)

    Returns:
        (fig, axes) where axes is list[list[Axes]] of shape (n_rows, n_cols)
    """
    fig = plt.figure(figsize=(fig_w + 1.2, fig_h))
    # first GridSpec column (ratio 0.05) holds the colorbar; heatmap axes fill the rest
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[0.05] + [1] * n_cols,
                  hspace=0.4, wspace=wspace)
    cbar_ax = fig.add_subplot(gs[:, 0])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.tick_left()
    axes = [[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)] for r in range(n_rows)]
    for row in axes:
        for ax in row:
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
    return fig, axes

#-------------------------------------------------------------------------------

# global cross-set plots

def plot_global_accuracy_heatmap(df: pd.DataFrame, outdir: Path):
    """Plot accuracy heatmap across all algorithms (including no_union) and all metrics (including cosine).

    Also writes accuracy_heatmap_global.json with the raw matrix values.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[df['algorithm'] != 'brute']
    algos = ordered_algos(sub['algorithm'].unique(), include_no_union=True)
    metrics = ordered_metrics(sub['metric'].unique(), exclude=set())  # include cosine

    # only count datasets with cache hits; accuracy is undefined for pure misses
    mat = np.full((len(algos), len(metrics)), np.nan)
    for i, a in enumerate(algos):
        for j, m in enumerate(metrics):
            rows = sub[(sub['algorithm'] == a) & (sub['metric'] == m)
                       & (sub['cache_hits'] > 0)]
            if len(rows):
                mat[i, j] = rows['accuracy'].mean()

    mask = np.isnan(mat)
    fig_w = max(6, len(metrics) * 2.2)
    fig_h = max(4, len(algos) * 0.9)
    fig, ax_grid = _left_cbar_fig(1, 1, 'RdYlGn', 0, 1, 'Accuracy', fig_w, fig_h)
    ax = ax_grid[0][0]
    ax.set_facecolor('#cccccc')
    sns.heatmap(mat, annot=True, fmt='.1%', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=cap_metrics(metrics), yticklabels=lbls(algos), ax=ax,
                mask=mask, cbar=False)
    for r in range(len(algos)):
        for c in range(len(metrics)):
            if mask[r, c]:
                ax.text(c + 0.5, r + 0.5, 'no hits *', ha='center', va='center',
                        fontsize=7.5, color='black')
    ax.set_title('Accuracy by Algorithm and Metric (All Datasets)', pad=12)
    save_fig(fig, outdir, 'accuracy_heatmap_global.png')

    with open(outdir / 'accuracy_heatmap_global.json', 'w') as f:
        json.dump({'algorithms': algos, 'metrics': metrics,
                   'matrix': [[None if np.isnan(v) else v for v in row] for row in mat]},
                  f, indent=2)

#-------------------------------------------------------------------------------

def plot_global_hit_rate_heatmap(df: pd.DataFrame, outdir: Path):
    """Plot hit rate heatmap for main algorithms only (excludes no_union variants and cosine metric).

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[(df['algorithm'] != 'brute')
             & (~df['algorithm'].isin(NO_UNION_ALGOS))]
    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())

    mat = np.full((len(algos), len(metrics)), np.nan)
    for i, a in enumerate(algos):
        for j, m in enumerate(metrics):
            vals = sub[(sub['algorithm'] == a) & (sub['metric'] == m)]['hit_rate']
            if len(vals):
                mat[i, j] = vals.mean()

    fig_w = max(6, len(metrics) * 2.2)
    fig_h = max(4, len(algos) * 0.9)
    fig, ax_grid = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w, fig_h)
    ax = ax_grid[0][0]
    sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=cap_metrics(metrics), yticklabels=lbls(algos), ax=ax,
                cbar=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Hit Rate by Algorithm and Metric (All Datasets)', pad=12)
    save_fig(fig, outdir, 'hit_rate_heatmap_global.png')

#-------------------------------------------------------------------------------

def plot_per_set_hit_rate_heatmaps(df: pd.DataFrame, outdir: Path):
    """Plot all per-set hit-rate heatmaps in a single 2x3 grid with a shared left colorbar.

    Y-axis labels appear only on the rightmost panel of each row.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[(df['algorithm'] != 'brute') & (~df['algorithm'].isin(NO_UNION_ALGOS))]
    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    prefixes = sorted(sub['set_prefix'].unique())

    n_cols = 3
    n_rows = (len(prefixes) + n_cols - 1) // n_cols
    cell_w = max(4, len(metrics) * 2.0)
    cell_h = max(3, len(algos) * 0.85)

    fig = plt.figure(figsize=(cell_w * n_cols + 1.2, cell_h * n_rows))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[0.05] + [1] * n_cols,
                  hspace=0.45, wspace=0.3)
    cbar_ax = fig.add_subplot(gs[:, 0])
    axes2d = [[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)] for r in range(n_rows)]

    for idx, pfx in enumerate(prefixes):
        row_i, col_i = divmod(idx, n_cols)
        ax = axes2d[row_i][col_i]
        pfx_sub = sub[sub['set_prefix'] == pfx]
        mat = np.full((len(algos), len(metrics)), np.nan)
        for i, a in enumerate(algos):
            for j, m in enumerate(metrics):
                rows = pfx_sub[(pfx_sub['algorithm'] == a) & (pfx_sub['metric'] == m)]
                if len(rows):
                    mat[i, j] = rows['hit_rate'].mean()
        mask = np.isnan(mat)
        ax.set_facecolor('#cccccc')
        # rightmost occupied column in this row (< n_cols-1 when the last row is partial)
        row_end_col = min(n_cols - 1, len(prefixes) - 1 - row_i * n_cols)
        yticks_per = lbls(algos) if col_i == row_end_col else []
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=cap_metrics(metrics), yticklabels=yticks_per, ax=ax,
                    mask=mask, cbar=False)
        if col_i == row_end_col:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
        for r in range(len(algos)):
            for c in range(len(metrics)):
                if mask[r, c]:
                    ax.text(c + 0.5, r + 0.5, 'no hits *', ha='center', va='center',
                            fontsize=7, color='black')
        ax.set_title(f'Set {pfx[3:]}', pad=8)

    for idx in range(len(prefixes), n_rows * n_cols):
        row_i, col_i = divmod(idx, n_cols)
        axes2d[row_i][col_i].set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap='YlOrRd', norm=mpl.colors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='Hit Rate')
    cbar_ax.yaxis.set_label_position('left')
    cbar_ax.yaxis.tick_left()

    fig.suptitle('Hit Rate by Algorithm and Distance Metric (Per Test Set)', y=1.02)
    save_fig(fig, outdir, 'hit_rate_heatmap_per_set.png')

#-------------------------------------------------------------------------------

def plot_angular_validation(summaries: List[Dict], outdir: Path):
    """Plot angular-vs-cosine hit-set match rate heatmap across all algorithms and test sets.

    Match rate should be 1.0 everywhere; any deviation indicates a consistency issue.
    Includes no_union variants.

    Args:
        summaries: output of load_summaries (validation data read from each dict's 'validations' key)
        outdir: output directory
    """
    rows = []
    for ds in summaries:
        name = ds['dataset']
        for key, val in ds.get('validations', {}).items():
            algo = key.replace('_angular_vs_cosine', '')
            hits_checked = val.get('cache_hits_checked', 0)
            if hits_checked == 0:
                continue
            rows.append({
                'set_prefix': set_prefix(name),
                'algorithm':  algo,
                'match_rate': val.get('match_rate', np.nan),
            })
    if not rows:
        print("No validation data found, skipping.")
        return

    vdf = pd.DataFrame(rows)
    pivoted = vdf.groupby(['set_prefix', 'algorithm'])['match_rate'].mean().reset_index()
    algos = ordered_algos(pivoted['algorithm'].unique(), include_no_union=True)
    prefixes = sorted(pivoted['set_prefix'].unique())

    mat = np.full((len(algos), len(prefixes)), np.nan)
    for i, a in enumerate(algos):
        for j, p in enumerate(prefixes):
            row = pivoted[(pivoted['algorithm'] == a) & (pivoted['set_prefix'] == p)]
            if len(row):
                mat[i, j] = row.iloc[0]['match_rate']

    mask = np.isnan(mat)
    fig_w = max(6, len(prefixes) * 1.5)
    fig_h = max(4, len(algos) * 0.9)
    fig, ax_grid = _left_cbar_fig(1, 1, 'RdYlGn', 0, 1, 'Match Rate', fig_w, fig_h)
    ax = ax_grid[0][0]
    ax.set_facecolor('#cccccc')
    sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=prefixes, yticklabels=lbls(algos), ax=ax,
                mask=mask, cbar=False)
    for i in range(len(algos)):
        for j in range(len(prefixes)):
            if mask[i, j]:
                ax.text(j + 0.5, i + 0.5, 'no hits *', ha='center', va='center',
                        fontsize=7.5, color='black')
    ax.set_title('Angular Returned Response vs. Cosine Brute-Force Response', pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
    _add_footnote(fig, _NOTE_NO_HITS)
    save_fig(fig, outdir, 'angular_cosine_validation.png')

#-------------------------------------------------------------------------------

def plot_lemma_breakdown_global(df: pd.DataFrame, outdir: Path):
    """Plot stacked bar chart of CIG vs HGG contribution share within Combined algorithm hits.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    comb = df[(df['algorithm'] == 'combined') & (df['cache_hits'] > 0)].copy()
    if comb.empty:
        print("No combined hits for lemma breakdown, skipping.")
        return

    comb['l1_share'] = comb['lemma1_hits'] / comb['cache_hits']
    comb['l2_share'] = comb['lemma2_hits'] / comb['cache_hits']

    metrics = ordered_metrics(comb['metric'].unique())
    prefixes = sorted(comb['set_prefix'].unique())
    n_met = len(metrics)
    pal = sns.color_palette('tab10', 2)

    fig, axes = plt.subplots(1, n_met, figsize=(5 * n_met + 2, 5), sharey=True)
    if n_met == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        sub = comb[comb['metric'] == metric]
        agg = sub.groupby('set_prefix')[['l1_share', 'l2_share']].mean()
        agg = agg.reindex([p for p in prefixes if p in agg.index])
        x = np.arange(len(agg))
        ax.bar(x, agg['l1_share'], label='CIG',  color=pal[0])
        ax.bar(x, agg['l2_share'], bottom=agg['l1_share'], label='HGG', color=pal[1])
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=20, ha='right')
        ax.set_title(cap_metric(metric))
        pct_fmt(ax)

    axes[0].set_ylabel('Share of Combined Hits')
    handles = [plt.Rectangle((0, 0), 1, 1, color=pal[0]),
               plt.Rectangle((0, 0), 1, 1, color=pal[1])]
    axes[-1].legend(handles, ['CIG', 'HGG'],
                    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig.suptitle('CIG vs HGG Contribution Within Combined Algorithm', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'lemma_breakdown_global.png')

#-------------------------------------------------------------------------------

def plot_avg_time(df: pd.DataFrame, outdir: Path):
    """Plot average query time per algorithm (absolute, microseconds), log y-scale.

    Excludes no_union variants and brute force (brute force timing is not tracked
    because it omits real memory access costs).

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[(df['algorithm'] != 'brute') & (~df['algorithm'].isin(NO_UNION_ALGOS))].copy()
    if sub.empty:
        print("No timing data for non-brute algorithms, skipping.")
        return

    algos = ordered_algos(sub['algorithm'].unique())
    metrics = ordered_metrics(sub['metric'].unique())
    prefixes = sorted(sub['set_prefix'].unique())
    pal = algo_palette(algos)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        n_grps = len(algos)
        width = 0.8 / n_grps
        x = np.arange(len(prefixes))
        for i, algo in enumerate(algos):
            ys = []
            for pfx in prefixes:
                a_vals = m_sub[(m_sub['algorithm'] == algo)
                               & (m_sub['set_prefix'] == pfx)]['avg_time_us'].dropna()
                ys.append(a_vals.mean() if len(a_vals) else np.nan)
            ax.bar(x + (i - n_grps / 2 + 0.5) * width, ys, width,
                   label=lbl(algo), color=pal.get(algo))

        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(prefixes, rotation=20, ha='right')
        ax.set_title(cap_metric(metric))

    axes[0].set_ylabel('Avg Query Time (us, log scale)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                    borderaxespad=0, fontsize=8)
    fig.suptitle('Average Query Time by Algorithm', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'avg_time_by_algorithm.png')

#-------------------------------------------------------------------------------
# Set 1: Baseline
#-------------------------------------------------------------------------------

def plot_set1(df: pd.DataFrame, outdir: Path):
    """Plot Set 1 (baseline): average distance calculations per query, log y-scale.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[df['set_prefix'] == 'set1']
    if sub.empty:
        print("No set1 data.")
        return

    algos = ordered_algos(
        (a for a in sub['algorithm'].unique() if a != 'brute'),
        include_no_union=False,
    )
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub = sub[(sub['metric'] == metric) & (sub['algorithm'] != 'brute')
                    & (~sub['algorithm'].isin(NO_UNION_ALGOS))]
        per_seed = m_sub.groupby(['config_key', 'algorithm'])['avg_distance_calcs'].mean().reset_index()
        stats = per_seed.groupby('algorithm')['avg_distance_calcs'].agg(['mean', 'std']).reindex(algos)

        for i, algo in enumerate(algos):
            if algo not in stats.index:
                continue
            mean = stats.loc[algo, 'mean']
            std = 0 if pd.isna(stats.loc[algo, 'std']) else stats.loc[algo, 'std']
            ax.bar(i, mean, 0.6, color=pal[algo],
                   yerr=std, capsize=4, error_kw=dict(lw=1.2), label=lbl(algo))

        ax.set_yscale('log')
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(lbls(algos), rotation=30, ha='right')
        ax.set_title(cap_metric(metric))

    axes[0].set_ylabel('Avg Distance Calcs/Query (log scale)')
    fig.suptitle('Set 1 - Average Distance Calculations per Query (K=100, N=20)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set1_distance_calcs.png')

#-------------------------------------------------------------------------------
# Set 2: Perturbation levels
#-------------------------------------------------------------------------------

def plot_set2(df: pd.DataFrame, pq: pd.DataFrame, outdir: Path):
    """Plot Set 2 (perturbation levels): hit-rate heatmap, lemma breakdown, and angle histograms.

    Args:
        df: summary DataFrame from build_summary_df
        pq: per-query DataFrame from load_per_query_df (used for lemma breakdown and histograms)
        outdir: output directory
    """
    sub = df[df['set_prefix'] == 'set2']
    if sub.empty:
        print("No set2 data.")
        return

    levels_present = [l for l in PERT_ORDER if l in sub['pert_variant'].dropna().unique()]
    algos = ordered_algos(
        (a for a in sub['algorithm'].unique() if a != 'brute'),
        include_no_union=False,
    )
    metrics = ordered_metrics(sub['metric'].unique())

    # consolidated hit-rate heatmap: all perturbation levels side-by-side
    n_levels = len(levels_present)
    cell_w2 = max(4, len(metrics) * 2.2)
    fig, ax_grid2 = _left_cbar_fig(
        1, n_levels, 'YlOrRd', 0, 1, 'Hit Rate',
        cell_w2 * n_levels, max(4, len(algos) * 0.9),
        wspace=0.15,
    )

    for ax_i, (ax, level) in enumerate(zip(ax_grid2[0], levels_present)):
        lvl_sub = sub[(sub['pert_variant'] == level) & (sub['algorithm'] != 'brute')
                      & (~sub['algorithm'].isin(NO_UNION_ALGOS))]
        mat = np.full((len(algos), len(metrics)), np.nan)
        for i, algo in enumerate(algos):
            for j, metric in enumerate(metrics):
                vals = lvl_sub[(lvl_sub['algorithm'] == algo)
                               & (lvl_sub['metric'] == metric)]['hit_rate']
                if len(vals):
                    mat[i, j] = vals.mean()
        level_label = PERT_LABELS[level].replace('\n', ' ')
        yticks = lbls(algos) if ax_i == n_levels - 1 else []
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=cap_metrics(metrics), yticklabels=yticks,
                    ax=ax, cbar=False)
        if yticks:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(f'Perturbation: {level_label}')

    fig.suptitle('Set 2 - Hit Rate by Algorithm and Metric', y=1.02)
    save_fig(fig, outdir, 'set2_hit_rate_heatmap_all.png')

    # lemma breakdown: perturbation angle on x-axis from per-query data
    if not pq.empty and 'hit_source' in pq.columns:
        pq2_hit = pq[(pq['set_prefix'] == 'set2') & (pq['cache_hit'] == 1)
                     & (pq['algorithm'] == 'combined')].dropna(subset=['angle_deg'])
        if not pq2_hit.empty:
            panel_metrics = [m for m in ['angular', 'euclidean'] if m in metrics]
            if not panel_metrics:
                panel_metrics = metrics[:2]
            pal2 = sns.color_palette('tab10', 2)
            n_panels = len(panel_metrics)
            fig, panel_axes = plt.subplots(
                1, n_panels, figsize=(8 * n_panels, 5), sharey=True, squeeze=False
            )
            for col_i, panel_metric in enumerate(panel_metrics):
                ax = panel_axes[0][col_i]
                pq2m = pq2_hit[pq2_hit['metric'] == panel_metric].copy()
                bins = np.linspace(pq2m['angle_deg'].min(), pq2m['angle_deg'].max(), 50) \
                       if not pq2m.empty else np.linspace(0, 180, 50)
                legend_handles = []
                for src, col, label_str in [('lemma1', pal2[0], 'CIG'), ('lemma2', pal2[1], 'HGG')]:
                    sub_src = pq2m[pq2m['hit_source'] == src]['angle_deg'] if not pq2m.empty else pd.Series([], dtype=float)
                    if len(sub_src):
                        ax.hist(sub_src, bins=bins, density=True, alpha=0.55,
                                color=col, label=label_str, edgecolor='none')
                    legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.55, label=label_str))
                ax.set_xlabel('Base Perturbation Angle (degrees)')
                if col_i == 0:
                    ax.set_ylabel('Density of Cache Hits')
                ax.set_title(cap_metric(panel_metric))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
                if col_i == n_panels - 1:
                    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
                              loc='upper left', borderaxespad=0)
                ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
                ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
            fig.suptitle('Set 2 - Algorithmic Hit Source by Base Perturbation Angle (Metric)', y=1.02)
            plt.tight_layout()
            save_fig(fig, outdir, 'set2_lemma_breakdown.png')

    # consolidated angle histograms: rows=metrics, cols=[CIG, HGG, Combined]
    if pq.empty or pq['angle_deg'].isna().all():
        print("No per-query angle data for set2 histograms, skipping.")
        return

    pq2 = pq[pq['set_prefix'] == 'set2'].dropna(subset=['angle_deg'])
    if pq2.empty:
        return

    hist_algos = [a for a in ['lemma1', 'lemma2', 'combined']
                  if a in pq2['algorithm'].unique()]
    n_rows_h = len(metrics)
    n_cols_h = len(hist_algos)
    bins = np.linspace(0, 180, 181 // 5 + 1)  # 5-deg bins from 0 to 180

    fig, axes2d = plt.subplots(
        n_rows_h, n_cols_h,
        figsize=(4.5 * n_cols_h, 3.5 * n_rows_h),
        sharex=True, sharey='row', squeeze=False,
    )
    for row_i, metric in enumerate(metrics):
        for col_i, algo in enumerate(hist_algos):
            ax = axes2d[row_i][col_i]
            pq_m = pq2[(pq2['metric'] == metric) & (pq2['algorithm'] == algo)]
            hits = pq_m[pq_m['cache_hit'] == 1]['angle_deg']
            misses = pq_m[pq_m['cache_hit'] == 0]['angle_deg']
            if len(hits):
                ax.hist(hits, bins=bins, density=True, alpha=0.55,
                        color='steelblue', label='Hit', edgecolor='none')
            if len(misses):
                ax.hist(misses, bins=bins, density=True, alpha=0.55,
                        color='tomato', label='Miss', edgecolor='none')
            if row_i == 0:
                ax.set_title(lbl(algo))
            if col_i == 0:
                ax.set_ylabel(f'{cap_metric(metric)}\nDensity')
            if row_i == n_rows_h - 1:
                ax.set_xlabel('Perturbation Angle (degrees)')
            ax.set_xlim(0, 180)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(30))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(2.5))
            if row_i == 0 and col_i == n_cols_h - 1:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    fig.suptitle('Set 2 - Angle Distribution: Hits vs Misses', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set2_angle_hist_consolidated.png')

    # zoomed version: 0-5 degree range, 0.25-deg bins
    bins_zoom = np.linspace(0, 5, 21)  # 20 bins of 0.25 deg each
    fig_z, axes2d_z = plt.subplots(
        n_rows_h, n_cols_h,
        figsize=(4.5 * n_cols_h, 3.5 * n_rows_h),
        sharex=True, sharey='row', squeeze=False,
    )
    for row_i, metric in enumerate(metrics):
        for col_i, algo in enumerate(hist_algos):
            ax = axes2d_z[row_i][col_i]
            pq_m = pq2[(pq2['metric'] == metric) & (pq2['algorithm'] == algo)]
            hits = pq_m[pq_m['cache_hit'] == 1]['angle_deg']
            misses = pq_m[pq_m['cache_hit'] == 0]['angle_deg']
            if len(hits):
                ax.hist(hits, bins=bins_zoom, density=True, alpha=0.55,
                        color='steelblue', label='Hit', edgecolor='none')
            if len(misses):
                ax.hist(misses, bins=bins_zoom, density=True, alpha=0.55,
                        color='tomato', label='Miss', edgecolor='none')
            if row_i == 0:
                ax.set_title(lbl(algo))
            if col_i == 0:
                ax.set_ylabel(f'{cap_metric(metric)}\nDensity')
            if row_i == n_rows_h - 1:
                ax.set_xlabel('Perturbation Angle (degrees)')
            ax.set_xlim(0, 5)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.25))
            if row_i == 0 and col_i == n_cols_h - 1:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig_z.suptitle('Set 2 - Angle Distribution: Hits vs Misses (0-5 deg zoom)', y=1.02)
    plt.tight_layout()
    save_fig(fig_z, outdir, 'set2_angle_hist_zoomed.png')

#-------------------------------------------------------------------------------
# Set 3: K/N ratios
#-------------------------------------------------------------------------------

def plot_set3(df: pd.DataFrame, outdir: Path):
    """Plot Set 3 (K/N ratios): KxN hit-rate heatmap for Combined, plus K/N ratio matrix for all algos.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[df['set_prefix'] == 'set3']
    if sub.empty:
        print("No set3 data.")
        return

    metrics = ordered_metrics(sub['metric'].unique())
    all_algos = set(sub['algorithm'].unique())
    non_brute_algos = ordered_algos(
        (a for a in all_algos if a != 'brute'),
        include_no_union=False,
    )

    # combined-only KxN heatmaps
    for metric in metrics:
        m_sub = sub[(sub['metric'] == metric) & (sub['algorithm'] == 'combined')]
        agg = m_sub.groupby(['K', 'N'])['hit_rate'].mean().reset_index()
        K_vals = sorted(agg['K'].unique())
        N_vals = sorted(agg['N'].unique())
        if not K_vals or not N_vals:
            continue

        mat = np.full((len(K_vals), len(N_vals)), np.nan)
        for _, row in agg.iterrows():
            i = K_vals.index(row['K'])
            j = N_vals.index(row['N'])
            mat[i, j] = row['hit_rate']

        fig_w3a = max(8, len(N_vals) * 1.3)
        fig_h3a = max(6, len(K_vals) * 0.9)
        fig, ax_grid3a = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w3a, fig_h3a)
        ax = ax_grid3a[0][0]
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=[int(n) for n in N_vals],
                    yticklabels=[int(k) for k in K_vals], ax=ax,
                    cbar=False)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('N')
        ax.set_ylabel('K')
        ax.set_title(f'Set 3 - Hit Rate vs K and N (Combined, {cap_metric(metric)})', pad=12)
        save_fig(fig, outdir, f'set3_kn_heatmap_combined_{metric}.png')

    # hit-rate matrix by K/N ratio vs algorithm
    def _fmt_ratio(r):
        return str(int(r)) if r == int(r) else f'{r:.2f}'.rstrip('0').rstrip('.')

    for metric in metrics:
        m_sub = sub[(sub['metric'] == metric)
                    & (sub['algorithm'].isin(set(non_brute_algos)))]
        agg = m_sub.groupby(['K', 'N', 'algorithm'])['hit_rate'].mean().reset_index()
        # collapse distinct (K, N) pairs sharing the same ratio into a single row
        agg['kn_ratio'] = (agg['K'] / agg['N']).round(4)
        ratio_agg = agg.groupby(['kn_ratio', 'algorithm'])['hit_rate'].mean().reset_index()

        kn_ratios = sorted(ratio_agg['kn_ratio'].unique())
        mat = np.full((len(kn_ratios), len(non_brute_algos)), np.nan)
        for i, ratio in enumerate(kn_ratios):
            for j, algo in enumerate(non_brute_algos):
                row = ratio_agg[(ratio_agg['kn_ratio'] == ratio)
                                & (ratio_agg['algorithm'] == algo)]
                if len(row):
                    mat[i, j] = row.iloc[0]['hit_rate']

        fig_w3b = max(7, len(non_brute_algos) * 2.0)
        fig_h3b = max(6, len(kn_ratios) * 0.55)
        fig, ax_grid3b = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w3b, fig_h3b)
        ax = ax_grid3b[0][0]
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=lbls(non_brute_algos),
                    yticklabels=[_fmt_ratio(r) for r in kn_ratios],
                    ax=ax, cbar=False)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('K/N Ratio')
        ax.set_title(f'Set 3 - Hit Rate by K/N Ratio and Algorithm ({cap_metric(metric)})', pad=12)
        save_fig(fig, outdir, f'set3_kn_ratio_matrix_{metric}.png')

#-------------------------------------------------------------------------------
# Set 4: Variable N
#-------------------------------------------------------------------------------

def plot_set4(df: pd.DataFrame, pq: pd.DataFrame, outdir: Path):
    """Plot Set 4 (variable N): hit rate and computation ratio by variability level, plus N-value line plots.

    Args:
        df: summary DataFrame from build_summary_df
        pq: per-query DataFrame from load_per_query_df
        outdir: output directory
    """
    sub = df[df['set_prefix'] == 'set4']
    if sub.empty:
        print("No set4 data.")
        return

    levels_present = [l for l in VARIABILITY_ORDER if l in sub['n_variability'].dropna().unique()]
    algos = ordered_algos(
        (a for a in sub['algorithm'].unique() if a != 'brute'),
        include_no_union=False,
    )
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)

    def _bar_per_level(bar_axes, value_col, normalize_by_brute=False):
        for ax, metric in zip(bar_axes, metrics):
            m_sub = sub[(sub['metric'] == metric)
                          & (sub['algorithm'] != 'brute')
                          & (~sub['algorithm'].isin(NO_UNION_ALGOS))]
            brute_c = sub[(sub['metric'] == metric)
                          & (sub['algorithm'] == 'brute')]['avg_distance_calcs'].mean()
            n_grps = len(algos)
            width = 0.8 / n_grps
            x = np.arange(len(levels_present))

            for i, algo in enumerate(algos):
                a_sub = m_sub[m_sub['algorithm'] == algo]
                # average per config/seed first to prevent unequal seed counts from skewing stats
                per_seed = a_sub.groupby(['n_variability', 'config_key'])[value_col].mean().reset_index()
                lvl_agg = per_seed.groupby('n_variability')[value_col].agg(['mean', 'std'])
                ys, errs = [], []
                for lvl in levels_present:
                    if lvl in lvl_agg.index:
                        mean = lvl_agg.loc[lvl, 'mean']
                        std = lvl_agg.loc[lvl, 'std']
                        std = 0 if pd.isna(std) else std
                        if normalize_by_brute and brute_c and not np.isnan(brute_c):
                            ys.append(1 - mean / brute_c)
                            errs.append(std / brute_c)
                        else:
                            ys.append(mean)
                            errs.append(std)
                    else:
                        ys.append(0); errs.append(0)

                ax.bar(x + (i - n_grps / 2 + 0.5) * width, ys, width,
                       label=lbl(algo), color=pal[algo],
                       yerr=errs, capsize=3, error_kw=dict(lw=1))

            ax.set_xticks(x)
            ax.set_xticklabels([l.title() for l in levels_present])
            ax.set_title(cap_metric(metric))
            ax.set_xlabel('N Variability Level')
            pct_fmt(ax)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    _bar_per_level(axes, 'hit_rate')
    for ax in axes:
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel('Hit Rate')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    fig.suptitle('Set 4 - Hit Rate by N Variability Level (K=100)', y=1.02)
    _add_footnote(fig, '* HGG had zero cache hits across all variability levels and metrics.')
    plt.tight_layout()
    save_fig(fig, outdir, 'set4_hit_rate_vs_variability.png')

    # average distance calculations per query by variability level, log scale
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m_sub = sub[(sub['metric'] == metric)
                    & (sub['algorithm'] != 'brute')
                    & (~sub['algorithm'].isin(NO_UNION_ALGOS))]
        n_grps = len(algos)
        width = 0.8 / n_grps
        x = np.arange(len(levels_present))
        for i, algo in enumerate(algos):
            a_sub = m_sub[m_sub['algorithm'] == algo]
            per_seed = a_sub.groupby(['n_variability', 'config_key'])['avg_distance_calcs'].mean().reset_index()
            lvl_agg = per_seed.groupby('n_variability')['avg_distance_calcs'].agg(['mean', 'std'])
            ys, errs = [], []
            for lvl in levels_present:
                if lvl in lvl_agg.index:
                    mean = lvl_agg.loc[lvl, 'mean']
                    std = 0 if pd.isna(lvl_agg.loc[lvl, 'std']) else lvl_agg.loc[lvl, 'std']
                    ys.append(mean)
                    errs.append(std)
                else:
                    ys.append(np.nan); errs.append(0)
            ax.bar(x + (i - n_grps / 2 + 0.5) * width, ys, width,
                   label=lbl(algo), color=pal[algo],
                   yerr=errs, capsize=3, error_kw=dict(lw=1))
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([l.title() for l in levels_present])
        ax.set_xlabel('N Variability Level')
        ax.set_title(cap_metric(metric))
    axes[0].set_ylabel('Avg Distance Calcs/Query (log scale)')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    fig.suptitle('Set 4 - Average Distance Calculations by Variability Level (K=100)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set4_distance_calcs.png')

    # N-value vs hit rate line plots by variability level, from per-query data
    if pq.empty or 'n_value' not in pq.columns or pq['n_value'].isna().all():
        print("No per-query N values for set4 line plots, skipping.")
        return

    pq4 = pq[pq['set_prefix'] == 'set4'].dropna(subset=['n_value'])
    if pq4.empty:
        return

    n_rows_p = len(levels_present)
    n_cols_p = len(metrics)
    fig, axes2d = plt.subplots(
        n_rows_p, n_cols_p,
        figsize=(5 * n_cols_p + 2, 3.5 * n_rows_p),
        sharey=True, squeeze=False,
    )
    for row_i, lvl in enumerate(levels_present):
        pq_lvl = pq4[pq4['n_variability'] == lvl]
        for col_i, metric in enumerate(metrics):
            ax = axes2d[row_i][col_i]
            pq_m = pq_lvl[pq_lvl['metric'] == metric]
            for algo in algos:
                pq_a = pq_m[pq_m['algorithm'] == algo]
                if pq_a.empty:
                    continue
                agg = (pq_a.groupby('n_value')['cache_hit']
                              .mean()
                              .reset_index()
                              .sort_values('n_value'))
                ls = '--' if algo == 'lemma1' else '-'
                ax.plot(agg['n_value'], agg['cache_hit'],
                        marker='o', markersize=3, linestyle=ls,
                        label=lbl(algo), color=pal.get(algo))
                mean_hr = agg['cache_hit'].mean()
                ax.axhline(mean_hr, linestyle=':', color=pal.get(algo), alpha=0.5)
            pct_fmt(ax)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(10, 60)
            if row_i == 0:
                ax.set_title(cap_metric(metric))
            if col_i == 0:
                ax.set_ylabel('Hit Rate')
            if row_i == n_rows_p - 1:
                ax.set_xlabel('Sampled N Value')

    axes2d[0][-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          borderaxespad=0, fontsize=8)
    fig.suptitle('Set 4 - Hit Rate vs Sampled N Value by Variability Level (K=100)', y=1.02)
    _add_footnote(fig, 'Dashed line = CIG (overlaps Combined in most conditions).')
    plt.tight_layout()
    save_fig(fig, outdir, 'set4_hit_rate_vs_n_value.png')

#-------------------------------------------------------------------------------
# Set 5: Union effectiveness
#-------------------------------------------------------------------------------

def plot_set5(df: pd.DataFrame, outdir: Path):
    """Plot Set 5 (union effectiveness): hit rate of union vs no-union variants by cluster count.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[df['set_prefix'] == 'set5']
    if sub.empty:
        print("No set5 data.")
        return

    metrics = ordered_metrics(sub['metric'].unique())
    all_algos = set(sub['algorithm'].unique())

    # only include algorithm pairs where at least one variant is present in data
    union_pairs = []
    for base in ['lemma1', 'lemma2', 'combined']:
        nu = f'{base}_no_union'
        if base in all_algos or nu in all_algos:
            union_pairs.append((base, nu))
    if not union_pairs:
        print("No union/no-union pairs in set5 data.")
        return

    X_vals = sorted(sub['num_clusters'].dropna().unique().astype(int))
    if not X_vals:
        print("No num_clusters values in set5.")
        return

    n_rows = len(union_pairs)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols + 2, 4 * n_rows),
                             sharey='row', sharex=True,
                             squeeze=False)

    pair_colors = sns.color_palette('tab10', n_rows * 2)

    for row, (union_algo, no_union_algo) in enumerate(union_pairs):
        c_union = pair_colors[row * 2]
        c_no_union = pair_colors[row * 2 + 1]

        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            m_sub = sub[sub['metric'] == metric]

            for algo, color in [(union_algo, c_union), (no_union_algo, c_no_union)]:
                a_sub = m_sub[m_sub['algorithm'] == algo]
                if a_sub.empty:
                    continue
                per_seed = a_sub.groupby(['num_clusters', 'dataset'])['hit_rate'].mean().reset_index()
                lvl_agg = per_seed.groupby('num_clusters')['hit_rate'].agg(['mean', 'std'])
                ys = [float(lvl_agg.loc[x, 'mean']) if x in lvl_agg.index else np.nan for x in X_vals]
                errs = [float(lvl_agg.loc[x, 'std']) if x in lvl_agg.index else 0 for x in X_vals]
                errs = [0 if np.isnan(e) else e for e in errs]
                ax.plot(X_vals, ys, marker='o', label=lbl(algo), color=color)

            # column header: metric name (top row only)
            if row == 0:
                ax.set_title(cap_metric(metric))
            pct_fmt(ax)
            ax.set_ylim(0, 1.05)
            # legend per panel because each row shows a different algo pair
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      borderaxespad=0, fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel('Number of Cluster Centers (X)')
    for row_i, (row_axes, (union_algo, _)) in enumerate(zip(axes, union_pairs)):
        row_axes[0].set_ylabel(f'{lbl(union_algo)}\nHit Rate')

    fig.suptitle('Set 5 - Union vs No-Union Hit Rate by Cluster Count (K=50, N varies)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set5_union_vs_no_union.png')

    # --- K/N ratio plot (per X value) ---
    # K is fixed at 50; N varies so K/N = 50/N
    kn_sub = sub.dropna(subset=['K', 'N', 'num_clusters'])
    if kn_sub.empty or kn_sub['N'].nunique() < 2:
        return

    kn_sub = kn_sub.copy()
    kn_sub['kn_ratio'] = kn_sub['K'] / kn_sub['N']
    kn_ratios = sorted(kn_sub['kn_ratio'].unique())
    x_values = sorted(kn_sub['num_clusters'].dropna().unique().astype(int))
    x_colors = sns.color_palette('viridis', len(x_values))

    fig2, axes2 = plt.subplots(n_rows, n_cols,
                               figsize=(5 * n_cols + 2, 4 * n_rows),
                               sharey='row', sharex=True,
                               squeeze=False)

    for row, (union_algo, no_union_algo) in enumerate(union_pairs):
        for col, metric in enumerate(metrics):
            ax = axes2[row][col]
            m_sub = kn_sub[kn_sub['metric'] == metric]

            for algo, ls in [(union_algo, '-'), (no_union_algo, '--')]:
                a_sub = m_sub[m_sub['algorithm'] == algo]
                if a_sub.empty:
                    continue
                for xi, X in enumerate(x_values):
                    x_data = a_sub[a_sub['num_clusters'] == X]
                    if x_data.empty:
                        continue
                    per_seed = x_data.groupby(['kn_ratio', 'dataset'])['hit_rate'].mean().reset_index()
                    agg = per_seed.groupby('kn_ratio')['hit_rate'].mean().reset_index()
                    label = f'X={X} {lbl(algo)}' if row == 0 and col == 0 else None
                    ax.plot(agg['kn_ratio'], agg['hit_rate'],
                            marker='o', linestyle=ls,
                            color=x_colors[xi], label=label)

            if row == 0:
                ax.set_title(cap_metric(metric))
            pct_fmt(ax)
            ax.set_ylim(0, 1.05)

        axes2[row][0].set_ylabel(f'{lbl(union_algo)}\nHit Rate')

    for ax in axes2[-1]:
        ax.set_xlabel('K/N Ratio')
    axes2[0][-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                        borderaxespad=0, fontsize=7)
    fig2.suptitle('Set 5 - Hit Rate vs K/N Ratio by Cluster Count (K=50, N varies)', y=1.02)
    _add_footnote(fig2, 'Solid = union variant; dashed = no-union variant.')
    plt.tight_layout()
    save_fig(fig2, outdir, 'set5_hit_rate_vs_kn_ratio.png')

#-------------------------------------------------------------------------------
# Set 6: Cache size variation
#-------------------------------------------------------------------------------

def plot_set6(df: pd.DataFrame, outdir: Path):
    """Plot Set 6 (cache size variation): hit rate and distance calc line plots, plus hit-rate heatmap.

    Args:
        df: summary DataFrame from build_summary_df
        outdir: output directory
    """
    sub = df[df['set_prefix'] == 'set6']
    if sub.empty:
        print("No set6 data.")
        return

    algos = ordered_algos(
        (a for a in sub['algorithm'].unique() if a != 'brute'),
        include_no_union=False,
    )
    metrics = ordered_metrics(sub['metric'].unique())
    pal = algo_palette(algos)
    cache_sizes = sorted(sub['num_cache_queries'].dropna().unique().astype(int))
    if not cache_sizes:
        print("No cache size data in set6.")
        return

    def _get_agg(a_sub, value_col, sizes):
        """Aggregate value_col by cache size, averaging across seeds first then taking mean/std.

        Args:
            a_sub: DataFrame filtered to one algorithm and metric
            value_col: column to aggregate (e.g. 'hit_rate', 'avg_distance_calcs')
            sizes: ordered list of integer cache sizes to align output to

        Returns:
            (ys, errs) lists of mean and std values aligned to sizes; NaN for missing sizes
        """
        a_sub = a_sub.copy()
        a_sub['num_cache_queries'] = a_sub['num_cache_queries'].astype(int)
        # average per seed first so each seed contributes equally to the final mean/std
        per_seed = a_sub.groupby(['num_cache_queries', 'dataset'])[value_col].mean().reset_index()
        lvl_agg = per_seed.groupby('num_cache_queries')[value_col].agg(['mean', 'std'])
        ys = [float(lvl_agg.loc[c, 'mean']) if c in lvl_agg.index else np.nan for c in sizes]
        errs = [float(lvl_agg.loc[c, 'std']) if c in lvl_agg.index else 0 for c in sizes]
        errs = [0 if np.isnan(e) else e for e in errs]
        return ys, errs

    # hit rate vs cache size
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub = sub[(sub['metric'] == metric) & (sub['algorithm'] != 'brute')
                    & (~sub['algorithm'].isin(NO_UNION_ALGOS))]
        for algo in algos:
            a_sub = m_sub[m_sub['algorithm'] == algo]
            if a_sub.empty:
                continue
            ys, errs = _get_agg(a_sub, 'hit_rate', cache_sizes)
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal[algo], linestyle=ls)
        ax.set_xlabel('Cache Size (num queries)')
        ax.set_title(cap_metric(metric))
        pct_fmt(ax)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel('Hit Rate')
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    _add_footnote(fig, 'Dashed line = CIG (overlaps Combined in most conditions).')
    fig.suptitle('Set 6 - Hit Rate vs Cache Size (K=100, N=20)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set6_hit_rate_vs_cache_size.png')

    # distance calcs vs cache size (log scale)
    all_algos_dc = ordered_algos(
        (a for a in sub['algorithm'].unique() if a != 'brute'),
        include_no_union=False,
    )
    pal_all = algo_palette(all_algos_dc)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics) + 2, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        m_sub = sub[sub['metric'] == metric]
        for algo in all_algos_dc:
            a_sub = m_sub[m_sub['algorithm'] == algo]
            if a_sub.empty:
                continue
            ys, errs = _get_agg(a_sub, 'avg_distance_calcs', cache_sizes)
            ls = '--' if algo == 'lemma1' else '-'
            ax.plot(cache_sizes, ys, marker='o', label=lbl(algo),
                    color=pal_all.get(algo), linestyle=ls)

        ax.set_yscale('log')
        ax.set_xlabel('Cache Size (num queries)')
        ax.set_ylabel('Avg Distance Calcs/Query (log scale)')
        ax.set_title(cap_metric(metric))

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=8)
    _add_footnote(fig, 'Dashed line = CIG (overlaps Combined in most conditions).')
    fig.suptitle('Set 6 - Average Distance Calculations vs Cache Size (K=100, N=20)', y=1.02)
    plt.tight_layout()
    save_fig(fig, outdir, 'set6_distance_calcs_vs_cache_size.png')

    # hit-rate heatmap: algorithm vs cache size, one per metric
    for metric in metrics:
        m_sub = sub[(sub['metric'] == metric) & (sub['algorithm'] != 'brute')
                      & (~sub['algorithm'].isin(NO_UNION_ALGOS))]
        algos_m = ordered_algos(m_sub['algorithm'].unique(), include_no_union=False)
        mat = np.full((len(algos_m), len(cache_sizes)), np.nan)
        for i, algo in enumerate(algos_m):
            a_sub_m = m_sub[m_sub['algorithm'] == algo].copy()
            a_sub_m['num_cache_queries'] = a_sub_m['num_cache_queries'].astype(int)
            per_seed = a_sub_m.groupby(['num_cache_queries', 'dataset'])['hit_rate'].mean().reset_index()
            lvl_agg = per_seed.groupby('num_cache_queries')['hit_rate'].mean()
            for j, c in enumerate(cache_sizes):
                if c in lvl_agg.index:
                    mat[i, j] = lvl_agg[c]

        fig_w6 = max(7, len(cache_sizes) * 1.5)
        fig_h6 = max(4, len(algos_m) * 0.9)
        fig, ax_grid6 = _left_cbar_fig(1, 1, 'YlOrRd', 0, 1, 'Hit Rate', fig_w6, fig_h6)
        ax = ax_grid6[0][0]
        sns.heatmap(mat, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=1,
                    xticklabels=[int(c) for c in cache_sizes],
                    yticklabels=lbls(algos_m), ax=ax,
                    cbar=False)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xlabel('Cache Size (num queries)')
        ax.set_title(f'Set 6 - Hit Rate: Algorithm vs Cache Size ({cap_metric(metric)}, K=100, N=20)', pad=12)
        save_fig(fig, outdir, f'set6_hit_rate_heatmap_{metric}.png')

#-------------------------------------------------------------------------------
# actual analysis
#-------------------------------------------------------------------------------

def analyze(raw_dir: str = 'simulations/synthetic/raw', out_name: str = None):
    """Run the full analysis pipeline: load data, produce all plots, and write outputs.

    Args:
        raw_dir: path to raw simulation results directory
        out_name: output subdirectory name under processed/; defaults to current timestamp

    Returns:
        Path to the created output base directory
    """
    print("=" * 80)
    print("Synthetic Analysis")
    print("=" * 80)

    if out_name is None:
        out_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    base = Path('simulations/synthetic/processed') / out_name
    dirs = {
        'global': base / 'global',
        'set1': base / 'set1',
        'set2': base / 'set2',
        'set3': base / 'set3',
        'set4': base / 'set4',
        'set5': base / 'set5',
        'set6': base / 'set6',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading summaries from {raw_dir}...")
    summaries = load_summaries(raw_dir)
    if not summaries:
        print("No results found.")
        return

    df = build_summary_df(summaries)

    print("\nLoading per-query data...")
    pq = load_per_query_df(raw_dir, summaries)
    if not pq.empty:
        print(f"{len(pq):,} per-query rows")
    else:
        print("No per_query.csv files found (angle scatter plots will be skipped)")

    print("\n[Global]")
    plot_global_accuracy_heatmap(df, dirs['global'])
    plot_global_hit_rate_heatmap(df, dirs['global'])
    plot_per_set_hit_rate_heatmaps(df, dirs['global'])
    plot_angular_validation(summaries, dirs['global'])
    plot_lemma_breakdown_global(df, dirs['global'])
    plot_avg_time(df, dirs['global'])

    print("\n[Set 1: Baseline]")
    plot_set1(df, dirs['set1'])

    print("\n[Set 2: Perturbation levels]")
    plot_set2(df, pq, dirs['set2'])

    print("\n[Set 3: K/N ratios]")
    plot_set3(df, dirs['set3'])

    print("\n[Set 4: Variable N]")
    plot_set4(df, pq, dirs['set4'])

    print("\n[Set 5: Union effectiveness]")
    plot_set5(df, dirs['set5'])

    print("\n[Set 6: Cache size]")
    plot_set6(df, dirs['set6'])

    print(f"\nDone. Results in: {base}")
    return base

#-------------------------------------------------------------------------------

def _default_raw_dir() -> str:
    """Auto-detect the raw dir: picks the latest timestamped subdir (YYYYMMDD_HHMMSS) if present,
    otherwise falls back to 'simulations/synthetic/raw'.

    Returns:
        path string suitable for passing to analyze()
    """
    base = Path('simulations/synthetic/raw')
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    )
    if stamped:
        chosen = stamped[-1]
        print(f"Auto-selected run: {chosen}")
        return str(chosen)
    return str(base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze synthetic simulation results")
    parser.add_argument('--raw-dir',  default=None, help="Path to raw results dir (default: latest timestamped run)")
    parser.add_argument('--out-name', default=None)
    args = parser.parse_args()

    raw_dir = args.raw_dir or _default_raw_dir()
    analyze(raw_dir, args.out_name)

#-------------------------------------------------------------------------------

