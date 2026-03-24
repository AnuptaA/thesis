#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys, re, argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from simulations.synthetic.analyze_synthetic import (
    load_summaries,
    build_summary_df,
    load_per_query_df,
    plot_global_accuracy_heatmap,
    plot_global_hit_rate_heatmap,
    plot_per_set_hit_rate_heatmaps,
    plot_per_set_accuracy_heatmaps,
    plot_angular_validation,
    plot_lemma_breakdown_global,
    plot_time_vs_brute,
    plot_set1,
    plot_set2,
    plot_set3,
    plot_set4,
    plot_set5,
    plot_set6,
)

#-------------------------------------------------------------------------------

def find_latest_raw_dir() -> Path:
    base = Path('simulations/synthetic/raw')
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    )
    if stamped:
        return stamped[-1]

    return base

#-------------------------------------------------------------------------------

def analyze_partial(raw_dir: Path, verbose: bool = False):
    """
    Run the full analysis pipeline on whatever summaries exist in raw_dir.
    Partial datasets (simulation still running) are fine.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path('simulations/synthetic/processed') / f'test_{ts}'

    dirs = {s: outdir / s for s in ['global','set1','set2','set3','set4','set5','set6']}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # load
    summaries = load_summaries(str(raw_dir))
    if not summaries:
        print("No summary.json files found yet in raw_dir, cannot analyze")
        return

    df = build_summary_df(summaries)
    pq = load_per_query_df(str(raw_dir), summaries)

    sets_present = sorted(df['set_prefix'].unique())
    print(f"\nDatasets loaded: {len(summaries)}")
    print(f"Sets present: {sets_present}")
    print(f"Per-query rows: {len(pq):,}" if not pq.empty else "Per-query rows: 0 (per_query.csv not yet written)")

    if verbose:
        print("\nAlgorithms found:", sorted(df['algorithm'].unique()))
        print("Metrics found:", sorted(df['metric'].unique()))
        print()

    # global
    print("\n[Global]")
    plot_global_accuracy_heatmap(df, dirs['global'])
    plot_global_hit_rate_heatmap(df, dirs['global'])
    plot_per_set_hit_rate_heatmaps(df, dirs['global'])
    plot_per_set_accuracy_heatmaps(df, dirs['global'])
    plot_angular_validation(summaries, dirs['global'])
    plot_lemma_breakdown_global(df, dirs['global'])
    plot_time_vs_brute(df, dirs['global'])

    # per set
    print("\n[Set 1]")
    plot_set1(df, dirs['set1'])

    print("\n[Set 2]")
    plot_set2(df, pq, dirs['set2'])

    print("\n[Set 3]")
    plot_set3(df, dirs['set3'])

    print("\n[Set 4]")
    plot_set4(df, dirs['set4'])

    print("\n[Set 5]")
    plot_set5(df, dirs['set5'])

    print("\n[Set 6]")
    plot_set6(df, dirs['set6'])

    # summary
    print()
    print("=" * 70)
    total_pngs = len(list(outdir.rglob("*.png")))
    print(f"Done. {total_pngs} PNGs written to: {outdir}.")
    print("=" * 70)
    return outdir


#-------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis on partial or complete raw simulation results."
    )
    parser.add_argument(
        '--raw-dir', default=None,
        help="Path to raw dir (default: latest timestamped run)"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Print extra info about algorithms/metrics found"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else find_latest_raw_dir()
    analyze_partial(raw_dir, verbose=args.verbose)
