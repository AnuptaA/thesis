#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
matplotlib.use('Agg')

from datasets.dataloaders import (
    load_sift_dataset,
    load_sift_benchmark,
    load_esci_data,
    load_esci_benchmark,
)
from datasets.synthetic.generator import load_synthetic_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

ESCI_FULL_SAMPLE_SIZE = 10_000
ESCI_FULL_SAMPLE_SEED = 0

#-------------------------------------------------------------------------------

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms < 1e-10, 1e-10, norms)


def angular_pairwise_nn(vectors: np.ndarray) -> np.ndarray:
    """
    For each vector in vectors, find its nearest neighbor within the same set
    (excluding self).

    Args:
        vectors: (N, D) array; need not be normalized

    Returns:
        (N,) array of angular distances to the nearest other vector in the set
    """
    vn = _normalize(vectors)
    gram = vn @ vn.T
    np.fill_diagonal(gram, -2.0)
    dist_mat = np.arccos(np.clip(gram, -1.0, 1.0)) / np.pi
    return dist_mat.min(axis=1)


#-------------------------------------------------------------------------------

def load_rq_values(benchmark_dir: Path, dataset: str, cache_K: int) -> np.ndarray:
    """
    Load r_Q values (cache radii) from precomputed ground truth, if available.

    Returns:
        r_Q values as a 1-D array, or None if not available.
    """
    if dataset == "sift":
        gt_file = benchmark_dir / f"cache_gt_angular_K{cache_K}.npz"
        if not gt_file.exists():
            return None
        data = np.load(gt_file)
        rqs = []
        i = 0
        while f'distances_{i}' in data:
            dists = data[f'distances_{i}']
            rqs.append(float(dists[-1]))
            i += 1
        return np.array(rqs)

    if dataset == "esci":
        gt_file = benchmark_dir / f"cache_gt_K{cache_K}.npz"
        if not gt_file.exists():
            return None
        data = np.load(gt_file)
        rqs = []
        i = 0
        while f'distances_{i}' in data:
            dists = data[f'distances_{i}']
            rqs.append(float(dists[-1]))
            i += 1
        return np.array(rqs)

    return None

#-------------------------------------------------------------------------------

def _cdf(arr: np.ndarray):
    s = np.sort(arr)
    c = np.arange(1, len(s) + 1) / len(s)
    return s, c


def _configure_x_axis(ax):
    """Dense minor ticks, sparse major labels on [0, 1]."""
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.tick_params(axis='x', which='minor', length=3)


def plot_and_save(
    nn_dists: np.ndarray,
    rq_values: np.ndarray,
    title: str,
    label_nn: str,
    out_path: Path,
    sample_note: str = None,
):
    """Save CDF and histogram plots."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bins = np.linspace(0, 1, 51)

    # CDF
    ax = axes[0]
    sx, sy = _cdf(nn_dists)
    ax.plot(sx, sy, label=label_nn, color='steelblue')
    if rq_values is not None:
        sr, cr = _cdf(rq_values)
        ax.plot(sr, cr, label='r_Q (cache radius)', color='firebrick', linestyle=':')
    ax.set_xlabel('Angular Distance')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of Distances')
    ax.legend(fontsize=8)
    _configure_x_axis(ax)
    ax.set_ylim(0, 1.05)

    # Histogram
    ax2 = axes[1]
    ax2.hist(nn_dists, bins=bins, alpha=0.6, label=label_nn, color='steelblue', density=True)
    if rq_values is not None:
        ax2.hist(rq_values, bins=bins, alpha=0.4, label='r_Q', color='firebrick', density=True)
    ax2.set_xlabel('Angular Distance')
    ax2.set_ylabel('Density')
    ax2.set_title('Histogram of Distances')
    ax2.legend(fontsize=8)
    _configure_x_axis(ax2)

    note_lines = []
    if sample_note:
        note_lines.append(sample_note)
    note_lines.append(
        f'{label_nn} - median: {np.median(nn_dists):.4f}, '
        f'p10: {np.percentile(nn_dists, 10):.4f}, '
        f'p90: {np.percentile(nn_dists, 90):.4f}'
    )
    if rq_values is not None:
        note_lines.append(f'r_Q - median: {np.median(rq_values):.4f}')

    fig.suptitle(title, y=1.02)
    fig.text(0.5, -0.02, '  |  '.join(note_lines), ha='center', fontsize=7, color='gray')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot: {out_path}")

#-------------------------------------------------------------------------------

def _dist_stats(arr: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean":   float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p10":    float(np.percentile(arr, 10)),
        f"{prefix}_p90":    float(np.percentile(arr, 90)),
    }


def summarize(
    nn_dists: np.ndarray,
    rq_values: np.ndarray,
    extra: dict = None,
) -> dict:
    d = {"n_queries": int(len(nn_dists))}
    d.update(_dist_stats(nn_dists, "nn_dist"))
    if rq_values is not None:
        d.update(_dist_stats(rq_values, "rq"))
        d["frac_within_median_rq"] = float((nn_dists < d["rq_median"]).mean())
    if extra:
        d.update(extra)
    return d

#-------------------------------------------------------------------------------

def run_benchmark_mode(dataset: str, benchmark: str, output_dir: Path = None):
    """Characterize the combined cache+test query set from a prepared benchmark directory."""
    if dataset == "synthetic":
        benchmark_dir = Path("datasets/synthetic/data") / benchmark
        _, cache_q, test_q, config = load_synthetic_dataset(str(benchmark_dir))
        rq_values = None
        default_out_dir = Path("simulations/synthetic/processed") / benchmark
        dset_label = f"Synthetic ({benchmark})"

    elif dataset == "sift":
        benchmark_dir = Path("datasets/sift") / benchmark
        _, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        cache_K = config.get("cache_K", 100)
        rq_values = load_rq_values(benchmark_dir, "sift", cache_K)
        default_out_dir = Path("simulations/sift/processed") / benchmark
        dset_label = f"SIFT ({benchmark})"

    elif dataset == "esci":
        benchmark_dir = Path("datasets/esci") / benchmark
        cache_q, test_q, config = load_esci_benchmark(str(benchmark_dir))
        cache_K = config.get("cache_K", 99)
        rq_values = load_rq_values(benchmark_dir, "esci", cache_K)
        default_out_dir = Path("simulations/esci/processed") / benchmark
        dset_label = f"ESCI ({benchmark})"

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    out_dir = output_dir if output_dir is not None else default_out_dir
    stem = benchmark

    queries = np.vstack([cache_q, test_q])
    logger.info(f"Query set: {queries.shape} (cache {cache_q.shape[0]} + test {test_q.shape[0]})")

    nn_dists = angular_pairwise_nn(queries)

    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Query Set Similarity - {dset_label}"
    plot_and_save(
        nn_dists, rq_values, title,
        label_nn="NN dist within query set",
        out_path=out_dir / f"{stem}_query_similarity.png",
    )

    stats = summarize(
        nn_dists, rq_values,
        extra={"benchmark": benchmark, "dataset": dataset, "mode": "benchmark"},
    )
    with open(out_dir / f"{stem}_query_similarity.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats: {out_dir / f'{stem}_query_similarity.json'}")
    print(json.dumps(stats, indent=2))

#-------------------------------------------------------------------------------

def run_full_mode(dataset: str, output_dir: Path = None):
    """Characterize the full original query population."""
    if dataset == "sift":
        _, _, queries, _ = load_sift_dataset("datasets/sift")
        # queries: (10000, 128) - full pairwise is fast
        sample_note = None
        default_out_dir = Path("simulations/sift/processed/full_query_population")
        dset_label = "SIFT full query population (10K)"

    elif dataset == "esci":
        _, queries, _, _ = load_esci_data("datasets/esci", load_embeddings=False)
        # queries: (130652, 384) - full pairwise not feasible; sample
        rng = np.random.default_rng(ESCI_FULL_SAMPLE_SEED)
        idx = rng.choice(len(queries), size=ESCI_FULL_SAMPLE_SIZE, replace=False)
        idx.sort()
        queries = queries[idx]
        sample_note = f"Random sample: {ESCI_FULL_SAMPLE_SIZE:,} of 130,652 queries (seed={ESCI_FULL_SAMPLE_SEED})"
        default_out_dir = Path("simulations/esci/processed/full_query_population")
        dset_label = f"ESCI full query population (sample {ESCI_FULL_SAMPLE_SIZE:,}/130K)"

    elif dataset == "synthetic":
        raise ValueError("synthetic --full is not supported; use --benchmark with a set1 baseline config")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    out_dir = output_dir if output_dir is not None else default_out_dir
    stem = "full_population"

    logger.info(f"Full query set: {queries.shape[0]} vectors, dim={queries.shape[1]}")
    nn_dists = angular_pairwise_nn(queries)

    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"Query Population Similarity - {dset_label}"
    plot_and_save(
        nn_dists, None, title,
        label_nn="NN dist within query population",
        out_path=out_dir / f"{stem}_query_similarity.png",
        sample_note=sample_note,
    )

    extra = {"dataset": dataset, "mode": "full"}
    if sample_note:
        extra["sample_note"] = sample_note
        extra["sample_size"] = ESCI_FULL_SAMPLE_SIZE
        extra["sample_seed"] = ESCI_FULL_SAMPLE_SEED
    stats = summarize(nn_dists, None, extra=extra)
    with open(out_dir / f"{stem}_query_similarity.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats: {out_dir / f'{stem}_query_similarity.json'}")
    print(json.dumps(stats, indent=2))

#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Characterize intra-query-set similarity for SIFT, ESCI, or synthetic benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset', required=True,
        choices=['synthetic', 'sift', 'esci'],
        help='Dataset to characterize',
    )
    parser.add_argument(
        '--benchmarks', nargs='+', metavar='NAME', default=None,
        help='One or more benchmark directory names to characterize',
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Also characterize the full query population',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: simulations/<dataset>/processed/<timestamp>)',
    )

    args = parser.parse_args()

    if not args.benchmarks and not args.full:
        parser.error("at least one of --benchmarks or --full is required")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"simulations/{args.dataset}/processed/{ts}")

    for bench in (args.benchmarks or []):
        run_benchmark_mode(args.dataset, bench, output_dir=out_dir)

    if args.full:
        run_full_mode(args.dataset, output_dir=out_dir)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------
