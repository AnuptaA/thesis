#!/usr/bin/env python3
"""
Compile all analysis PNGs from a processed run into a single PDF,
with a header page for each section.
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------

SECTIONS = [
    ('global', 'Global / Cross-Set', [
        'accuracy_heatmap_global.png',
        'hit_rate_heatmap_global.png',
        'hit_rate_heatmap_per_set.png',
        'angular_cosine_validation.png',
        'lemma_breakdown_global.png',
        'time_relative_to_brute.png',
    ]),
    ('set1', 'Set 1 -- Baseline (K=100, N=20)', None),
    ('set2', 'Set 2 -- Perturbation Levels (K=100, N=20)', [
        'set2_hit_rate_heatmap_all.png',
        'set2_lemma_breakdown.png',
        'set2_angle_hist_consolidated.png',
        'set2_angle_hist_zoomed.png',
    ]),
    ('set3', 'Set 3 -- K/N Ratios', [
        'set3_kn_heatmap_combined_euclidean.png',
        'set3_kn_heatmap_combined_angular.png',
        'set3_kn_ratio_matrix_euclidean.png',
        'set3_kn_ratio_matrix_angular.png',
    ]),
    ('set4', 'Set 4 -- Variable N (K=100)', [
        'set4_hit_rate_by_var_level.png',
        'set4_dist_calcs_by_var_level.png',
        'set4_hit_rate_vs_n_value.png',
    ]),
    ('set5', 'Set 5 -- Union Effectiveness (K=100, N=20)', None),
    ('set6', 'Set 6 -- Cache Size Variation (K=100, N=20)', None),
]

# ---------------------------------------------------------------------------

def _default_processed_dir() -> Path:
    base = Path(__file__).parent.parent.parent / 'simulations' / 'synthetic' / 'processed'
    stamped = sorted(
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r'\d{8}_\d{6}', d.name)
    )
    if stamped:
        chosen = stamped[-1]
        print(f"Auto-selected run: {chosen}")
        return chosen
    raise FileNotFoundError(f"No timestamped run directories found in {base}")


def add_header_page(pdf: PdfPages, title: str, run_name: str):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('#1a1a2e')
    ax.axis('off')
    ax.text(0.5, 0.6, title, transform=ax.transAxes,
            ha='center', va='center', fontsize=28, color='white', fontweight='bold',
            wrap=True)
    ax.text(0.5, 0.38, f'Run: {run_name}', transform=ax.transAxes,
            ha='center', va='center', fontsize=13, color='#aaaacc')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def add_image_page(pdf: PdfPages, img_path: Path):
    img = mpimg.imread(str(img_path))
    h, w = img.shape[:2]
    # fit to letter landscape (11 x 8.5 in at 100 dpi)
    page_w, page_h = 11.0, 8.5
    scale = min(page_w / (w / 100), page_h / (h / 100))
    fig_w = (w / 100) * scale
    fig_h = (h / 100) * scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(img_path.name, fontsize=7, color='#555555', pad=3)
    plt.subplots_adjust(left=0, right=1, top=0.97, bottom=0)
    pdf.savefig(fig, bbox_inches='tight', dpi=150)
    plt.close(fig)


def compile(processed_dir: Path, out_path: Path):
    run_name = processed_dir.name

    print(f"Compiling PDF from: {processed_dir}")
    print(f"Output: {out_path}")

    with PdfPages(str(out_path)) as pdf:
        # cover page
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#0f3460')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor('#0f3460')
        ax.axis('off')
        ax.text(0.5, 0.62, 'Synthetic Simulation Analysis', transform=ax.transAxes,
                ha='center', va='center', fontsize=34, color='white', fontweight='bold')
        ax.text(0.5, 0.50, f'Run: {run_name}', transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='#b0c4de')
        ax.text(0.5, 0.40, datetime.now().strftime('%B %d, %Y'), transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='#b0c4de')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        for subdir, section_title, ordered_files in SECTIONS:
            section_dir = processed_dir / subdir
            if not section_dir.exists():
                print(f"  Skipping {subdir}/ (not found)")
                continue

            # resolve file list: use explicit order if given, else sort
            if ordered_files:
                files = []
                for name in ordered_files:
                    p = section_dir / name
                    if p.exists():
                        files.append(p)
                    # also pick up any extra PNGs not in the explicit list
                extra = sorted(
                    p for p in section_dir.glob('*.png')
                    if p.name not in ordered_files
                )
                files.extend(extra)
            else:
                files = sorted(section_dir.glob('*.png'))

            if not files:
                print(f"  Skipping {subdir}/ (no PNGs)")
                continue

            print(f"  [{subdir}] {len(files)} image(s)")
            add_header_page(pdf, section_title, run_name)
            for img_path in files:
                add_image_page(pdf, img_path)

    print(f"\nDone: {out_path}  ({out_path.stat().st_size / 1_000_000:.1f} MB)")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile analysis PNGs into a PDF')
    parser.add_argument('--processed-dir', default=None,
                        help='Path to processed run dir (default: latest timestamped)')
    parser.add_argument('--out', default=None,
                        help='Output PDF path (default: <processed_dir>/analysis.pdf)')
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir) if args.processed_dir else _default_processed_dir()
    out_path = Path(args.out) if args.out else processed_dir / 'analysis.pdf'

    compile(processed_dir, out_path)
