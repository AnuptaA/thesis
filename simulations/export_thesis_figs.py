#!/usr/bin/env python3
# One-time script: run all analyses and export thesis figures to thesis_figs/.
# Excludes combined_no_union (algorithm 6) from all plots.
# Run from the project root: python simulations/export_thesis_figs.py

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).parent.parent.resolve()
# Synthetic analyze() uses CWD-relative paths, so pin CWD to project root.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd

import simulations.synthetic.analyze_synthetic as syn
import simulations.sift.analyze_sift as sift
import simulations.esci.analyze_esci as esci

#-------------------------------------------------------------------------------

THESIS_FIGS = ROOT / "thesis_figs"

TEMP_OUT = {
    "synthetic": ROOT / "simulations/synthetic/processed/_thesis_export",
    "sift": ROOT / "simulations/sift/processed/_thesis_export",
    "esci": ROOT / "simulations/esci/processed/_thesis_export",
}

# (workload_key, source_relative_to_temp_out, dest_subdir_relative_to_thesis_figs)
COPY_MAP: List[Tuple[str, str, str]] = [
    # Synthetic global
    ("synthetic", "global/accuracy_heatmap_global.png",              "synthetic/global"),
    ("synthetic", "global/angular_cosine_validation.png",            "synthetic/global"),
    ("synthetic", "global/avg_time_by_algorithm.png",                "synthetic/global"),
    ("synthetic", "global/lemma_breakdown_global.png",               "synthetic/global"),
    # Synthetic set1
    ("synthetic", "set1/set1_distance_calcs.png",                    "synthetic/set1"),
    ("synthetic", "set1/set1_hit_rate_heatmap.png",                  "synthetic/set1"),
    # Synthetic set2
    ("synthetic", "set2/set2_hit_rate_heatmap_all.png",              "synthetic/set2"),
    ("synthetic", "set2/set2_angle_hist_zoomed.png",                 "synthetic/set2"),
    ("synthetic", "set2/set2_angle_hist_consolidated.png",           "synthetic/set2"),
    ("synthetic", "set2/set2_lemma_breakdown.png",                   "synthetic/set2"),
    # Synthetic set3
    ("synthetic", "set3/set3_kn_heatmap_combined_euclidean.png",     "synthetic/set3"),
    ("synthetic", "set3/set3_kn_heatmap_combined_angular.png",       "synthetic/set3"),
    ("synthetic", "set3/set3_kn_ratio_matrix_euclidean.png",         "synthetic/set3"),
    ("synthetic", "set3/set3_kn_ratio_matrix_angular.png",           "synthetic/set3"),
    # Synthetic set4
    ("synthetic", "set4/set4_hit_rate_vs_n_value.png",               "synthetic/set4"),
    ("synthetic", "set4/set4_distance_calcs.png",                    "synthetic/set4"),
    # Synthetic set5 (combined row excluded via patch below)
    ("synthetic", "set5/set5_hit_rate_vs_kn_ratio_angular.png",      "synthetic/set5"),
    ("synthetic", "set5/set5_hit_rate_vs_kn_ratio_euclidean.png",    "synthetic/set5"),
    # Synthetic set6
    ("synthetic", "set6/set6_hit_rate_heatmap_euclidean.png",        "synthetic/set6"),
    ("synthetic", "set6/set6_hit_rate_heatmap_angular.png",          "synthetic/set6"),
    ("synthetic", "set6/set6_distance_calcs_vs_cache_size.png",      "synthetic/set6"),
    ("synthetic", "set6/set6_coverage_and_hit_rate.png",             "synthetic/set6"),
    # SIFT global
    ("sift", "global/hit_rate_heatmap_global.png",                   "sift/global"),
    # SIFT cache_scaling -> set2
    ("sift", "cache_scaling/set2_coverage_and_hit_rate.png",         "sift/set2"),
    ("sift", "cache_scaling/distance_calcs_vs_cache_size.png",       "sift/set2"),
    # ESCI global
    ("esci", "global/hit_rate_heatmap_global.png",                   "esci/global"),
    # ESCI set2
    ("esci", "set2/set2_coverage_and_hit_rate.png",                  "esci/set2"),
    ("esci", "set2/distance_calcs_vs_cache_size.png",                "esci/set2"),
]

#-------------------------------------------------------------------------------

def _apply_patches() -> None:
    """Patch synthetic analyze functions to exclude combined_no_union from all plots."""

    # plot_set5: builds union_pairs from ['lemma1', 'lemma2', 'combined'].
    # Filter both combined and combined_no_union so only lemma1/lemma2 rows appear.
    _orig_set5 = syn.plot_set5

    def _patched_set5(df: pd.DataFrame, outdir: Path) -> None:
        df_filtered = df[~df["algorithm"].isin({"combined", "combined_no_union"})]
        _orig_set5(df_filtered, outdir)

    syn.plot_set5 = _patched_set5

    # plot_global_accuracy_heatmap: passes include_no_union=True, so combined_no_union
    # appears. Filter it from the df before the function sees it.
    _orig_acc = syn.plot_global_accuracy_heatmap

    def _patched_acc(df: pd.DataFrame, outdir: Path) -> None:
        _orig_acc(df[df["algorithm"] != "combined_no_union"], outdir)

    syn.plot_global_accuracy_heatmap = _patched_acc

    # plot_angular_validation: also passes include_no_union=True. It reads from
    # summaries dicts; filter combined_no_union out of each dict's validations.
    _orig_val = syn.plot_angular_validation

    def _patched_val(summaries: list, outdir: Path) -> None:
        filtered = []
        for ds in summaries:
            ds2 = dict(ds)
            ds2["validations"] = {
                k: v for k, v in ds.get("validations", {}).items()
                if not k.startswith("combined_no_union")
            }
            filtered.append(ds2)
        _orig_val(filtered, outdir)

    syn.plot_angular_validation = _patched_val

#-------------------------------------------------------------------------------

def _copy_files() -> None:
    """Copy figures listed in COPY_MAP from temp outputs to thesis_figs/."""
    missing = []
    for workload, src_rel, dest_subdir in COPY_MAP:
        src = TEMP_OUT[workload] / src_rel
        dest_dir = THESIS_FIGS / dest_subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)
            print(f"  copied {workload}/{src_rel}")
        else:
            missing.append(f"{workload}/{src_rel}")

    if missing:
        print("\nWARNING: the following expected files were not found:")
        for m in missing:
            print(f"  {m}")

#-------------------------------------------------------------------------------

def main() -> None:
    _apply_patches()

    # --- Synthetic ---
    syn_raw = syn._default_raw_dir()
    # analyze() writes to simulations/synthetic/processed/<out_name> relative to CWD
    syn.analyze(raw_dir=syn_raw, out_name="_thesis_export")

    # --- SIFT ---
    sift_raw = sift._default_raw_dir()
    sift.analyze(raw_dir=sift_raw, output_dir=str(TEMP_OUT["sift"]))

    # --- ESCI ---
    esci_raw = esci._default_raw_dir()
    esci.analyze(raw_dir=esci_raw, output_dir=str(TEMP_OUT["esci"]))

    # --- Copy to thesis_figs/ ---
    print(f"\nCopying figures to {THESIS_FIGS} ...")
    _copy_files()

    print(f"\nDone. Files written to {THESIS_FIGS}")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
