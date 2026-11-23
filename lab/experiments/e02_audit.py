from __future__ import annotations

import math
from pathlib import Path
import time
import numpy as np
import pandas as pd

from lab.data import load_bsd, load_generators
from lab.gp import pari_invariants

SEED = 20261121
N_SAMPLE = 1000
N_RANK3 = 20
REAL_TOL = 1e-6

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "02-data-audit"


def _nearest_square(x: float) -> tuple[int, float]:
    if not np.isfinite(x) or x <= 0:
        return 0, float("nan")
    root = max(1, int(round(math.sqrt(x))))
    best_c = root * root
    best_d = abs(x - best_c)
    for k in (max(1, root - 1), root + 1):
        c = k * k
        d = abs(x - c)
        if d < best_d:
            best_c = c
            best_d = d
    return best_c, best_d


def run_audit() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd()
    df_gens = load_generators()
    df_full = df_bsd.merge(df_gens, on="label", how="inner")

    rng = np.random.default_rng(SEED)
    indices_1000 = np.sort(rng.choice(len(df_full), size=N_SAMPLE, replace=False))
    sample_1000 = df_full.iloc[indices_1000]

    labels_seen = set(sample_1000["label"])
    candidates_r3 = df_full.loc[
        (df_full["rank"] == 3) & (~df_full["label"].isin(labels_seen))
    ].reset_index(drop=True)

    rng_r3 = np.random.default_rng(SEED)
    indices_r3 = np.sort(rng_r3.choice(len(candidates_r3), size=N_RANK3, replace=False))
    sample_r3 = candidates_r3.iloc[indices_r3]

    sample = pd.concat([sample_1000, sample_r3], ignore_index=True)

    batch_curves = [
        (row["label"], row["ainvs"], row["gens"])
        for _, row in sample.iterrows()
    ]

    t_gp0 = time.perf_counter()
    df_pari = pari_invariants(batch_curves)
    t_gp1 = time.perf_counter()

    comp = pd.DataFrame(
        {
            "label": sample["label"],
            "conductor": sample["conductor"],
            "rank_table": sample["rank"],
            "rank_pari": df_pari["rank_an"],
            "l_value_table": sample["l_value"],
            "l_value_pari": df_pari["l_value"],
            "omega_table": sample["omega"],
            "omega_pari": df_pari["omega"],
            "tamagawa_table": sample["tamagawa"],
            "tamagawa_pari": df_pari["tamagawa"],
            "torsion_table": sample["torsion"],
            "torsion_pari": df_pari["torsion"],
            "regulator_table": sample["regulator"],
            "regulator_pari": df_pari["regulator"],
            "sha_an_table": sample["sha_an"],
            "root_number_pari": df_pari["root_number"],
        }
    )

    comp["matches_rank"] = comp["rank_table"] == comp["rank_pari"]
    comp["matches_tamagawa"] = comp["tamagawa_table"] == comp["tamagawa_pari"]
    comp["matches_torsion"] = comp["torsion_table"] == comp["torsion_pari"]

    comp["err_rel_l_value"] = (
        np.abs(comp["l_value_pari"] - comp["l_value_table"]) / comp["l_value_table"]
    )
    comp["matches_l_value"] = comp["err_rel_l_value"] <= REAL_TOL

    comp["err_rel_omega"] = (
        np.abs(comp["omega_pari"] - comp["omega_table"]) / comp["omega_table"]
    )
    comp["matches_omega"] = comp["err_rel_omega"] <= REAL_TOL

    comp["err_rel_regulator"] = (
        np.abs(comp["regulator_pari"] - comp["regulator_table"]) / comp["regulator_table"]
    )
    comp["matches_regulator"] = comp["err_rel_regulator"] <= REAL_TOL

    comp["matches_parity"] = ((-1) ** comp["rank_pari"]) == comp["root_number_pari"]

    denom = comp["omega_pari"] * comp["regulator_pari"] * comp["tamagawa_pari"]
    comp["sha_calc"] = comp["l_value_pari"] * (comp["torsion_pari"] ** 2) / denom

    sq_list = [_nearest_square(v) for v in comp["sha_calc"]]
    comp["nearest_square"] = [c for c, _ in sq_list]
    comp["dist_perfect_square"] = [d for _, d in sq_list]

    path_csv = OUTPUT_DIR / "comparison.csv"
    comp.to_csv(path_csv, index=False)

    t_end = time.perf_counter()

    summary = {
        "n_sample": len(comp),
        "time_total_s": t_end - t_start,
        "time_gp_s": t_gp1 - t_gp0,
        "by_rank": comp["rank_table"].value_counts().sort_index().to_dict(),
        "pct_rank": 100.0 * float(comp["matches_rank"].mean()),
        "pct_l_value": 100.0 * float(comp["matches_l_value"].mean()),
        "pct_omega": 100.0 * float(comp["matches_omega"].mean()),
        "pct_tamagawa": 100.0 * float(comp["matches_tamagawa"].mean()),
        "pct_torsion": 100.0 * float(comp["matches_torsion"].mean()),
        "pct_regulator": 100.0 * float(comp["matches_regulator"].mean()),
        "pct_parity": 100.0 * float(comp["matches_parity"].mean()),
    }

    print(f"Sample: {summary['n_sample']} curves (1000 at random + 20 of rank 3, seed={SEED})")
    print(f"Total time: {summary['time_total_s']:.2f} s (GP batch 1020: {summary['time_gp_s']:.2f} s)")
    print(f"Distribution by rank in the sample: {summary['by_rank']}")
    print(
        f"Matches (%): rank={summary['pct_rank']:.1f}%, "
        f"l_value={summary['pct_l_value']:.1f}%, omega={summary['pct_omega']:.1f}%, "
        f"tamagawa={summary['pct_tamagawa']:.1f}%, torsion={summary['pct_torsion']:.1f}%, "
        f"regulator={summary['pct_regulator']:.1f}%, parity={summary['pct_parity']:.1f}%"
    )

    for col_err, name in [
        ("err_rel_l_value", "l_value"),
        ("err_rel_omega", "omega"),
        ("err_rel_regulator", "regulator"),
    ]:
        idx_max = int(comp[col_err].idxmax())
        row_max = comp.iloc[idx_max]
        print(
            f"Worst discrepancy {name}: label={row_max['label']}, "
            f"err_rel={row_max[col_err]:.3e}"
        )

    print(
        f"sha_calc valid in {comp['dist_perfect_square'].notna().sum()}/{len(comp)} curves; "
        f"max dist to a perfect square: {comp['dist_perfect_square'].max():.3e}"
    )
    return summary


if __name__ == "__main__":
    run_audit()
