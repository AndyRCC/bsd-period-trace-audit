from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import DERIVED_DIR, MAX_BATCH_SIZE, load_ap_cache, discriminant
from lab.stats import (
    stratified_bins,
    stratum_key,
    partial_correlation_bins,
    partial_correlation_cov,
)
from lab.gp import reduction_types

N_PRIMES = 25
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "10-closing-h16"
BAD_EULER_PATH = DERIVED_DIR / "bad_euler.csv"
REDUCTION_TYPES_PATH = DERIVED_DIR / "reduction_types.csv"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]
TARGET_PRIMES = (3, 5, 7)


def build_or_load_reduction_types(classes: pd.DataFrame) -> pd.DataFrame:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    total = len(classes)
    cols_exp = ["label", "n_split", "n_no_split", "n_additive"]
    if REDUCTION_TYPES_PATH.exists():
        df_ex = pd.read_csv(REDUCTION_TYPES_PATH)
        if len(df_ex) == total and list(df_ex.columns) == cols_exp:
            return df_ex

    curves = [(row["label"], row["ainvs"]) for _, row in classes.iterrows()]
    parts: list[pd.DataFrame] = []
    for start in range(0, total, MAX_BATCH_SIZE):
        end = min(start + MAX_BATCH_SIZE, total)
        t0 = time.perf_counter()
        df_batch = reduction_types(curves[start:end])
        dt = time.perf_counter() - t0
        print(
            f"  reduction_types batch {start}-{end} ({end - start} curves): {dt:.2f} s",
            flush=True,
        )
        parts.append(df_batch)

    df_total = pd.concat(parts, ignore_index=True)
    tmp = REDUCTION_TYPES_PATH.with_suffix(".tmp.csv")
    df_total.to_csv(tmp, index=False)
    tmp.replace(REDUCTION_TYPES_PATH)
    return df_total


def run_h16() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)

    if not BAD_EULER_PATH.exists():
        raise FileNotFoundError(f"Missing required cache: {BAD_EULER_PATH}")
    df_eb = pd.read_csv(BAD_EULER_PATH)
    if len(df_eb) != len(classes):
        raise RuntimeError(
            f"bad_euler.csv has {len(df_eb)} rows, expected {len(classes)}"
        )

    t_tr0 = time.perf_counter()
    df_tr = build_or_load_reduction_types(classes)
    t_tr1 = time.perf_counter()
    print(f"Cache reduction_types ready at {t_tr1 - t_tr0:.2f} s", flush=True)

    classes = classes.merge(df_eb, on="label", how="left")
    classes = classes.merge(df_tr, on="label", how="left")

    # Exact sanity check: n_split + n_no_split + n_additive == n_bad in every class
    sum_types = (
        classes["n_split"].to_numpy(dtype=int)
        + classes["n_no_split"].to_numpy(dtype=int)
        + classes["n_additive"].to_numpy(dtype=int)
    )
    n_bad = classes["n_bad"].to_numpy(dtype=int)
    disagreements = int(np.sum(sum_types != n_bad))
    if disagreements > 0:
        raise RuntimeError(
            f"Inconsistency between reduction_types and bad_euler in {disagreements} classes"
        )
    print(
        f"Check reduction_types vs bad_euler: 0 disagreements in {len(classes)} classes.",
        flush=True,
    )

    classes["log_disc"] = [
        math.log(abs(discriminant(ai))) for ai in classes["ainvs"]
    ]
    classes["log_omega"] = np.log(classes["omega"].to_numpy(dtype=np.float64))

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primes = {int(p): j for j, p in enumerate(primes)}

    rows_p12: list[dict[str, str | int | float]] = []
    rows_chain: list[dict[str, str | int | float]] = []

    for rank in (1, 2):
        for win_name, n_min, n_max in WINDOWS:
            mask_rv = (
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = classes.loc[mask_rv].reset_index(drop=True)

            l_val = sub["l_value"].to_numpy(dtype=np.float64)
            log_om = sub["log_omega"].to_numpy(dtype=np.float64)
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            cond = sub["conductor"].to_numpy(dtype=np.int32)
            cp = sub["tamagawa"].to_numpy(dtype=np.int64)
            T = sub["torsion"].to_numpy(dtype=np.int64)
            sha = sub["sha_an"].to_numpy(dtype=np.float64)
            n_split = sub["n_split"].to_numpy(dtype=np.int64)
            n_nonsplit = sub["n_no_split"].to_numpy(dtype=np.int64)

            # Quintiles of B computed within each (rank, window)
            q_B = np.asarray(
                pd.qcut(log_B, q=5, labels=False, duplicates="drop"),
                dtype=np.int64,
            )
            n_qB_bins = int(len(np.unique(q_B)))
            print(
                f"Rank {rank}, window {win_name}: n={len(sub)}, B quintiles={n_qB_bins}, counts={np.bincount(q_B).tolist()}",
                flush=True,
            )

            # ---- P14: within strata, given L (4 cases of rank x window) ----
            strat_rv = stratum_key(cp, T, sha)
            bins_rv = stratified_bins(l_val, strat_rv, q_max=100, n_min=200)
            m_rv = bins_rv >= 0

            rho_B_disc, nef_B_disc = partial_correlation_bins(
                log_B[m_rv], log_disc[m_rv], bins_rv[m_rv]
            )
            se_B_disc = float(1.0 / np.sqrt(nef_B_disc))
            rows_chain.append(
                {
                    "test": "P14",
                    "rank": rank,
                    "window": win_name,
                    "p": "all",
                    "even": "logB_vs_logdisc",
                    "n": int(np.sum(m_rv)),
                    "rho": rho_B_disc,
                    "se": se_B_disc,
                    "z": rho_B_disc / se_B_disc,
                }
            )

            n_mult = n_split + n_nonsplit
            m_mult = n_mult > 0
            frac_split = n_split[m_mult].astype(np.float64) / n_mult[m_mult].astype(
                np.float64
            )
            bins_mult = stratified_bins(
                l_val[m_mult], strat_rv[m_mult], q_max=100, n_min=200
            )
            m_mult_v = bins_mult >= 0
            rho_fp_disc, nef_fp_disc = partial_correlation_bins(
                frac_split[m_mult_v],
                log_disc[m_mult][m_mult_v],
                bins_mult[m_mult_v],
            )
            se_fp_disc = float(1.0 / np.sqrt(nef_fp_disc))
            rows_chain.append(
                {
                    "test": "P14",
                    "rank": rank,
                    "window": win_name,
                    "p": "all",
                    "even": "frac_split_vs_logdisc",
                    "n": int(np.sum(m_mult_v)),
                    "rho": rho_fp_disc,
                    "se": se_fp_disc,
                    "z": rho_fp_disc / se_fp_disc,
                }
            )

            # ---- P12 and P13: per prime p in {3, 5, 7}, p ∤ N only ----
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)

            for p_int in TARGET_PRIMES:
                j = idx_primes[p_int]
                mask_b = (cond % p_int) != 0
                ap_b = ap_sub[mask_b, j].astype(np.float64)
                l_b = l_val[mask_b]
                log_om_b = log_om[mask_b]
                log_B_b = log_B[mask_b]
                log_disc_b = log_disc[mask_b]
                cp_b = cp[mask_b]
                T_b = T[mask_b]
                sha_b = sha[mask_b]
                q_B_b = q_B[mask_b]

                # Exact strata (Task 9)
                strat_base = stratum_key(cp_b, T_b, sha_b)
                bins_base = stratified_bins(
                    l_b, strat_base, q_max=100, n_min=200
                )
                m_base = bins_base >= 0
                rho_strat, nef_strat = partial_correlation_bins(
                    ap_b[m_base], log_om_b[m_base], bins_base[m_base]
                )
                se_strat = float(1.0 / np.sqrt(nef_strat))

                # Exact strata x quintile of B
                strat_qB = stratum_key(cp_b, T_b, sha_b, extra=q_B_b)
                bins_qB = stratified_bins(
                    l_b, strat_qB, q_max=100, n_min=200
                )
                m_qB = bins_qB >= 0
                rho_qB, nef_qB = partial_correlation_bins(
                    ap_b[m_qB], log_om_b[m_qB], bins_qB[m_qB]
                )
                se_qB = float(1.0 / np.sqrt(nef_qB))

                # Exact strata + linear log B
                rho_logB, nef_logB = partial_correlation_cov(
                    ap_b[m_base],
                    log_om_b[m_base],
                    bins_base[m_base],
                    log_B_b[m_base],
                )
                se_logB = float(1.0 / np.sqrt(nef_logB))

                # Exact strata + linear log|Δ|
                rho_logdisc, nef_logdisc = partial_correlation_cov(
                    ap_b[m_base],
                    log_om_b[m_base],
                    bins_base[m_base],
                    log_disc_b[m_base],
                )
                se_logdisc = float(1.0 / np.sqrt(nef_logdisc))

                pct_red_qB = 100.0 * (1.0 - abs(rho_qB) / abs(rho_strat))
                pct_red_logB = 100.0 * (1.0 - abs(rho_logB) / abs(rho_strat))
                pct_red_logdisc = 100.0 * (
                    1.0 - abs(rho_logdisc) / abs(rho_strat)
                )

                rows_p12.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n_stratum": int(np.sum(m_base)),
                        "rho_stratum": rho_strat,
                        "se_stratum": se_strat,
                        "n_qB": int(np.sum(m_qB)),
                        "rho_qB": rho_qB,
                        "se_qB": se_qB,
                        "pct_red_qB": pct_red_qB,
                        "rho_logB": rho_logB,
                        "se_logB": se_logB,
                        "pct_red_logB": pct_red_logB,
                        "rho_logdisc": rho_logdisc,
                        "se_logdisc": se_logdisc,
                        "pct_red_logdisc": pct_red_logdisc,
                    }
                )

                # P13: within strata and given L: ρ(ap, log B) and ρ(log B, log Ω)
                rho_ap_B, nef_ap_B = partial_correlation_bins(
                    ap_b[m_base], log_B_b[m_base], bins_base[m_base]
                )
                se_ap_B = float(1.0 / np.sqrt(nef_ap_B))
                rows_chain.append(
                    {
                        "test": "P13",
                        "rank": rank,
                        "window": win_name,
                        "p": str(p_int),
                        "even": "ap_vs_logB",
                        "n": int(np.sum(m_base)),
                        "rho": rho_ap_B,
                        "se": se_ap_B,
                        "z": rho_ap_B / se_ap_B,
                    }
                )

                rho_B_om, nef_B_om = partial_correlation_bins(
                    log_B_b[m_base], log_om_b[m_base], bins_base[m_base]
                )
                se_B_om = float(1.0 / np.sqrt(nef_B_om))
                rows_chain.append(
                    {
                        "test": "P13",
                        "rank": rank,
                        "window": win_name,
                        "p": str(p_int),
                        "even": "logB_vs_logomega",
                        "n": int(np.sum(m_base)),
                        "rho": rho_B_om,
                        "se": se_B_om,
                        "z": rho_B_om / se_B_om,
                    }
                )

    df_p12 = pd.DataFrame(rows_p12)
    df_p12.to_csv(OUTPUT_DIR / "p12_B_strata.csv", index=False)

    df_chain = pd.DataFrame(rows_chain)
    df_chain.to_csv(OUTPUT_DIR / "p13_p14_chain.csv", index=False)

    # Figure h16.png: ranks 1 and 2, p = 3, 5, 7, the 4 series with ±2 SE bars
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150, sharey=True)
    series_plot = [
        ("rho_stratum", "se_stratum", "o-", "#c0392b", -0.21, "Exact strata (H16 baseline)"),
        ("rho_qB", "se_qB", "s-", "#2980b9", -0.07, "Strata × quintiles of B"),
        ("rho_logB", "se_logB", "^--", "#16a085", 0.07, "Strata + linear log B"),
        ("rho_logdisc", "se_logdisc", "D-", "#8e44ad", 0.21, "Strata + linear log|Δ|"),
    ]
    labels_x = [
        "p=3\n(1–50k)",
        "p=5\n(1–50k)",
        "p=7\n(1–50k)",
        "p=3\n(50k–100k)",
        "p=5\n(50k–100k)",
        "p=7\n(50k–100k)",
    ]
    pairs_xp = [
        ("1-49999", 3),
        ("1-49999", 5),
        ("1-49999", 7),
        ("50000-99999", 3),
        ("50000-99999", 5),
        ("50000-99999", 7),
    ]
    x_base = np.arange(len(pairs_xp), dtype=float)

    for ax, rank, title in [
        (axes[0], 1, "A. Rank 1: ρ(aₚ, log Ω | L′(1), stratum) ± 2 SE"),
        (axes[1], 2, "B. Rank 2: ρ(aₚ, log Ω | L″(1)/2, stratum) ± 2 SE"),
    ]:
        sub_r = df_p12.loc[df_p12["rank"] == rank]
        for col_rho, col_se, fmt, color, dx, label in series_plot:
            ys = []
            errs = []
            for win, p_val in pairs_xp:
                row = sub_r.loc[
                    (sub_r["window"] == win) & (sub_r["p"] == p_val)
                ].iloc[0]
                ys.append(float(row[col_rho]))
                errs.append(2.0 * float(row[col_se]))
            ax.errorbar(
                x_base + dx,
                ys,
                yerr=errs,
                fmt=fmt,
                color=color,
                ecolor=color,
                elinewidth=1.25,
                capsize=3.2,
                markersize=5.6,
                linewidth=1.5,
                label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.axvline(2.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels_x)
        ax.set_ylabel("Partial correlation ρ(aₚ, log Ω | L, stratum)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.6)

    fig.suptitle(
        "Closing H16 (P12): control by quintiles of B, linear log B and linear log|Δ| within exact strata",
        fontsize=11.8,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "h16.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"\nTotal time e10_h16: {t_end - t_start:.2f} s")

    print("\n--- P12: EXACT STRATA VS B AND log|Δ| ---")
    print(df_p12.to_string(index=False))

    print("\n--- P13 AND P14: LINKS OF THE H17 CHAIN ---")
    print(df_chain.to_string(index=False))

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_h16()
