from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.gp import ap_coefficients, ap_averages

SEED = 20261121
N_PRIMES_MAIN = 1000
N_SHUFFLES = 100
P_MAX_OUT = 20_360
N_BINS_X = 50
X_MAX = 0.9

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "04-murmurations"


def control_shuffled(
    matrix_ap: np.ndarray,
    ranks: np.ndarray,
    n_shuffled: int = N_SHUFFLES,
    seed: int = SEED,
) -> np.ndarray:
    ap = np.asarray(matrix_ap, dtype=np.float64)
    r = np.asarray(ranks)
    n0 = int(np.sum(r == 0))
    n1 = int(np.sum(r == 1))
    if n0 == 0 or n1 == 0:
        raise ValueError("Curves of rank 0 and rank 1 are required")

    sum_total = ap.sum(axis=0)
    rng = np.random.default_rng(seed)
    s_nulls = np.empty(n_shuffled, dtype=np.float64)

    for b in range(n_shuffled):
        perm = rng.permutation(r)
        sum0 = ap[perm == 0].sum(axis=0)
        f0 = sum0 / n0
        f1 = (sum_total - sum0) / n1
        d = f0 - f1
        s_nulls[b] = float(np.sqrt(np.mean(d * d)))

    return s_nulls


def _bin_in_x(
    primes: np.ndarray,
    values: np.ndarray,
    n_mean: float,
    n_bins: int = N_BINS_X,
    x_max: float = X_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    x = primes / float(n_mean)
    bounds = np.linspace(0.0, x_max, n_bins + 1)
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    means = np.full(n_bins, np.nan, dtype=np.float64)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x >= bounds[i]) & (x < bounds[i + 1])
        else:
            mask = (x >= bounds[i]) & (x <= bounds[i + 1])
        if np.any(mask):
            means[i] = float(np.mean(values[mask]))

    return centers, means


def run_murmurations() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=30_000)
    classes = one_per_class(df_bsd)

    # 1. Main range [7500, 10000], ranks 0 and 1 only
    sub_main = classes.loc[
        (classes["conductor"] >= 7_500)
        & (classes["conductor"] <= 10_000)
        & (classes["rank"].isin([0, 1]))
    ].reset_index(drop=True)

    n0_main = int((sub_main["rank"] == 0).sum())
    n1_main = int((sub_main["rank"] == 1).sum())
    n_mean_main = float(sub_main["conductor"].mean())

    curves_main_ap = [
        (row["label"], row["ainvs"]) for _, row in sub_main.iterrows()
    ]
    t_ap0 = time.perf_counter()
    df_ap = ap_coefficients(curves_main_ap, n_primes=N_PRIMES_MAIN)
    t_ap1 = time.perf_counter()

    primes_main = np.array(df_ap.columns, dtype=np.int32)
    matrix_ap = df_ap.to_numpy(dtype=np.int16)
    ranks_main = sub_main["rank"].to_numpy(dtype=np.int8)
    conductors_main = sub_main["conductor"].to_numpy(dtype=np.int32)

    # Main version: all primes
    ap_f64 = matrix_ap.astype(np.float64)
    f0_all = ap_f64[ranks_main == 0].mean(axis=0)
    f1_all = ap_f64[ranks_main == 1].mean(axis=0)
    d_all = f0_all - f1_all
    s_real_all = float(np.sqrt(np.mean(d_all * d_all)))
    corr_f0_f1_all = float(np.corrcoef(f0_all, f1_all)[0, 1])

    # Variant: without the primes that divide the conductor
    good = (conductors_main[:, None] % primes_main[None, :]) != 0
    mask0 = good[ranks_main == 0]
    mask1 = good[ranks_main == 1]
    n0_good = mask0.sum(axis=0)
    n1_good = mask1.sum(axis=0)
    sum0_good = np.where(mask0, ap_f64[ranks_main == 0], 0.0).sum(axis=0)
    sum1_good = np.where(mask1, ap_f64[ranks_main == 1], 0.0).sum(axis=0)
    f0_good_only = sum0_good / n0_good
    f1_good_only = sum1_good / n1_good
    d_good_only = f0_good_only - f1_good_only
    s_real_good_only = float(np.sqrt(np.mean(d_good_only * d_good_only)))
    corr_f0_f1_good_only = float(np.corrcoef(f0_good_only, f1_good_only)[0, 1])

    # 2. Control: 100 shuffles with seed 20261121
    s_nulls = control_shuffled(
        matrix_ap, ranks_main, n_shuffled=N_SHUFFLES, seed=SEED
    )
    mean_nulls = float(np.mean(s_nulls))
    sd_nulls = float(np.std(s_nulls, ddof=1))
    max_nulls = float(np.max(s_nulls))
    z_score = (s_real_all - mean_nulls) / sd_nulls

    # Also compute null correlations between f0 and f1, as a description
    rng_corr = np.random.default_rng(SEED)
    sum_tot = ap_f64.sum(axis=0)
    corrs_null = np.empty(N_SHUFFLES, dtype=np.float64)
    for b in range(N_SHUFFLES):
        perm = rng_corr.permutation(ranks_main)
        s0 = ap_f64[perm == 0].sum(axis=0)
        f0_b = s0 / n0_main
        f1_b = (sum_tot - s0) / n1_main
        corrs_null[b] = float(np.corrcoef(f0_b, f1_b)[0, 1])

    df_nulls = pd.DataFrame(
        {
            "iteration": np.arange(1, N_SHUFFLES + 1),
            "s_null": s_nulls,
            "corr_f0_f1_null": corrs_null,
        }
    )
    df_nulls.to_csv(OUTPUT_DIR / "nulls.csv", index=False)

    # 3. Out of sample: time 500 curves and extrapolate
    sub_out_cand = classes.loc[
        (classes["conductor"] >= 20_000)
        & (classes["conductor"] <= 25_000)
        & (classes["rank"].isin([0, 1]))
    ].reset_index(drop=True)

    batch_bench = [
        (row["label"], row["ainvs"], int(row["rank"]))
        for _, row in sub_out_cand.iloc[:500].iterrows()
    ]
    t_b0 = time.perf_counter()
    _ = ap_averages(batch_bench, p_max=P_MAX_OUT, exclude_bad=False)
    t_b1 = time.perf_counter()
    t_bench_500 = t_b1 - t_b0
    t_extrapolated = t_bench_500 * (len(sub_out_cand) / 500.0)

    if t_extrapolated > 600.0:
        n_max_out = 22_500
        out_rank_label = "20000-22500"
        sub_out = classes.loc[
            (classes["conductor"] >= 20_000)
            & (classes["conductor"] <= 22_500)
            & (classes["rank"].isin([0, 1]))
        ].reset_index(drop=True)
    else:
        n_max_out = 25_000
        out_rank_label = "20000-25000"
        sub_out = sub_out_cand

    n0_out = int((sub_out["rank"] == 0).sum())
    n1_out = int((sub_out["rank"] == 1).sum())
    n_mean_out = float(sub_out["conductor"].mean())

    batch_out = [
        (row["label"], row["ainvs"], int(row["rank"]))
        for _, row in sub_out.iterrows()
    ]
    t_f0 = time.perf_counter()
    df_avg_out = ap_averages(batch_out, p_max=P_MAX_OUT, exclude_bad=False)
    t_f1 = time.perf_counter()
    t_gp_out = t_f1 - t_f0

    f0_out_df = df_avg_out.loc[df_avg_out["group"] == 0].sort_values("p")
    f1_out_df = df_avg_out.loc[df_avg_out["group"] == 1].sort_values("p")
    primes_out = f0_out_df["p"].to_numpy(dtype=np.int32)
    f0_out = f0_out_df["mean"].to_numpy(dtype=np.float64)
    f1_out = f1_out_df["mean"].to_numpy(dtype=np.float64)
    d_out = f0_out - f1_out
    s_out = float(np.sqrt(np.mean(d_out * d_out)))

    # 4. Save averages.csv: columns rank_N, variant, group, p, n, mean
    rows_avg: list[pd.DataFrame] = []
    for g, n_arr, f_arr in [
        (0, np.full(len(primes_main), n0_main, dtype=int), f0_all),
        (1, np.full(len(primes_main), n1_main, dtype=int), f1_all),
    ]:
        rows_avg.append(
            pd.DataFrame(
                {
                    "rank_N": "7500-10000",
                    "variant": "all",
                    "group": g,
                    "p": primes_main,
                    "n": n_arr,
                    "mean": f_arr,
                }
            )
        )
    for g, n_arr, f_arr in [
        (0, n0_good, f0_good_only),
        (1, n1_good, f1_good_only),
    ]:
        rows_avg.append(
            pd.DataFrame(
                {
                    "rank_N": "7500-10000",
                    "variant": "good_only",
                    "group": g,
                    "p": primes_main,
                    "n": n_arr,
                    "mean": f_arr,
                }
            )
        )
    df_out_table = df_avg_out[["group", "p", "n", "mean"]].copy()
    df_out_table.insert(0, "variant", "all")
    df_out_table.insert(0, "rank_N", out_rank_label)
    rows_avg.append(df_out_table)

    df_averages_total = pd.concat(rows_avg, ignore_index=True)
    df_averages_total.to_csv(OUTPUT_DIR / "averages.csv", index=False)

    # 5. Binning into 50 intervals of x = p / N_mean in [0, 0.9]
    centers_x, d_bin_main_raw = _bin_in_x(primes_main, d_all, n_mean_main)
    _, d_bin_out_raw = _bin_in_x(primes_out, d_out, n_mean_out)

    d_norm_main = d_all / np.sqrt(primes_main)
    d_norm_out = d_out / np.sqrt(primes_out)
    _, d_bin_main_norm = _bin_in_x(primes_main, d_norm_main, n_mean_main)
    _, d_bin_out_norm = _bin_in_x(primes_out, d_norm_out, n_mean_out)

    mask_bins_raw = np.isfinite(d_bin_main_raw) & np.isfinite(d_bin_out_raw)
    corr_scale_raw = float(
        np.corrcoef(
            d_bin_main_raw[mask_bins_raw], d_bin_out_raw[mask_bins_raw]
        )[0, 1]
    )
    mask_bins_norm = np.isfinite(d_bin_main_norm) & np.isfinite(d_bin_out_norm)
    corr_scale_norm = float(
        np.corrcoef(
            d_bin_main_norm[mask_bins_norm], d_bin_out_norm[mask_bins_norm]
        )[0, 1]
    )

    # 6. Figures
    # Figure 1: murmurations.png
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150, sharey=True)
    ax1.scatter(primes_main, f0_all, s=8, alpha=0.55, color="#1f5a99", label=f"Rank 0 (n={n0_main})")
    ax1.scatter(primes_main, f1_all, s=8, alpha=0.55, color="#c0392b", label=f"Rank 1 (n={n1_main})")
    ax1.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Prime p (first 1 000 primes, p ≤ 7 919)")
    ax1.set_ylabel("Average of aₚ")
    ax1.set_title(f"A. All primes (S = {s_real_all:.4f})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.scatter(primes_main, f0_good_only, s=8, alpha=0.55, color="#1f5a99", label="Rank 0 (p ∤ N)")
    ax2.scatter(primes_main, f1_good_only, s=8, alpha=0.55, color="#c0392b", label="Rank 1 (p ∤ N)")
    ax2.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Prime p (first 1 000 primes, p ≤ 7 919)")
    ax2.set_title(f"B. Excluding primes p | N (S = {s_real_good_only:.4f})")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    fig1.suptitle(
        "Murmurations of elliptic curves for N ∈ [7 500, 10 000] (one curve per class)",
        fontsize=12,
        fontweight="bold",
    )
    fig1.tight_layout()
    fig1.savefig(OUTPUT_DIR / "murmurations.png", bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: control.png
    fig2, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    ax.hist(
        s_nulls,
        bins=15,
        color="#7f8c8d",
        edgecolor="#2c3e50",
        alpha=0.8,
        label=f"Null distribution (100 shuffles, μ={mean_nulls:.4f}, σ={sd_nulls:.4f})",
    )
    ax.axvline(
        s_real_all,
        color="#c0392b",
        linewidth=2.2,
        linestyle="-",
        label=f"Observed S = {s_real_all:.4f} (z = {z_score:.2f})",
    )
    ax.set_xlabel("Statistic S = √(mean of (f₀(p) − f₁(p))²)")
    ax.set_ylabel("Frequency (out of 100 shuffles)")
    ax.set_title("Control by permuting the rank labels in N ∈ [7 500, 10 000]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper center")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "control.png", bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: scale.png
    fig3, (ax_b, ax_n) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)
    ax_b.plot(
        centers_x,
        d_bin_main_raw,
        "o-",
        color="#1f5a99",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [7 500, 10 000] (N̄={n_mean_main:.0f})",
    )
    ax_b.plot(
        centers_x,
        d_bin_out_raw,
        "s-",
        color="#d35400",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [{out_rank_label.replace('-', ', ')}] (N̄={n_mean_out:.0f})",
    )
    ax_b.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_b.set_xlabel("Dimensionless scale x = p / N̄")
    ax_b.set_ylabel("Bin average of D(p) = f₀(p) − f₁(p)")
    ax_b.set_title(f"A. Raw D(p) (r = {corr_scale_raw:.4f})")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left")

    ax_n.plot(
        centers_x,
        d_bin_main_norm,
        "o-",
        color="#1f5a99",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [7 500, 10 000] (N̄={n_mean_main:.0f})",
    )
    ax_n.plot(
        centers_x,
        d_bin_out_norm,
        "s-",
        color="#d35400",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [{out_rank_label.replace('-', ', ')}] (N̄={n_mean_out:.0f})",
    )
    ax_n.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_n.set_xlabel("Dimensionless scale x = p / N̄")
    ax_n.set_ylabel("Bin average of D(p) / √p")
    ax_n.set_title(f"B. Normalized D(p) / √p (r = {corr_scale_norm:.4f})")
    ax_n.grid(True, linestyle=":", alpha=0.6)
    ax_n.legend(loc="upper left")

    fig3.suptitle(
        "Scale invariance of murmurations in 50 intervals of x = p / N̄ ∈ [0, 0.9]",
        fontsize=12,
        fontweight="bold",
    )
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "scale.png", bbox_inches="tight")
    plt.close(fig3)

    t_end = time.perf_counter()

    summary = {
        "n0_main": n0_main,
        "n1_main": n1_main,
        "n_mean_main": n_mean_main,
        "t_ap_main_s": t_ap1 - t_ap0,
        "s_real_all": s_real_all,
        "s_real_good_only": s_real_good_only,
        "corr_f0_f1_all": corr_f0_f1_all,
        "corr_f0_f1_good_only": corr_f0_f1_good_only,
        "mean_nulls": mean_nulls,
        "sd_nulls": sd_nulls,
        "max_nulls": max_nulls,
        "z_score": z_score,
        "corr_null_mean": float(np.mean(corrs_null)),
        "t_bench_500_s": t_bench_500,
        "t_extrapolated_s": t_extrapolated,
        "out_rank_label": out_rank_label,
        "n0_out": n0_out,
        "n1_out": n1_out,
        "n_mean_out": n_mean_out,
        "n_primes_out": len(primes_out),
        "p_last_out": int(primes_out[-1]),
        "t_gp_out_s": t_gp_out,
        "s_out": s_out,
        "corr_scale_raw": corr_scale_raw,
        "corr_scale_norm": corr_scale_norm,
        "t_total_s": t_end - t_start,
    }

    for k, v in summary.items():
        print(f"{k}: {v}")
    return summary


if __name__ == "__main__":
    run_murmurations()
