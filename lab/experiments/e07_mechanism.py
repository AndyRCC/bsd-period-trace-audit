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
    partial_correlation_bins,
    partial_correlation_cov,
    covariance_in_bins,
)
from lab.gp import bad_euler_factor

SEED = 20261121
N_PRIMES = 25

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "07-mechanism-h8"
BAD_EULER_PATH = DERIVED_DIR / "bad_euler.csv"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def _build_or_load_bad_euler(classes: pd.DataFrame) -> pd.DataFrame:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    total = len(classes)
    if BAD_EULER_PATH.exists():
        df_ex = pd.read_csv(BAD_EULER_PATH)
        if len(df_ex) == total and list(df_ex.columns) == ["label", "log_B", "n_bad"]:
            return df_ex

    curves = [(row["label"], row["ainvs"]) for _, row in classes.iterrows()]
    parts: list[pd.DataFrame] = []
    for start in range(0, total, MAX_BATCH_SIZE):
        end = min(start + MAX_BATCH_SIZE, total)
        df_batch = bad_euler_factor(curves[start:end])
        parts.append(df_batch)

    df_total = pd.concat(parts, ignore_index=True)
    df_total.to_csv(BAD_EULER_PATH, index=False)
    return df_total


def run_mechanism_h8() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)

    t_eb0 = time.perf_counter()
    df_eb = _build_or_load_bad_euler(classes)
    t_eb1 = time.perf_counter()

    classes = classes.merge(df_eb, on="label", how="left")
    classes["log_disc"] = [
        math.log(abs(discriminant(ai))) for ai in classes["ainvs"]
    ]

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    rows_desc: list[dict[str, str | int | float]] = []
    rows_col: list[dict[str, str | int | float]] = []

    rng_placebo = np.random.default_rng(SEED)

    for rank in (0, 1):
        for win_name, n_min, n_max in WINDOWS:
            sub = classes.loc[
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ].reset_index(drop=True)

            bins_L = pd.qcut(
                sub["l_value"], q=100, labels=False, duplicates="drop"
            ).to_numpy(dtype=int)

            omega = sub["omega"].to_numpy(dtype=np.float64)
            log_omega = np.log(omega)
            log_L = np.log(sub["l_value"].to_numpy(dtype=np.float64))
            log_reg = (
                np.zeros(len(sub), dtype=np.float64)
                if rank == 0
                else np.log(sub["regulator"].to_numpy(dtype=np.float64))
            )
            log_tam = np.log(sub["tamagawa"].to_numpy(dtype=np.float64))
            log_sha = np.log(sub["sha_an"].to_numpy(dtype=np.float64))
            log_tor = np.log(sub["torsion"].to_numpy(dtype=np.float64))
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            z_placebo = rng_placebo.normal(0.0, 1.0, size=len(sub))

            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)
            good_sub = (cond_sub[:, None] % primes[None, :]) != 0

            for j, p_prime in enumerate(primes):
                p_int = int(p_prime)
                mask_b = good_sub[:, j]
                n_b = int(np.sum(mask_b))
                ap_p = ap_sub[mask_b, j].astype(np.float64)
                b_L = bins_L[mask_b]

                # Part A: exact decomposition of Cov(ap, log Ω | L)
                cov_total = covariance_in_bins(ap_p, log_omega[mask_b], b_L)
                cov_L = covariance_in_bins(ap_p, log_L[mask_b], b_L)
                cov_reg = (
                    0.0
                    if rank == 0
                    else -covariance_in_bins(ap_p, log_reg[mask_b], b_L)
                )
                cov_tam = -covariance_in_bins(ap_p, log_tam[mask_b], b_L)
                cov_sha = -covariance_in_bins(ap_p, log_sha[mask_b], b_L)
                cov_tor = 2.0 * covariance_in_bins(ap_p, log_tor[mask_b], b_L)

                acc = cov_L + cov_reg + cov_tam + cov_sha + cov_tor
                error_id = abs(cov_total - acc)
                if error_id > 1e-6:
                    raise RuntimeError(
                        f"Identity failure at rank={rank}, window={win_name}, p={p_int}: "
                        f"error_identity={error_id:.3e} > 1e-6"
                    )

                rows_desc.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n": n_b,
                        "cov_total": cov_total,
                        "cov_L": cov_L,
                        "cov_reg": cov_reg,
                        "cov_tam": cov_tam,
                        "cov_sha": cov_sha,
                        "cov_tor": cov_tor,
                        "acc": acc,
                        "error_identity": error_id,
                        "pct_L": 100.0 * cov_L / cov_total,
                        "pct_reg": 100.0 * cov_reg / cov_total,
                        "pct_tam": 100.0 * cov_tam / cov_total,
                        "pct_sha": 100.0 * cov_sha / cov_total,
                        "pct_tor": 100.0 * cov_tor / cov_total,
                    }
                )

                # Part B: collider test
                rho_omega, n_ef = partial_correlation_bins(
                    ap_p, omega[mask_b], b_L
                )
                rho_logomega, _ = partial_correlation_bins(
                    ap_p, log_omega[mask_b], b_L
                )
                rho_logB, _ = partial_correlation_cov(
                    ap_p, log_omega[mask_b], b_L, log_B[mask_b]
                )
                rho_logtam, _ = partial_correlation_cov(
                    ap_p, log_omega[mask_b], b_L, log_tam[mask_b]
                )
                rho_logdisc, _ = partial_correlation_cov(
                    ap_p, log_omega[mask_b], b_L, log_disc[mask_b]
                )
                rho_placebo, _ = partial_correlation_cov(
                    ap_p, log_omega[mask_b], b_L, z_placebo[mask_b]
                )
                rho_ap_logB, _ = partial_correlation_bins(
                    ap_p, log_B[mask_b], b_L
                )
                se_val = float(1.0 / np.sqrt(n_ef)) if n_ef > 0 else float("nan")

                rows_col.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n": n_b,
                        "rho_omega": rho_omega,
                        "rho_logomega": rho_logomega,
                        "rho_logB": rho_logB,
                        "rho_logtam": rho_logtam,
                        "rho_logdisc": rho_logdisc,
                        "rho_placebo": rho_placebo,
                        "rho_ap_logB": rho_ap_logB,
                        "se": se_val,
                    }
                )

    df_desc = pd.DataFrame(rows_desc)
    df_desc.to_csv(OUTPUT_DIR / "decomposition.csv", index=False)

    df_col = pd.DataFrame(rows_col)
    df_col.to_csv(OUTPUT_DIR / "collider.csv", index=False)

    # Figure mechanism.png: two panels (rank 0 and rank 1 at N < 50 000)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150, sharey=True)

    series_cfg = [
        ("rho_logomega", "o-", "#1f5a99", "No covariate: ρ(aₚ, log Ω | L)"),
        ("rho_logB", "s--", "#e67e22", "Controlling Z = log B"),
        ("rho_logtam", "^-", "#c0392b", "Controlling Z = log ∏c"),
        ("rho_logdisc", "d-", "#27ae60", "Controlling Z = log |Δ|"),
        ("rho_placebo", "x:", "#7f8c8d", "Controlling Z = placebo noise"),
    ]

    for ax, rank, title in [
        (ax0, 0, "A. Rank 0 (N < 50 000): ρ(aₚ, log Ω | L, Z)"),
        (ax1, 1, "B. Rank 1 (N < 50 000): ρ(aₚ, log Ω | L′(1), Z)"),
    ]:
        sub_c = df_col.loc[
            (df_col["rank"] == rank) & (df_col["window"] == "1-49999")
        ].reset_index(drop=True)
        for col_y, fmt, color, label in series_cfg:
            ax.plot(
                sub_c["p"],
                sub_c[col_y],
                fmt,
                color=color,
                markersize=4.5,
                linewidth=1.5,
                label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Prime p (p ≤ 97)")
        ax.set_ylabel("Partial correlation ρ")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Effect of candidate covariates on ρ(aₚ, log Ω | L) at N < 50 000 (one curve per class)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mechanism.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"Time for the cache bad_euler.csv: {t_eb1 - t_eb0:.2f} s")
    print(f"Total time e07_mechanism: {t_end - t_start:.2f} s")
    print(f"Maximum error_identity over the 100 cases: {df_desc['error_identity'].max():.3e}")

    print("\n--- DECOMPOSITION (p in {2, 3, 5, 7}) ---")
    print(
        df_desc.loc[df_desc["p"].isin([2, 3, 5, 7])][
            [
                "rank",
                "window",
                "p",
                "cov_total",
                "pct_L",
                "pct_reg",
                "pct_tam",
                "pct_sha",
                "pct_tor",
                "error_identity",
            ]
        ].to_string(index=False)
    )

    print("\n--- COLLIDER (p in {2, 3, 5, 7}) ---")
    print(
        df_col.loc[df_col["p"].isin([2, 3, 5, 7])][
            [
                "rank",
                "window",
                "p",
                "rho_omega",
                "rho_logomega",
                "rho_logB",
                "rho_logtam",
                "rho_logdisc",
                "rho_placebo",
                "rho_ap_logB",
                "se",
            ]
        ].to_string(index=False)
    )

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_mechanism_h8()
