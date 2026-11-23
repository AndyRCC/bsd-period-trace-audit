from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import DERIVED_DIR, load_ap_cache, discriminant
from lab.stats import (
    partial_correlation_bins,
    partial_correlation_cov,
    covariance_in_bins,
)

N_PRIMES = 25
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "08-rank1"
BAD_EULER_PATH = DERIVED_DIR / "bad_euler.csv"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def run_rank1() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)

    if not BAD_EULER_PATH.exists():
        raise FileNotFoundError(f"Missing cache {BAD_EULER_PATH}")
    df_eb = pd.read_csv(BAD_EULER_PATH)
    classes = classes.merge(df_eb, on="label", how="left")

    classes["log_disc"] = [
        math.log(abs(discriminant(ai))) for ai in classes["ainvs"]
    ]
    classes["log_N"] = np.log(classes["conductor"].to_numpy(dtype=np.float64))
    classes["sigma"] = classes["log_disc"] / classes["log_N"]

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    rows_cov: list[dict[str, str | int | float]] = []
    rows_dir: list[dict[str, str | int | float]] = []
    rows_desc_r2: list[dict[str, str | int | float]] = []

    for rank in (0, 1, 2):
        for win_name, n_min, n_max in WINDOWS:
            sub = classes.loc[
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ].reset_index(drop=True)

            bins_L = pd.qcut(
                sub["l_value"], q=100, labels=False, duplicates="drop"
            ).to_numpy(dtype=int)

            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
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
            sigma = sub["sigma"].to_numpy(dtype=np.float64)
            log_omega_over_R = log_omega - log_reg

            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)
            good_sub = (cond_sub[:, None] % primes[None, :]) != 0

            sets_a: list[tuple[str, np.ndarray | None]] = [
                ("none", None),
                ("log_B", log_B),
                ("log_tam", log_tam),
                ("log_disc", log_disc),
                ("sigma", sigma),
                ("log_B+log_tam", np.column_stack([log_B, log_tam])),
                ("log_B+log_disc", np.column_stack([log_B, log_disc])),
                ("log_tam+log_disc", np.column_stack([log_tam, log_disc])),
                (
                    "log_B+log_tam+log_disc",
                    np.column_stack([log_B, log_tam, log_disc]),
                ),
            ]

            Z0_full = np.column_stack([log_tam, log_tor, log_sha])
            sets_b: list[tuple[str, np.ndarray]] = [
                ("Z0", Z0_full),
                ("Z0+log_disc", np.column_stack([Z0_full, log_disc])),
                ("Z0+log_B", np.column_stack([Z0_full, log_B])),
                (
                    "Z0+log_disc+log_B",
                    np.column_stack([Z0_full, log_disc, log_B]),
                ),
            ]

            for j, p_prime in enumerate(primes):
                p_int = int(p_prime)
                mask_b = good_sub[:, j]
                n_b = int(np.sum(mask_b))
                ap_p = ap_sub[mask_b, j].astype(np.float64)
                b_L = bins_L[mask_b]
                log_om_b = log_omega[mask_b]

                # Part A: covariates
                for cset_name, z_mat in sets_a:
                    if z_mat is None:
                        rho_val, n_ef = partial_correlation_bins(
                            ap_p, log_om_b, b_L
                        )
                    else:
                        z_sub = z_mat[mask_b]
                        rho_val, n_ef = partial_correlation_cov(
                            ap_p, log_om_b, b_L, z_sub
                        )
                    se_val = (
                        float(1.0 / np.sqrt(n_ef)) if n_ef > 0 else float("nan")
                    )
                    rows_cov.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "p": p_int,
                            "n": n_b,
                            "covariate_set": cset_name,
                            "rho": rho_val,
                            "se": se_val,
                        }
                    )

                # Part B: free direction (ranks 1 and 2)
                if rank in (1, 2):
                    log_om_r_b = log_omega_over_R[mask_b]
                    for cset_name_b, z_mat_b in sets_b:
                        rho_b, n_ef_b = partial_correlation_cov(
                            ap_p, log_om_r_b, b_L, z_mat_b[mask_b]
                        )
                        se_b = (
                            float(1.0 / np.sqrt(n_ef_b))
                            if n_ef_b > 0
                            else float("nan")
                        )
                        rows_dir.append(
                            {
                                "rank": rank,
                                "window": win_name,
                                "p": p_int,
                                "n": n_b,
                                "covariate_set": cset_name_b,
                                "rho": rho_b,
                                "se": se_b,
                            }
                        )

                    # Check of the size mechanism: ρ(log|Δ|, log(Ω/R) | L, Z0)
                    rho_tam_mech, n_ef_mech = partial_correlation_cov(
                        log_disc[mask_b], log_om_r_b, b_L, Z0_full[mask_b]
                    )
                    se_mech = (
                        float(1.0 / np.sqrt(n_ef_mech))
                        if n_ef_mech > 0
                        else float("nan")
                    )
                    rows_dir.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "p": p_int,
                            "n": n_b,
                            "covariate_set": "log_disc_vs_log_omega_R|Z0",
                            "rho": rho_tam_mech,
                            "se": se_mech,
                        }
                    )

                # Part C: exact decomposition at rank 2
                if rank == 2:
                    cov_total = covariance_in_bins(ap_p, log_om_b, b_L)
                    cov_L = covariance_in_bins(ap_p, log_L[mask_b], b_L)
                    cov_reg = -covariance_in_bins(ap_p, log_reg[mask_b], b_L)
                    cov_tam = -covariance_in_bins(ap_p, log_tam[mask_b], b_L)
                    cov_sha = -covariance_in_bins(ap_p, log_sha[mask_b], b_L)
                    cov_tor = 2.0 * covariance_in_bins(ap_p, log_tor[mask_b], b_L)

                    acc = cov_L + cov_reg + cov_tam + cov_sha + cov_tor
                    error_id = abs(cov_total - acc)
                    if error_id > 1e-6:
                        raise RuntimeError(
                            f"Identity failure at rank 2, window={win_name}, p={p_int}: {error_id}"
                        )
                    rows_desc_r2.append(
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

    df_cov = pd.DataFrame(rows_cov)
    df_cov.to_csv(OUTPUT_DIR / "covariates.csv", index=False)

    df_dir = pd.DataFrame(rows_dir)
    df_dir.to_csv(OUTPUT_DIR / "free_direction.csv", index=False)

    df_desc_r2 = pd.DataFrame(rows_desc_r2)
    df_desc_r2.to_csv(OUTPUT_DIR / "decomposition_r2.csv", index=False)

    # Figure rank1.png: for rank 1 and N < 50 000, ρ(aₚ, log Ω | L') against p for each covariate set
    sub_fig = df_cov.loc[
        (df_cov["rank"] == 1) & (df_cov["window"] == "1-49999")
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)

    styles = [
        ("none", "o-", "#1f5a99", 2.0, "No covariates"),
        ("log_B", "s--", "#e67e22", 1.5, "{log B}"),
        ("log_tam", "^-", "#c0392b", 1.6, "{log ∏c}"),
        ("log_disc", "d-", "#27ae60", 1.6, "{log |Δ|}"),
        ("sigma", "v:", "#16a085", 1.4, "{σ = log|Δ|/log N}"),
        ("log_B+log_tam", "P--", "#8e44ad", 1.4, "{log B, log ∏c}"),
        ("log_B+log_disc", "X--", "#d35400", 1.4, "{log B, log |Δ|}"),
        ("log_tam+log_disc", "*:", "#2c3e50", 1.5, "{log ∏c, log |Δ|}"),
        (
            "log_B+log_tam+log_disc",
            "h-",
            "#111111",
            2.2,
            "{log B, log ∏c, log |Δ|}",
        ),
    ]

    for cset, fmt, col, lw, label in styles:
        sc = sub_fig.loc[sub_fig["covariate_set"] == cset].sort_values("p")
        ax.plot(
            sc["p"],
            sc["rho"],
            fmt,
            color=col,
            linewidth=lw,
            markersize=5,
            label=label,
        )

    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prime p (first 25 primes, p ≤ 97)")
    ax.set_ylabel("Partial correlation ρ(aₚ, log Ω | L′(1), Z)")
    ax.set_title(
        "Rank 1, N < 50 000: ρ(aₚ, log Ω | L′(1), Z) for the 9 covariate sets"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", ncol=2, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rank1.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"Total time e08_rank1: {t_end - t_start:.2f} s")

    print("\n--- PART A: COVARIATES AT RANK 1 (p in {3, 5, 7}) ---")
    piv_a = df_cov.loc[
        (df_cov["rank"] == 1) & (df_cov["p"].isin([3, 5, 7]))
    ].pivot(index=["window", "p"], columns="covariate_set", values="rho")
    print(piv_a.to_string())

    print("\n--- PART B: FREE DIRECTION AT RANKS 1 AND 2 (p in {3, 5, 7}) ---")
    piv_b = df_dir.loc[df_dir["p"].isin([3, 5, 7])].pivot(
        index=["rank", "window", "p"], columns="covariate_set", values="rho"
    )
    print(piv_b.to_string())

    print("\n--- PART C: RANK-2 DECOMPOSITION (p in {2, 3, 5, 7}) ---")
    print(
        df_desc_r2.loc[df_desc_r2["p"].isin([2, 3, 5, 7])][
            [
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

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_rank1()
