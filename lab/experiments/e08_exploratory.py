from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import DERIVED_DIR, load_ap_cache, discriminant
from lab.stats import partial_correlation_bins, partial_correlation_cov

N_PRIMES = 25
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "results" / "08-rank1" / "exploratory"
BAD_EULER_PATH = DERIVED_DIR / "bad_euler.csv"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def run_exploratory() -> dict:
    t_start = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)
    df_eb = pd.read_csv(BAD_EULER_PATH)
    classes = classes.merge(df_eb, on="label", how="left")
    classes["log_disc"] = [
        math.log(abs(discriminant(ai))) for ai in classes["ainvs"]
    ]

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    # E1: with L vs without L (conditioning only on 100 quantiles of N)
    rows_e1: list[dict[str, str | int | float]] = []
    # E2: pure subpopulation (Tamagawa=1, Torsion=1, Sha=1)
    rows_e2: list[dict[str, str | int | float]] = []
    # E3: parity of ap (p+1-ap even vs odd) against log ∏c in curves with T=1
    rows_e3: list[dict[str, str | int | float]] = []

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
            bins_N = pd.qcut(
                sub["conductor"], q=100, labels=False, duplicates="drop"
            ).to_numpy(dtype=int)

            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
            log_L = np.log(sub["l_value"].to_numpy(dtype=np.float64))
            log_reg = (
                np.zeros(len(sub), dtype=np.float64)
                if rank == 0
                else np.log(sub["regulator"].to_numpy(dtype=np.float64))
            )
            log_tam = np.log(sub["tamagawa"].to_numpy(dtype=np.float64))
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            tam_int = sub["tamagawa"].to_numpy(dtype=int)
            tor_int = sub["torsion"].to_numpy(dtype=int)
            sha_int = np.rint(sub["sha_an"].to_numpy(dtype=float)).astype(int)

            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)
            good_sub = (cond_sub[:, None] % primes[None, :]) != 0

            mask_pure = (tam_int == 1) & (tor_int == 1) & (sha_int == 1)
            mask_t1_c1 = (tam_int == 1) & (tor_int == 1)
            mask_t1 = tor_int == 1

            for j, p_prime in enumerate(primes):
                p_int = int(p_prime)
                if p_int not in (2, 3, 5, 7, 11, 13):
                    continue
                mask_b = good_sub[:, j]
                ap_p = ap_sub[mask_b, j].astype(np.float64)
                b_L = bins_L[mask_b]
                b_N = bins_N[mask_b]

                # E1
                rho_om_L, n_ef_L = partial_correlation_bins(
                    ap_p, log_omega[mask_b], b_L
                )
                rho_om_N, n_ef_N = partial_correlation_bins(
                    ap_p, log_omega[mask_b], b_N
                )
                rho_L_N, _ = partial_correlation_bins(ap_p, log_L[mask_b], b_N)
                rho_tam_N, _ = partial_correlation_bins(
                    ap_p, log_tam[mask_b], b_N
                )
                rho_tam_L, _ = partial_correlation_bins(
                    ap_p, log_tam[mask_b], b_L
                )
                rho_reg_N = (
                    0.0
                    if rank == 0
                    else partial_correlation_bins(ap_p, log_reg[mask_b], b_N)[0]
                )
                rho_reg_L = (
                    0.0
                    if rank == 0
                    else partial_correlation_bins(ap_p, log_reg[mask_b], b_L)[0]
                )

                rows_e1.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n": int(np.sum(mask_b)),
                        "rho_omega_given_N": rho_om_N,
                        "rho_omega_given_L": rho_om_L,
                        "rho_L_given_N": rho_L_N,
                        "rho_tam_given_N": rho_tam_N,
                        "rho_tam_given_L": rho_tam_L,
                        "rho_reg_given_N": rho_reg_N,
                        "rho_reg_given_L": rho_reg_L,
                        "se": float(1.0 / np.sqrt(n_ef_L)),
                    }
                )

                # E2: pure subpopulation (∏c=1, T=1, Sha=1)
                m_p = mask_b & mask_pure
                n_pure = int(np.sum(m_p))
                if n_pure > 200:
                    b_L_pure = pd.qcut(
                        sub.loc[m_p, "l_value"],
                        q=min(100, n_pure // 20),
                        labels=False,
                        duplicates="drop",
                    ).to_numpy(dtype=int)
                    ap_pure = ap_sub[m_p, j].astype(np.float64)
                    rho_pure, nef_p = partial_correlation_bins(
                        ap_pure, log_omega[m_p], b_L_pure
                    )
                    rho_pure_disc, _ = partial_correlation_cov(
                        ap_pure, log_omega[m_p], b_L_pure, log_disc[m_p]
                    )
                    z_pure = float(rho_pure * np.sqrt(nef_p))

                    # Also for (∏c=1, T=1) with Sha free
                    m_tc = mask_b & mask_t1_c1
                    b_L_tc = pd.qcut(
                        sub.loc[m_tc, "l_value"],
                        q=min(100, int(np.sum(m_tc)) // 20),
                        labels=False,
                        duplicates="drop",
                    ).to_numpy(dtype=int)
                    rho_tc, nef_tc = partial_correlation_bins(
                        ap_sub[m_tc, j].astype(np.float64),
                        log_omega[m_tc],
                        b_L_tc,
                    )
                    z_tc = float(rho_tc * np.sqrt(nef_tc))

                    rows_e2.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "p": p_int,
                            "n_pure": n_pure,
                            "rho_pure_given_L": rho_pure,
                            "z_pure": z_pure,
                            "rho_pure_given_L_logdisc": rho_pure_disc,
                            "n_t1_c1": int(np.sum(m_tc)),
                            "rho_t1_c1_given_L": rho_tc,
                            "z_t1_c1": z_tc,
                        }
                    )

                # E3: parity of ap in curves with T=1 (odd p)
                if p_int > 2:
                    m_t = mask_b & mask_t1
                    ap_t = ap_sub[m_t, j]
                    tam_t = log_tam[m_t]
                    is_even = (ap_t % 2) == 0
                    n_even = int(np.sum(is_even))
                    n_odd = int(np.sum(~is_even))
                    if n_even > 50 and n_odd > 50:
                        m_tam_even = float(np.mean(tam_t[is_even]))
                        m_tam_odd = float(np.mean(tam_t[~is_even]))
                        v_even = float(np.var(tam_t[is_even], ddof=1))
                        v_odd = float(np.var(tam_t[~is_even], ddof=1))
                        se_diff = math.sqrt(v_even / n_even + v_odd / n_odd)
                        delta_tam = m_tam_even - m_tam_odd
                        z_tam = delta_tam / se_diff if se_diff > 0 else 0.0

                        rows_e3.append(
                            {
                                "rank": rank,
                                "window": win_name,
                                "p": p_int,
                                "n_T1": int(np.sum(m_t)),
                                "n_ap_even": n_even,
                                "n_ap_odd": n_odd,
                                "mean_ap_even": float(np.mean(ap_t[is_even])),
                                "mean_ap_odd": float(np.mean(ap_t[~is_even])),
                                "mean_logtam_ap_even": m_tam_even,
                                "mean_logtam_ap_odd": m_tam_odd,
                                "delta_logtam_even_minus_odd": delta_tam,
                                "z_welch": z_tam,
                            }
                        )

    df_e1 = pd.DataFrame(rows_e1)
    df_e1.to_csv(EXP_DIR / "e1_collider_vs_intrinsic.csv", index=False)

    df_e2 = pd.DataFrame(rows_e2)
    df_e2.to_csv(EXP_DIR / "e2_pure_subpopulation.csv", index=False)

    df_e3 = pd.DataFrame(rows_e3)
    df_e3.to_csv(EXP_DIR / "e3_ap_parity_tamagawa.csv", index=False)

    # Save a copy of the exploratory code in results/08-rank1/exploratory/
    this_path = Path(__file__).resolve()
    (EXP_DIR / "e08_exploratory.py").write_text(
        this_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Exploratory figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)

    # Panel A: E1 at rank 0 and rank 1 (N < 50 000): ρ(ap, log Ω | N) vs ρ(ap, log Ω | L)
    for r, col, mk in [(0, "#1f5a99", "o"), (1, "#c0392b", "s")]:
        s1 = df_e1.loc[(df_e1["rank"] == r) & (df_e1["window"] == "1-49999")]
        ax1.plot(
            s1["p"],
            s1["rho_omega_given_N"],
            f"{mk}--",
            color=col,
            alpha=0.65,
            label=f"Rank {r}: ρ(aₚ, log Ω | N) [L not fixed]",
        )
        ax1.plot(
            s1["p"],
            s1["rho_omega_given_L"],
            f"{mk}-",
            color=col,
            linewidth=1.8,
            label=f"Rank {r}: ρ(aₚ, log Ω | L)",
        )
    ax1.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("Prime p")
    ax1.set_ylabel("Partial correlation ρ")
    ax1.set_title("E1. Effect of fixing L vs fixing only N (N < 50 000)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=8.5)

    # Panel B: E2 pure subpopulation (∏c=1, T=1, Sha=1) at ranks 0, 1 and 2
    for r, col, mk in [
        (0, "#1f5a99", "o"),
        (1, "#c0392b", "s"),
        (2, "#27ae60", "^"),
    ]:
        s2 = df_e2.loc[(df_e2["rank"] == r) & (df_e2["window"] == "1-49999")]
        ax2.plot(
            s2["p"],
            s2["rho_pure_given_L"],
            f"{mk}-",
            color=col,
            linewidth=1.8,
            label=f"Rank {r} pure (∏c=1, T=1, Sha=1)",
        )
    ax2.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Prime p")
    ax2.set_ylabel("ρ(aₚ, log Ω | L) in the pure subpopulation")
    ax2.set_title("E2. Pure subpopulation ∏c=1, T=1, Sha=1 (N < 50 000)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        "Exploratory analyses E1 and E2 (EXPLORATORY)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(EXP_DIR / "exploratory.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"Total time e08_exploratory: {t_end - t_start:.2f} s")
    print("\n--- E1: WITH L VS WITHOUT L (p in {3, 5, 7}) ---")
    print(df_e1.loc[df_e1["p"].isin([3, 5, 7])].to_string(index=False))
    print("\n--- E2: PURE SUBPOPULATION (p in {3, 5, 7}) ---")
    print(df_e2.loc[df_e2["p"].isin([3, 5, 7])].to_string(index=False))
    print("\n--- E3: PARITY OF ap AGAINST log ∏c IN CURVES WITH T=1 (p in {3, 5, 7}) ---")
    print(
        df_e3.loc[
            (df_e3["rank"].isin([0, 1])) & (df_e3["p"].isin([3, 5, 7]))
        ].to_string(index=False)
    )
    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_exploratory()
