from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import load_ap_cache, discriminant, legendre_symbol
from lab.stats import stratified_bins, partial_correlation_bins

N_PRIMES = 25
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "09-legendre-strata"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]
TARGET_PRIMES = (3, 5, 7)


def _encode_strata(
    tam: np.ndarray, tor: np.ndarray, sha: np.ndarray
) -> tuple[np.ndarray, dict[int, str]]:
    c_code = np.zeros(len(tam), dtype=int)
    c_code[tam == 2] = 1
    c_code[tam == 3] = 2
    c_code[tam == 4] = 3
    c_code[(tam >= 5) & (tam <= 8)] = 4
    c_code[tam >= 9] = 5

    t_code = np.zeros(len(tor), dtype=int)
    t_code[tor == 2] = 1
    t_code[tor == 3] = 2
    t_code[tor >= 4] = 3

    s_code = np.zeros(len(sha), dtype=int)
    s_code[sha >= 4] = 1

    strat_id = c_code * 8 + t_code * 2 + s_code

    c_names = ["1", "2", "3", "4", "5-8", ">=9"]
    t_names = ["1", "2", "3", ">=4"]
    s_names = ["1", ">=4"]
    name_map: dict[int, str] = {}
    for ic, cn in enumerate(c_names):
        for it, tn in enumerate(t_names):
            for is_, sn in enumerate(s_names):
                name_map[ic * 8 + it * 2 + is_] = f"c={cn}|T={tn}|Sha={sn}"

    return strat_id, name_map


def run_legendre_strata() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)
    discs_py = [discriminant(ai) for ai in classes["ainvs"]]

    tam_all = classes["tamagawa"].to_numpy(dtype=int)
    tor_all = classes["torsion"].to_numpy(dtype=int)
    sha_all = np.rint(classes["sha_an"].to_numpy(dtype=float)).astype(int)
    strat_id_all, strat_map = _encode_strata(tam_all, tor_all, sha_all)
    classes["stratum_id"] = strat_id_all

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primes = {int(p): j for j, p in enumerate(primes)}

    rows_strata: list[dict[str, str | int | float]] = []
    rows_leg: list[dict[str, str | int | float]] = []

    # Part A: composition at ranks 1 and 2
    for rank in (1, 2):
        for win_name, n_min, n_max in WINDOWS:
            mask_rv = (
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = classes.loc[mask_rv].reset_index(drop=True)

            l_val = sub["l_value"].to_numpy(dtype=np.float64)
            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            strat_sub = sub["stratum_id"].to_numpy(dtype=int)
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)

            for p_int in TARGET_PRIMES:
                j = idx_primes[p_int]
                mask_b = (cond_sub % p_int) != 0
                ap_b = ap_sub[mask_b, j].astype(np.float64)
                l_b = l_val[mask_b]
                log_om_b = log_omega[mask_b]
                strat_b = strat_sub[mask_b]

                # 1. Global (100 quantiles of L over the whole sample p ∤ N)
                bins_glob = np.asarray(
                    pd.qcut(l_b, q=100, labels=False, duplicates="drop"),
                    dtype=int,
                )
                rho_glob, nef_glob = partial_correlation_bins(
                    ap_b, log_om_b, bins_glob
                )
                rows_strata.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "kind": "global",
                        "n": int(len(ap_b)),
                        "rho": rho_glob,
                        "se": float(1.0 / np.sqrt(nef_glob)),
                    }
                )

                # 2. Pooled within exact strata with stratified_bins
                bins_pooled = stratified_bins(
                    l_b, strat_b, q_max=100, n_min=200
                )
                m_val = bins_pooled >= 0
                rho_pooled, nef_pooled = partial_correlation_bins(
                    ap_b[m_val], log_om_b[m_val], bins_pooled[m_val]
                )
                rows_strata.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "kind": "pooled",
                        "n": int(np.sum(m_val)),
                        "rho": rho_pooled,
                        "se": float(1.0 / np.sqrt(nef_pooled)),
                    }
                )

                # 3. Pure stratum (stratum_id == 0: c=1|T=1|Sha=1)
                m_pure = strat_b == 0
                n_pure = int(np.sum(m_pure))
                bins_pure = stratified_bins(
                    l_b[m_pure],
                    np.zeros(n_pure, dtype=int),
                    q_max=100,
                    n_min=200,
                )
                rho_pure, nef_pure = partial_correlation_bins(
                    ap_b[m_pure], log_om_b[m_pure], bins_pure
                )
                rows_strata.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "kind": "pure",
                        "n": n_pure,
                        "rho": rho_pure,
                        "se": float(1.0 / np.sqrt(nef_pure)),
                    }
                )

                # 4. Each individual stratum with n >= 200
                for eid in sorted(np.unique(strat_b)):
                    m_e = strat_b == eid
                    n_e = int(np.sum(m_e))
                    if n_e < 200:
                        continue
                    b_e = stratified_bins(
                        l_b[m_e],
                        np.zeros(n_e, dtype=int),
                        q_max=100,
                        n_min=200,
                    )
                    rho_e, nef_e = partial_correlation_bins(
                        ap_b[m_e], log_om_b[m_e], b_e
                    )
                    rows_strata.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "p": p_int,
                            "kind": f"stratum:{strat_map[int(eid)]}",
                            "n": n_e,
                            "rho": rho_e,
                            "se": float(1.0 / np.sqrt(nef_e)),
                        }
                    )

    # Part B: route via (Δ/p) at ranks 0 and 1, T = 1 only, p ∈ {3, 5, 7}
    for rank in (0, 1):
        for win_name, n_min, n_max in WINDOWS:
            mask_rv_t1 = (
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
                & (classes["torsion"] == 1)
            ).to_numpy(dtype=bool)
            sub_t1 = classes.loc[mask_rv_t1].reset_index(drop=True)
            discs_t1 = [d for d, m in zip(discs_py, mask_rv_t1) if m]

            l_t1 = sub_t1["l_value"].to_numpy(dtype=np.float64)
            log_tam_t1 = np.log(sub_t1["tamagawa"].to_numpy(dtype=np.float64))
            cond_t1 = sub_t1["conductor"].to_numpy(dtype=np.int32)
            ap_t1 = df_ap.loc[list(sub_t1["label"])].to_numpy(dtype=np.int16)

            for p_int in TARGET_PRIMES:
                j = idx_primes[p_int]
                mask_b = (cond_t1 % p_int) != 0
                ap_b = ap_t1[mask_b, j]
                ap_f = ap_b.astype(np.float64)
                l_b = l_t1[mask_b]
                log_tam_b = log_tam_t1[mask_b]

                table_leg = [
                    0 if r == 0 else legendre_symbol(r, p_int)
                    for r in range(p_int)
                ]
                leg_all = np.array(
                    [table_leg[d % p_int] for d in discs_t1], dtype=np.int8
                )
                leg_b = leg_all[mask_b]

                # (i) Sanity: % of classes with odd ap and (Δ/p) = -1
                is_odd = (ap_b % 2) != 0
                n_odd_leg_minus = int(np.sum(is_odd & (leg_b == -1)))
                pct_sanity = 100.0 * n_odd_leg_minus / len(ap_b)

                # (ii) Mean of log ∏c by (Δ/p) = ±1, with Welch z
                m_plus = leg_b == 1
                m_minus = leg_b == -1
                n_plus = int(np.sum(m_plus))
                n_minus = int(np.sum(m_minus))
                med_tam_plus = float(np.mean(log_tam_b[m_plus]))
                med_tam_minus = float(np.mean(log_tam_b[m_minus]))
                var_plus = float(np.var(log_tam_b[m_plus], ddof=1))
                var_minus = float(np.var(log_tam_b[m_minus], ddof=1))
                se_leg = math.sqrt(var_plus / n_plus + var_minus / n_minus)
                delta_leg = med_tam_plus - med_tam_minus
                z_welch_leg = delta_leg / se_leg if se_leg > 0 else 0.0

                # (iii) ρ(ap, log ∏c | L) with L bins, and with L × (Δ/p) bins
                bins_L = np.asarray(
                    pd.qcut(l_b, q=100, labels=False, duplicates="drop"),
                    dtype=int,
                )
                bins_L_leg = bins_L * 2 + (leg_b == 1).astype(int)
                rho_tam_L, _ = partial_correlation_bins(ap_f, log_tam_b, bins_L)
                rho_tam_L_leg, _ = partial_correlation_bins(
                    ap_f, log_tam_b, bins_L_leg
                )
                pct_red_rho = 100.0 * (
                    1.0 - abs(rho_tam_L_leg) / abs(rho_tam_L)
                )

                # (iv) Within (Δ/p) = +1: mean of log ∏c with even vs odd ap
                ap_plus = ap_b[m_plus]
                tam_plus = log_tam_b[m_plus]
                is_even_plus = (ap_plus % 2) == 0
                n_even_plus = int(np.sum(is_even_plus))
                n_odd_plus = int(np.sum(~is_even_plus))
                med_tam_even_plus = float(np.mean(tam_plus[is_even_plus]))
                med_tam_odd_plus = float(np.mean(tam_plus[~is_even_plus]))
                v_even_plus = float(np.var(tam_plus[is_even_plus], ddof=1))
                v_odd_plus = float(np.var(tam_plus[~is_even_plus], ddof=1))
                se_even_odd = math.sqrt(
                    v_even_plus / n_even_plus + v_odd_plus / n_odd_plus
                )
                delta_even_odd_plus = med_tam_even_plus - med_tam_odd_plus
                z_even_odd_plus = (
                    delta_even_odd_plus / se_even_odd if se_even_odd > 0 else 0.0
                )

                rows_leg.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n_T1": int(len(ap_b)),
                        "pct_ap_odd_leg_minus": pct_sanity,
                        "n_leg_plus": n_plus,
                        "n_leg_minus": n_minus,
                        "mean_logtam_leg_plus": med_tam_plus,
                        "mean_logtam_leg_minus": med_tam_minus,
                        "delta_logtam_leg": delta_leg,
                        "z_welch_leg": z_welch_leg,
                        "rho_ap_logtam_L": rho_tam_L,
                        "rho_ap_logtam_L_leg": rho_tam_L_leg,
                        "pct_reduction_rho": pct_red_rho,
                        "n_legplus_ap_even": n_even_plus,
                        "n_legplus_ap_odd": n_odd_plus,
                        "mean_logtam_legplus_ap_even": med_tam_even_plus,
                        "mean_logtam_legplus_ap_odd": med_tam_odd_plus,
                        "delta_logtam_legplus_even_odd": delta_even_odd_plus,
                        "z_welch_legplus_even_odd": z_even_odd_plus,
                    }
                )

    df_strat = pd.DataFrame(rows_strata)
    df_strat.to_csv(OUTPUT_DIR / "strata.csv", index=False)

    df_leg = pd.DataFrame(rows_leg)
    df_leg.to_csv(OUTPUT_DIR / "legendre_tamagawa.csv", index=False)

    # Figure strata.png: for ranks 1 and 2, global, pooled and pure-stratum ρ at p = 3, 5, 7 with ±2 SE bars
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150, sharey=True)
    types_plot = [
        ("global", "o-", "#1f5a99", -0.15, "Global (L bins)"),
        ("pooled", "s-", "#c0392b", 0.0, "Pooled within exact strata"),
        ("pure", "^--", "#27ae60", 0.15, "Pure stratum (∏c=1, T=1, Sha=1)"),
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
        (axes[0], 1, "A. Rank 1: ρ(aₚ, log Ω | L′(1)) ± 2 SE"),
        (axes[1], 2, "B. Rank 2: ρ(aₚ, log Ω | L″(1)/2) ± 2 SE"),
    ]:
        sub_r = df_strat.loc[df_strat["rank"] == rank]
        for kind_key, fmt, col, dx, label in types_plot:
            ys = []
            errs = []
            for win, p_val in pairs_xp:
                row = sub_r.loc[
                    (sub_r["window"] == win)
                    & (sub_r["p"] == p_val)
                    & (sub_r["kind"] == kind_key)
                ].iloc[0]
                ys.append(float(row["rho"]))
                errs.append(2.0 * float(row["se"]))
            ax.errorbar(
                x_base + dx,
                ys,
                yerr=errs,
                fmt=fmt,
                color=col,
                ecolor=col,
                elinewidth=1.3,
                capsize=3.5,
                markersize=6,
                linewidth=1.6,
                label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.axvline(2.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(labels_x)
        ax.set_ylabel("Partial correlation ρ(aₚ, log Ω | L)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.8)

    fig.suptitle(
        "Composition effect across exact strata of (∏c, T, Sha) at ranks 1 and 2",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "strata.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"Total time e09_strata: {t_end - t_start:.2f} s")

    print("\n--- PART A: STRATA SUMMARY (global, pooled, pure) ---")
    print(
        df_strat.loc[df_strat["kind"].isin(["global", "pooled", "pure"])].to_string(
            index=False
        )
    )

    print("\n--- PART B: LEGENDRE–TAMAGAWA ROUTE (T = 1) ---")
    print(df_leg.to_string(index=False))

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_legendre_strata()
