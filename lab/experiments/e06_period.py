from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import load_ap_cache, discriminant, legendre_symbol
from lab.stats import (
    partial_correlation_bins,
    profile_difference,
    null_profile_distribution,
)

SEED = 20261121
N_PERM = 1000
N_PRIMES = 25

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "06-period-components"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def _size_label(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return ">=3"


def run_period_components() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)

    # Statistics at the level of the full class, for Part D
    grp_class = df_bsd.groupby(["conductor", "isoclass"], as_index=False).agg(
        sha_max_class=("sha_an", "max"),
        n_curves_in_class=("number", "count"),
    )

    classes = (
        one_per_class(df_bsd)
        .merge(grp_class, on=["conductor", "isoclass"], how="left")
        .reset_index(drop=True)
    )

    # Exact discriminant and its sign for each class (curve 1)
    discs_py = [discriminant(ai) for ai in classes["ainvs"]]
    sign_pos_all = np.array([d > 0 for d in discs_py], dtype=bool)
    classes["sign_pos"] = sign_pos_all
    classes["omega_plus"] = np.where(
        sign_pos_all, classes["omega"] / 2.0, classes["omega"]
    )

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    rows_corr: list[dict[str, str | int | float]] = []
    rows_sign: list[dict[str, str | int | float]] = []

    # Parts A and B
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
            sign_pos = sub["sign_pos"].to_numpy(dtype=bool)
            bins_L_sign = bins_L * 2 + sign_pos.astype(int)

            omega = sub["omega"].to_numpy(dtype=np.float64)
            omega_plus = sub["omega_plus"].to_numpy(dtype=np.float64)
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)

            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)
            good_sub = (cond_sub[:, None] % primes[None, :]) != 0

            # Partial correlations per prime (Part A and Part B.i/ii)
            for j, p_prime in enumerate(primes):
                p_int = int(p_prime)
                p_mod4 = int(p_int % 4)
                mask_b = good_sub[:, j]
                n_b = int(np.sum(mask_b))
                ap_p = ap_sub[mask_b, j].astype(np.float64)

                rho_L, n_ef_L = partial_correlation_bins(
                    ap_p, omega[mask_b], bins_L[mask_b]
                )
                rho_L_sign, _ = partial_correlation_bins(
                    ap_p, omega[mask_b], bins_L_sign[mask_b]
                )
                rho_L_om_plus, _ = partial_correlation_bins(
                    ap_p, omega_plus[mask_b], bins_L[mask_b]
                )
                se_val = float(1.0 / np.sqrt(n_ef_L)) if n_ef_L > 0 else float("nan")

                rows_corr.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "p_mod4": p_mod4,
                        "n": n_b,
                        "rho_L": rho_L,
                        "rho_L_sign": rho_L_sign,
                        "rho_L_omega_plus": rho_L_om_plus,
                        "se": se_val,
                    }
                )

            # Difference in mean ap between Δ > 0 and Δ < 0 within the L bins (Part B)
            delta_obs, _ = profile_difference(ap_sub, good_sub, sign_pos)
            deltas_nulls, _ = null_profile_distribution(
                ap_sub,
                good_sub,
                sign_pos,
                bins_L,
                n_perm=N_PERM,
                seed=SEED,
            )
            mean_null = deltas_nulls.mean(axis=0)
            sd_null = deltas_nulls.std(axis=0, ddof=1)
            delta_adjusted_L = delta_obs - mean_null
            z_sign = delta_adjusted_L / sd_null

            for j, p_prime in enumerate(primes):
                p_int = int(p_prime)
                rows_sign.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "p_mod4": int(p_int % 4),
                        "delta_sign": float(delta_adjusted_L[j]),
                        "z": float(z_sign[j]),
                    }
                )

    df_corr = pd.DataFrame(rows_corr)
    df_corr.to_csv(OUTPUT_DIR / "correlations.csv", index=False)

    df_sign = pd.DataFrame(rows_sign)
    df_sign.to_csv(OUTPUT_DIR / "sign.csv", index=False)

    # Part C: Legendre-symbol mechanism (odd p only, rank 0, all N < 100 000)
    mask_r0 = (classes["rank"] == 0).to_numpy(dtype=bool)
    sub_r0 = classes.loc[mask_r0].reset_index(drop=True)
    discs_r0 = [d for d, m in zip(discs_py, mask_r0) if m]
    abs_discs_r0 = [abs(d) for d in discs_r0]
    sign_r0 = np.array([1 if d > 0 else -1 for d in discs_r0], dtype=np.int8)
    cond_r0 = sub_r0["conductor"].to_numpy(dtype=np.int32)
    ap_r0 = df_ap.loc[list(sub_r0["label"])].to_numpy(dtype=np.int16)

    rows_leg: list[dict[str, int | float]] = []
    for j, p_prime in enumerate(primes):
        p_int = int(p_prime)
        if p_int == 2:
            continue
        p_mod4 = int(p_int % 4)
        table_leg = [0 if r == 0 else legendre_symbol(r, p_int) for r in range(p_int)]
        leg_minus1 = -1 if p_mod4 == 3 else 1

        mask_b = (cond_r0 % p_int) != 0
        leg_abs_all = np.array(
            [table_leg[ad % p_int] for ad in abs_discs_r0], dtype=np.int8
        )
        leg_abs = leg_abs_all[mask_b]
        sig_b = sign_r0[mask_b]
        leg_disc = np.where(sig_b > 0, leg_abs, leg_minus1 * leg_abs)
        ap_p = ap_r0[mask_b, j].astype(np.float64)

        prop_abs_res = float(np.mean(leg_abs == 1))
        corr_sig_leg = float(np.corrcoef(sig_b, leg_disc)[0, 1])

        mask_plus = leg_disc == 1
        mask_minus = leg_disc == -1
        n_plus = int(np.sum(mask_plus))
        n_minus = int(np.sum(mask_minus))
        mean_plus = float(np.mean(ap_p[mask_plus]))
        mean_minus = float(np.mean(ap_p[mask_minus]))

        rows_leg.append(
            {
                "p": p_int,
                "p_mod4": p_mod4,
                "prop_absdisc_residual": prop_abs_res,
                "corr_sign_legendre": corr_sig_leg,
                "mean_ap_leg_plus": mean_plus,
                "mean_ap_leg_minus": mean_minus,
                "n_plus": n_plus,
                "n_minus": n_minus,
            }
        )

    df_leg = pd.DataFrame(rows_leg)
    df_leg.to_csv(OUTPUT_DIR / "legendre.csv", index=False)

    # Part D: descriptive H7 by rank (0, 1, 2), window and class size
    classes["size_cat"] = [
        _size_label(int(c)) for c in classes["n_curves_in_class"]
    ]
    rows_iso: list[dict[str, str | int | float]] = []

    for rank in (0, 1, 2):
        for win_name, n_min, n_max in WINDOWS:
            base = classes.loc[
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ]
            for tam_tag in ("all", "1", "2", ">=3"):
                if tam_tag == "all":
                    sub_d = base
                else:
                    sub_d = base.loc[base["size_cat"] == tam_tag]

                n_cls = len(sub_d)
                if n_cls == 0:
                    continue
                sha1_ge4 = np.rint(sub_d["sha_an"].to_numpy(dtype=float)).astype(int) >= 4
                shamax_ge4 = (
                    np.rint(sub_d["sha_max_class"].to_numpy(dtype=float)).astype(int) >= 4
                )
                n_c1 = int(np.sum(sha1_ge4))
                n_max_c = int(np.sum(shamax_ge4))
                pct_c1 = 100.0 * n_c1 / n_cls
                pct_max = 100.0 * n_max_c / n_cls
                quotient = (n_max_c / n_c1) if n_c1 > 0 else float("nan")

                rows_iso.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "class_size": tam_tag,
                        "n_classes": n_cls,
                        "n_sha4_curve1": n_c1,
                        "pct_sha4_curve1": pct_c1,
                        "n_sha4_class": n_max_c,
                        "pct_sha4_class": pct_max,
                        "quotient": quotient,
                    }
                )

    df_iso = pd.DataFrame(rows_iso)
    df_iso.to_csv(OUTPUT_DIR / "isogeny_sha.csv", index=False)

    # Figure correlations.png: rank 0, window N < 50 000
    sub_plot = df_corr.loc[
        (df_corr["rank"] == 0) & (df_corr["window"] == "1-49999")
    ].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=150, sharey=True)
    colors_mod4 = {1: "#1f5a99", 3: "#c0392b", 2: "#7f8c8d"}
    labels_mod4 = {
        3: "p ≡ 3 (mod 4)",
        1: "p ≡ 1 (mod 4)",
        2: "p = 2",
    }

    for ax, col_y, title in [
        (ax1, "rho_L", "A. Not controlling sign(Δ): ρ(aₚ, Ω | L)"),
        (ax2, "rho_L_sign", "B. Controlling sign(Δ): ρ(aₚ, Ω | L, sign(Δ))"),
    ]:
        for mod_val in (3, 1, 2):
            m = sub_plot["p_mod4"] == mod_val
            sub_m = sub_plot.loc[m]
            ax.errorbar(
                sub_m["p"],
                sub_m[col_y],
                yerr=1.96 * sub_m["se"],
                fmt="o",
                color=colors_mod4[mod_val],
                ecolor=colors_mod4[mod_val],
                elinewidth=1.2,
                capsize=3,
                markersize=6,
                label=labels_mod4[mod_val],
            )
        ax.plot(
            sub_plot["p"],
            sub_plot["rho_L_omega_plus"],
            "x--",
            color="#27ae60",
            linewidth=1.1,
            markersize=5,
            alpha=0.8,
            label="ρ(aₚ, Ω⁺ | L)",
        )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Prime p (p ≤ 97)")
        ax.set_ylabel("Partial correlation ρₚ (±1.96 SE)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    fig.suptitle(
        "Partial correlation between aₚ and the period Ω at rank 0, N < 50 000 (one curve per class)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlations.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"Total time e06_period: {t_end - t_start:.2f} s")

    print("\n--- CORRELATIONS (rank 0, first 10 primes in both windows) ---")
    print(
        df_corr.loc[(df_corr["rank"] == 0) & (df_corr["p"] <= 29)].to_string(
            index=False
        )
    )

    print("\n--- SUMMARY BY p mod 4 (excluding p = 2) ---")
    for rank in (0, 1):
        for win, _, _ in WINDOWS:
            sc = df_corr.loc[
                (df_corr["rank"] == rank)
                & (df_corr["window"] == win)
                & (df_corr["p"] > 2)
            ]
            ss = df_sign.loc[
                (df_sign["rank"] == rank)
                & (df_sign["window"] == win)
                & (df_sign["p"] > 2)
            ]
            for mod_val in (1, 3):
                mc = sc.loc[sc["p_mod4"] == mod_val]
                ms = ss.loc[ss["p_mod4"] == mod_val]
                print(
                    f"rank={rank} win={win} p≡{mod_val}(mod4): "
                    f"mean(rho_L)={mc['rho_L'].mean():+.5f}, "
                    f"mean(rho_L_sign)={mc['rho_L_sign'].mean():+.5f}, "
                    f"mean(rho_L_omega_plus)={mc['rho_L_omega_plus'].mean():+.5f}, "
                    f"mean(delta_sign)={ms['delta_sign'].mean():+.5f}, "
                    f"mean(z)={ms['z'].mean():+.2f}"
                )

    print("\n--- SIGN (rank 0, p <= 29) ---")
    print(
        df_sign.loc[(df_sign["rank"] == 0) & (df_sign["p"] <= 29)].to_string(
            index=False
        )
    )

    print("\n--- LEGENDRE (rank 0, N < 100 000, p <= 31) ---")
    print(df_leg.loc[df_leg["p"] <= 31].to_string(index=False))

    print("\n--- ISOGENY SHA (Part D) ---")
    print(df_iso.to_string(index=False))

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_period_components()
