from __future__ import annotations

import math
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import DERIVED_DIR, load_ap_cache, discriminant
from lab.stats import (
    stratified_bins,
    stratum_key,
    partial_correlation_bins,
    partial_correlation_cov,
    partial_residuals_cov,
)

N_PRIMES = 25
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "results" / "10-closing-h16" / "exploratory"
BAD_EULER_PATH = DERIVED_DIR / "bad_euler.csv"
REDUCTION_TYPES_PATH = DERIVED_DIR / "reduction_types.csv"

WINDOWS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]
TARGET_PRIMES = (3, 5, 7)


def _exact_integer_key(cp: np.ndarray, T: np.ndarray, sha: np.ndarray) -> np.ndarray:
    cp_i = np.asarray(cp, dtype=np.int64)
    t_i = np.asarray(T, dtype=np.int64)
    s_i = np.rint(np.asarray(sha, dtype=np.float64)).astype(np.int64)
    # Bijective code without pooling tails (T <= 16, Sha <= 1000 for N < 100k)
    return (cp_i * 1000 + t_i) * 10000 + s_i


def _slope_in_bins(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> float:
    rx = partial_residuals_cov(x, bins)
    ry = partial_residuals_cov(y, bins)
    denom = float(np.sum(rx * rx))
    return float(np.sum(rx * ry) / denom) if denom > 0.0 else float("nan")


def run_exploratory() -> dict:
    t_start = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)
    df_eb = pd.read_csv(BAD_EULER_PATH)
    df_tr = pd.read_csv(REDUCTION_TYPES_PATH)
    classes = classes.merge(df_eb, on="label", how="left")
    classes = classes.merge(df_tr, on="label", how="left")

    classes["log_disc"] = [
        math.log(abs(discriminant(ai))) for ai in classes["ainvs"]
    ]
    classes["log_N"] = np.log(classes["conductor"].to_numpy(dtype=np.float64))
    classes["log_exc"] = classes["log_disc"] - classes["log_N"]
    classes["log_omega"] = np.log(classes["omega"].to_numpy(dtype=np.float64))
    classes["log_R"] = np.log(classes["regulator"].to_numpy(dtype=np.float64))
    classes["log_L"] = np.log(classes["l_value"].to_numpy(dtype=np.float64))

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primes = {int(p): j for j, p in enumerate(primes)}

    rows_e10_1: list[dict[str, str | int | float]] = []
    rows_e10_2a: list[dict[str, str | int | float]] = []
    rows_e10_2b: list[dict[str, str | int | float]] = []
    rows_e10_2c: list[dict[str, str | int | float]] = []
    rows_e10_3a: list[dict[str, str | int | float]] = []
    rows_e10_3b: list[dict[str, str | int | float]] = []

    for rank in (1, 2):
        for win_name, n_min, n_max in WINDOWS:
            mask_rv = (
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = classes.loc[mask_rv].reset_index(drop=True)

            l_val = sub["l_value"].to_numpy(dtype=np.float64)
            log_L = sub["log_L"].to_numpy(dtype=np.float64)
            log_om = sub["log_omega"].to_numpy(dtype=np.float64)
            log_R = sub["log_R"].to_numpy(dtype=np.float64)
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            log_N = sub["log_N"].to_numpy(dtype=np.float64)
            log_exc = sub["log_exc"].to_numpy(dtype=np.float64)
            cond = sub["conductor"].to_numpy(dtype=np.int32)
            cp = sub["tamagawa"].to_numpy(dtype=np.int64)
            T = sub["torsion"].to_numpy(dtype=np.int64)
            sha = sub["sha_an"].to_numpy(dtype=np.float64)
            n_split = sub["n_split"].to_numpy(dtype=np.int64)
            n_nonsplit = sub["n_no_split"].to_numpy(dtype=np.int64)
            n_addit = sub["n_additive"].to_numpy(dtype=np.int64)
            n_bad = sub["n_bad"].to_numpy(dtype=np.int64)

            strat_48 = stratum_key(cp, T, sha)
            strat_ex = _exact_integer_key(cp, T, sha)
            q_B = np.asarray(
                pd.qcut(log_B, q=5, labels=False, duplicates="drop"),
                dtype=np.int64,
            )
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)

            # ==================== E10.1 ====================
            for p_int in TARGET_PRIMES:
                j = idx_primes[p_int]
                mb = (cond % p_int) != 0
                ap_b = ap_sub[mb, j].astype(np.float64)
                l_b = l_val[mb]
                logL_b = log_L[mb]
                om_b = log_om[mb]
                B_b = log_B[mb]
                disc_b = log_disc[mb]
                N_b = log_N[mb]
                exc_b = log_exc[mb]
                strat_b = strat_48[mb]

                bins_L = stratified_bins(l_b, strat_b, q_max=100, n_min=200)
                mv = bins_L >= 0
                ap_v = ap_b[mv]
                logL_v = logL_b[mv]
                om_v = om_b[mv]
                B_v = B_b[mv]
                disc_v = disc_b[mv]
                N_v = N_b[mv]
                exc_v = exc_b[mv]
                strat_v = strat_b[mv]
                bins_v = bins_L[mv]

                # L not fixed (within strata only)
                rho_ap_om_noL, _ = partial_correlation_bins(ap_v, om_v, strat_v)
                rho_ap_disc_noL, _ = partial_correlation_bins(ap_v, disc_v, strat_v)
                rho_disc_L_noL, _ = partial_correlation_bins(disc_v, logL_v, strat_v)
                rho_disc_L_noL_givenB, _ = partial_correlation_cov(
                    disc_v, logL_v, strat_v, B_v
                )

                # Fixing L within strata
                rho_base, _ = partial_correlation_bins(ap_v, om_v, bins_v)
                rho_ap_disc_L, _ = partial_correlation_bins(ap_v, disc_v, bins_v)
                r_D_om, _ = partial_correlation_bins(disc_v, om_v, bins_v)
                passive_ceiling_pct = 100.0 * (1.0 - math.sqrt(max(0.0, 1.0 - r_D_om * r_D_om)))

                rho_logB, _ = partial_correlation_cov(ap_v, om_v, bins_v, B_v)
                rho_logdisc, _ = partial_correlation_cov(ap_v, om_v, bins_v, disc_v)
                rho_logN, _ = partial_correlation_cov(ap_v, om_v, bins_v, N_v)
                rho_logexc, _ = partial_correlation_cov(ap_v, om_v, bins_v, exc_v)
                rho_B_N, _ = partial_correlation_cov(
                    ap_v, om_v, bins_v, np.column_stack([B_v, N_v])
                )
                rho_B_exc, _ = partial_correlation_cov(
                    ap_v, om_v, bins_v, np.column_stack([B_v, exc_v])
                )
                rho_B_disc, _ = partial_correlation_cov(
                    ap_v, om_v, bins_v, np.column_stack([B_v, disc_v])
                )
                pct_red_disc = 100.0 * (1.0 - abs(rho_logdisc) / abs(rho_base))

                rows_e10_1.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n": int(np.sum(mv)),
                        "rho_ap_om_noL": rho_ap_om_noL,
                        "rho_ap_disc_noL": rho_ap_disc_noL,
                        "rho_disc_L_noL": rho_disc_L_noL,
                        "rho_disc_L_noL_givenB": rho_disc_L_noL_givenB,
                        "rho_base_L": rho_base,
                        "rho_ap_disc_L": rho_ap_disc_L,
                        "r_D_omega": r_D_om,
                        "passive_ceiling_pct": passive_ceiling_pct,
                        "rho_L_logB": rho_logB,
                        "rho_L_logdisc": rho_logdisc,
                        "pct_red_logdisc": pct_red_disc,
                        "rho_L_logN": rho_logN,
                        "rho_L_logexc": rho_logexc,
                        "rho_L_logB_logN": rho_B_N,
                        "rho_L_logB_logexc": rho_B_exc,
                        "rho_L_logB_logdisc": rho_B_disc,
                    }
                )

            # ==================== E10.2a: dissection of B vs f_split, f_additive ====================
            n_mult = n_split + n_nonsplit
            m_mult = n_mult > 0
            f_split = n_split[m_mult].astype(np.float64) / n_mult[m_mult].astype(np.float64)
            f_addit = n_addit[m_mult].astype(np.float64) / n_bad[m_mult].astype(np.float64)
            Z_f = np.column_stack([f_split, f_addit])

            for scheme_name, strat_arr in [("strata_48", strat_48), ("exact_integer_strata", strat_ex)]:
                bins_m = stratified_bins(l_val[m_mult], strat_arr[m_mult], q_max=100, n_min=200)
                mv_m = bins_m >= 0
                B_m = log_B[m_mult][mv_m]
                fp_m = f_split[mv_m]
                Zf_m = Z_f[mv_m]
                bm = bins_m[mv_m]

                for obj_name, obj_arr in [("log_disc", log_disc), ("log_exc", log_exc)]:
                    Y_m = obj_arr[m_mult][mv_m]
                    rho_B_Y, nef_m = partial_correlation_bins(B_m, Y_m, bm)
                    rho_fp_Y, _ = partial_correlation_bins(fp_m, Y_m, bm)
                    rho_B_Y_f, _ = partial_correlation_cov(B_m, Y_m, bm, Zf_m)
                    rho_fp_Y_B, _ = partial_correlation_cov(fp_m, Y_m, bm, B_m)
                    pct_red_B_f = 100.0 * (1.0 - abs(rho_B_Y_f) / abs(rho_B_Y))
                    rows_e10_2a.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "scheme": scheme_name,
                            "target": obj_name,
                            "n": int(np.sum(mv_m)),
                            "se": float(1.0 / math.sqrt(nef_m)),
                            "rho_B_Y": rho_B_Y,
                            "rho_fp_Y": rho_fp_Y,
                            "rho_B_Y_given_f": rho_B_Y_f,
                            "pct_red_B_by_f": pct_red_B_f,
                            "rho_fp_Y_given_B": rho_fp_Y_B,
                        }
                    )

            # ==================== E10.2b: by individual stratum (T=1, Sha=1) ====================
            groups_c = [
                ("c=1", cp == 1),
                ("c=2", cp == 2),
                ("c=3", cp == 3),
                ("c=4", cp == 4),
                ("c=5-8", (cp >= 5) & (cp <= 8)),
                ("c>=9", cp >= 9),
                ("c=6_exact", cp == 6),
                ("c=8_exact", cp == 8),
                ("c=12_exact", cp == 12),
            ]
            base_t1s1 = (T == 1) & (np.rint(sha) == 1)
            for c_name, c_mask in groups_c:
                m_ec = base_t1s1 & c_mask & m_mult
                n_ec = int(np.sum(m_ec))
                if n_ec < 200:
                    continue
                b_ec = stratified_bins(
                    l_val[m_ec], np.zeros(n_ec, dtype=int), q_max=100, n_min=200
                )
                B_ec = log_B[m_ec]
                disc_ec = log_disc[m_ec]
                exc_ec = log_exc[m_ec]
                om_ec = log_om[m_ec]
                fp_ec = n_split[m_ec].astype(np.float64) / n_mult[m_ec].astype(np.float64)
                fa_ec = n_addit[m_ec].astype(np.float64) / n_bad[m_ec].astype(np.float64)

                r_B_d, nef_ec = partial_correlation_bins(B_ec, disc_ec, b_ec)
                r_fp_d, _ = partial_correlation_bins(fp_ec, disc_ec, b_ec)
                r_B_ex, _ = partial_correlation_bins(B_ec, exc_ec, b_ec)
                r_fp_ex, _ = partial_correlation_bins(fp_ec, exc_ec, b_ec)
                r_B_d_f, _ = partial_correlation_cov(
                    B_ec, disc_ec, b_ec, np.column_stack([fp_ec, fa_ec])
                )
                r_B_om, _ = partial_correlation_bins(B_ec, om_ec, b_ec)
                rows_e10_2b.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "stratum": f"{c_name}|T=1|Sha=1",
                        "n": n_ec,
                        "se": float(1.0 / math.sqrt(nef_ec)),
                        "rho_B_logdisc": r_B_d,
                        "rho_fp_logdisc": r_fp_d,
                        "rho_B_logexc": r_B_ex,
                        "rho_fp_logexc": r_fp_ex,
                        "rho_B_logdisc_given_f": r_B_d_f,
                        "rho_B_logom": r_B_om,
                    }
                )

            # ==================== E10.2c: P12 with exact integer strata (no 5-8, >=9 tails) ====================
            for p_int in TARGET_PRIMES:
                j = idx_primes[p_int]
                mb = (cond % p_int) != 0
                ap_b = ap_sub[mb, j].astype(np.float64)
                l_b = l_val[mb]
                om_b = log_om[mb]
                B_b = log_B[mb]
                disc_b = log_disc[mb]
                strat_ex_b = strat_ex[mb]
                qB_b = q_B[mb]

                bins_ex = stratified_bins(l_b, strat_ex_b, q_max=100, n_min=200)
                mv_ex = bins_ex >= 0
                r_ex_base, nef_ex = partial_correlation_bins(
                    ap_b[mv_ex], om_b[mv_ex], bins_ex[mv_ex]
                )
                strat_ex_qB = strat_ex_b * 10 + qB_b
                bins_ex_qB = stratified_bins(l_b, strat_ex_qB, q_max=100, n_min=200)
                mv_ex_qB = bins_ex_qB >= 0
                r_ex_qB, _ = partial_correlation_bins(
                    ap_b[mv_ex_qB], om_b[mv_ex_qB], bins_ex_qB[mv_ex_qB]
                )
                r_ex_logB, _ = partial_correlation_cov(
                    ap_b[mv_ex], om_b[mv_ex], bins_ex[mv_ex], B_b[mv_ex]
                )
                r_ex_logdisc, _ = partial_correlation_cov(
                    ap_b[mv_ex], om_b[mv_ex], bins_ex[mv_ex], disc_b[mv_ex]
                )
                rows_e10_2c.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n_exact": int(np.sum(mv_ex)),
                        "se_exact": float(1.0 / math.sqrt(nef_ex)),
                        "rho_exact": r_ex_base,
                        "n_exact_qB": int(np.sum(mv_ex_qB)),
                        "rho_exact_qB": r_ex_qB,
                        "pct_red_qB": 100.0 * (1.0 - abs(r_ex_qB) / abs(r_ex_base)),
                        "rho_exact_logB": r_ex_logB,
                        "pct_red_logB": 100.0 * (1.0 - abs(r_ex_logB) / abs(r_ex_base)),
                        "rho_exact_logdisc": r_ex_logdisc,
                        "pct_red_logdisc": 100.0 * (1.0 - abs(r_ex_logdisc) / abs(r_ex_base)),
                    }
                )

            # ==================== E10.3a: slopes d(log Omega, log R, log L)/d(log|Delta|) ====================
            # Compare pure (c=1, T=1, Sha=1) vs exact non-pure (c>=2, T=1, Sha=1, by integer value of c)
            m_pure = (cp == 1) & (T == 1) & (np.rint(sha) == 1)
            m_c2p = (cp >= 2) & (T == 1) & (np.rint(sha) == 1)
            for sub_name, m_sub, strat_sub in [
                ("pure_c1_T1_Sha1", m_pure, np.zeros(len(sub), dtype=int)),
                ("non_pure_c>=2_integers_T1_Sha1", m_c2p, strat_ex),
            ]:
                # Keep strata with n >= 200
                vals_e, cnts_e = np.unique(strat_sub[m_sub], return_counts=True)
                valid = set(vals_e[cnts_e >= 200].tolist())
                m_use = m_sub & np.isin(strat_sub, list(valid))
                b_use = strat_sub[m_use]
                d_u = log_disc[m_use]
                om_u = log_om[m_use]
                R_u = log_R[m_use]
                L_u = log_L[m_use]
                B_u = log_B[m_use]

                rho_d_L, nef_u = partial_correlation_bins(d_u, L_u, b_use)
                rho_d_L_givenB, _ = partial_correlation_cov(d_u, L_u, b_use, B_u)
                rho_d_om, _ = partial_correlation_bins(d_u, om_u, b_use)
                rho_d_R, _ = partial_correlation_bins(d_u, R_u, b_use)
                b_om = _slope_in_bins(d_u, om_u, b_use)
                b_R = _slope_in_bins(d_u, R_u, b_use)
                b_L = _slope_in_bins(d_u, L_u, b_use)
                rows_e10_3a.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "subset": sub_name,
                        "n": int(np.sum(m_use)),
                        "se": float(1.0 / math.sqrt(nef_u)),
                        "rho_logdisc_logL": rho_d_L,
                        "rho_logdisc_logL_givenB": rho_d_L_givenB,
                        "rho_logdisc_logom": rho_d_om,
                        "rho_logdisc_logR": rho_d_R,
                        "slope_logom_over_logdisc": b_om,
                        "slope_logR_over_logdisc": b_R,
                        "slope_logL_over_logdisc": b_L,
                        "error_sum_slope": abs((b_om + b_R) - b_L),
                    }
                )

            # ==================== E10.3b: pure stratum at p = 2, 3, 5, 7 ====================
            for p_int in (2, 3, 5, 7):
                j = idx_primes[p_int]
                mp = m_pure & ((cond % p_int) != 0)
                n_p = int(np.sum(mp))
                if n_p < 200:
                    continue
                b_p = stratified_bins(
                    l_val[mp], np.zeros(n_p, dtype=int), q_max=100, n_min=200
                )
                ap_p = ap_sub[mp, j].astype(np.float64)
                om_p = log_om[mp]
                B_p = log_B[mp]
                N_p = log_N[mp]
                disc_p = log_disc[mp]

                r_base, nef_p = partial_correlation_bins(ap_p, om_p, b_p)
                se_p = float(1.0 / math.sqrt(nef_p))
                r_B, _ = partial_correlation_cov(ap_p, om_p, b_p, B_p)
                r_N, _ = partial_correlation_cov(ap_p, om_p, b_p, N_p)
                r_d, _ = partial_correlation_cov(ap_p, om_p, b_p, disc_p)
                r_ap_B, _ = partial_correlation_bins(ap_p, B_p, b_p)
                r_B_om, _ = partial_correlation_bins(B_p, om_p, b_p)
                r_B_d, _ = partial_correlation_bins(B_p, disc_p, b_p)
                rows_e10_3b.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "n_pure": n_p,
                        "se": se_p,
                        "rho_pure_base": r_base,
                        "z_base": r_base / se_p,
                        "rho_pure_logB": r_B,
                        "rho_pure_logN": r_N,
                        "rho_pure_logdisc": r_d,
                        "rho_ap_logB": r_ap_B,
                        "rho_logB_logom": r_B_om,
                        "rho_logB_logdisc": r_B_d,
                    }
                )

    df_e1 = pd.DataFrame(rows_e10_1)
    df_e2a = pd.DataFrame(rows_e10_2a)
    df_e2b = pd.DataFrame(rows_e10_2b)
    df_e2c = pd.DataFrame(rows_e10_2c)
    df_e3a = pd.DataFrame(rows_e10_3a)
    df_e3b = pd.DataFrame(rows_e10_3b)

    df_e1.to_csv(EXP_DIR / "e10_1_risk_diagnostic.csv", index=False)
    df_e2a.to_csv(EXP_DIR / "e10_2a_B_disc_dissection.csv", index=False)
    df_e2b.to_csv(EXP_DIR / "e10_2b_by_stratum.csv", index=False)
    df_e2c.to_csv(EXP_DIR / "e10_2c_pointwise_strata.csv", index=False)
    df_e3a.to_csv(EXP_DIR / "e10_3a_slopes_pure_vs_c2.csv", index=False)
    df_e3b.to_csv(EXP_DIR / "e10_3b_pure_controls.csv", index=False)

    # Save a copy of the code in results/10-closing-h16/exploratory/
    shutil.copy2(Path(__file__), EXP_DIR / "e10_exploratory.py")

    t_end = time.perf_counter()
    print(f"Total time e10_exploratory: {t_end - t_start:.2f} s\n")

    print("=== E10.1: DIAGNOSIS OF THE REGISTERED RISK ===")
    print(df_e1.to_string(index=False))

    print("\n=== E10.2a: DISSECTION log B vs f_split, f_additive ===")
    print(df_e2a.to_string(index=False))

    print("\n=== E10.2b: BY INDIVIDUAL STRATUM (T=1, Sha=1) ===")
    print(df_e2b.to_string(index=False))

    print("\n=== E10.2c: P12 IN EXACT INTEGER STRATA (NO TAILS) ===")
    print(df_e2c.to_string(index=False))

    print("\n=== E10.3a: SLOPES d(log Omega, log R, log L)/d(log|Delta|) ===")
    print(df_e3a.to_string(index=False))

    print("\n=== E10.3b: PURE STRATUM (p=2,3,5,7) ===")
    print(df_e3b.to_string(index=False))

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_exploratory()
