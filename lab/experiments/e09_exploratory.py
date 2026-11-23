from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import DERIVED_DIR, load_ap_cache, discriminant, legendre_symbol
from lab.stats import (
    stratified_bins,
    partial_correlation_bins,
    covariance_in_bins,
)
from lab.experiments.e09_strata import TARGET_PRIMES, WINDOWS, _encode_strata

N_PRIMES = 25
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "results" / "09-legendre-strata" / "exploratory"
BAD_EULER_PATH = DERIVED_DIR / "bad_euler.csv"


def run_exploratory_09() -> dict:
    t_start = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd).reset_index(drop=True)
    df_eb = pd.read_csv(BAD_EULER_PATH)
    classes = classes.merge(df_eb, on="label", how="left")
    discs_py = [discriminant(ai) for ai in classes["ainvs"]]

    tam_all = classes["tamagawa"].to_numpy(dtype=int)
    tor_all = classes["torsion"].to_numpy(dtype=int)
    sha_all = np.rint(classes["sha_an"].to_numpy(dtype=float)).astype(int)
    strat_id_all, _ = _encode_strata(tam_all, tor_all, sha_all)
    classes["stratum_id"] = strat_id_all

    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primes = {int(p): j for j, p in enumerate(primes)}

    # E9.1: exact strata at rank 0 and pointwise strata (∏c in {1,2,3,4}, Sha=1) at ranks 0, 1, 2
    rows_e91: list[dict[str, str | int | float]] = []
    # E9.2: 3 states of 2-torsion over F_p (0_roots, 1_root, 3_roots) vs log B at T = 1
    rows_e92: list[dict[str, str | int | float]] = []
    # E9.3: exact split of Cov(ap, log L^(r) | N) between log R and log Ω in the pure stratum at ranks 1 and 2
    rows_e93: list[dict[str, str | int | float]] = []

    for rank in (0, 1, 2):
        for win_name, n_min, n_max in WINDOWS:
            mask_rv = (
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = classes.loc[mask_rv].reset_index(drop=True)
            discs_rv = [d for d, m in zip(discs_py, mask_rv) if m]

            l_val = sub["l_value"].to_numpy(dtype=np.float64)
            log_L = np.log(l_val)
            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
            log_reg = (
                np.zeros(len(sub), dtype=np.float64)
                if rank == 0
                else np.log(sub["regulator"].to_numpy(dtype=np.float64))
            )
            log_tam = np.log(sub["tamagawa"].to_numpy(dtype=np.float64))
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            tam_sub = sub["tamagawa"].to_numpy(dtype=int)
            tor_sub = sub["torsion"].to_numpy(dtype=int)
            sha_sub = np.rint(sub["sha_an"].to_numpy(dtype=float)).astype(int)
            strat_sub = sub["stratum_id"].to_numpy(dtype=int)
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)

            for p_int in TARGET_PRIMES:
                j = idx_primes[p_int]
                mask_b = (cond_sub % p_int) != 0
                ap_b = ap_sub[mask_b, j].astype(np.float64)
                l_b = l_val[mask_b]
                log_om_b = log_omega[mask_b]
                strat_b = strat_sub[mask_b]
                tam_b = tam_sub[mask_b]
                sha_b = sha_sub[mask_b]

                # E9.1
                bins_glob = np.asarray(
                    pd.qcut(l_b, q=100, labels=False, duplicates="drop"),
                    dtype=int,
                )
                rho_glob, nef_glob = partial_correlation_bins(
                    ap_b, log_om_b, bins_glob
                )

                bins_pooled = stratified_bins(
                    l_b, strat_b, q_max=100, n_min=200
                )
                m_val = bins_pooled >= 0
                rho_pooled, nef_pooled = partial_correlation_bins(
                    ap_b[m_val], log_om_b[m_val], bins_pooled[m_val]
                )

                # Pointwise strata: ∏c in {1, 2, 3, 4} and Sha == 1 (where ∏c and Sha have no within-stratum variation)
                m_pw = (tam_b <= 4) & (sha_b == 1)
                bins_pw = stratified_bins(
                    l_b[m_pw], strat_b[m_pw], q_max=100, n_min=200
                )
                mv_p = bins_pw >= 0
                rho_pw, nef_pw = partial_correlation_bins(
                    ap_b[m_pw][mv_p], log_om_b[m_pw][mv_p], bins_pw[mv_p]
                )

                # Pointwise strata with ∏c in {2, 3, 4} and Sha == 1
                m_c234 = (tam_b >= 2) & (tam_b <= 4) & (sha_b == 1)
                bins_c234 = stratified_bins(
                    l_b[m_c234], strat_b[m_c234], q_max=100, n_min=200
                )
                mv_234 = bins_c234 >= 0
                rho_c234, nef_234 = partial_correlation_bins(
                    ap_b[m_c234][mv_234],
                    log_om_b[m_c234][mv_234],
                    bins_c234[mv_234],
                )

                rows_e91.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": p_int,
                        "rho_global": rho_glob,
                        "rho_pooled_48": rho_pooled,
                        "ratio_pooled_over_glob": rho_pooled / rho_glob,
                        "rho_pointwise_c1234_sha1": rho_pw,
                        "rho_pointwise_c234_sha1": rho_c234,
                        "se_pooled": float(1.0 / np.sqrt(nef_pooled)),
                        "se_c234": float(1.0 / np.sqrt(nef_234)),
                    }
                )

                # E9.2 (at ranks 0 and 1, T = 1 only)
                if rank in (0, 1):
                    m_t1 = mask_b & (tor_sub == 1)
                    ap_t1 = ap_sub[m_t1, j]
                    ap_t1_f = ap_t1.astype(np.float64)
                    l_t1 = l_val[m_t1]
                    tam_t1 = log_tam[m_t1]
                    b_t1 = log_B[m_t1]

                    table_leg = [
                        0 if r == 0 else legendre_symbol(r, p_int)
                        for r in range(p_int)
                    ]
                    leg_all = np.array(
                        [table_leg[d % p_int] for d in discs_rv], dtype=np.int8
                    )
                    leg_t1 = leg_all[m_t1]

                    # 3 states: 0 = 0_roots ((Δ/p)=+1, odd ap), 1 = 1_root ((Δ/p)=-1, even ap), 2 = 3_roots ((Δ/p)=+1, even ap)
                    is_even = (ap_t1 % 2) == 0
                    m_0roots = (leg_t1 == 1) & (~is_even)
                    m_1root = leg_t1 == -1
                    m_3roots = (leg_t1 == 1) & is_even

                    state3 = np.zeros(len(ap_t1), dtype=int)
                    state3[m_1root] = 1
                    state3[m_3roots] = 2

                    med_tam_0 = float(np.mean(tam_t1[m_0roots]))
                    med_tam_1 = float(np.mean(tam_t1[m_1root]))
                    med_tam_3 = float(np.mean(tam_t1[m_3roots]))

                    v0 = float(np.var(tam_t1[m_0roots], ddof=1))
                    v3 = float(np.var(tam_t1[m_3roots], ddof=1))
                    n0 = int(np.sum(m_0roots))
                    n1 = int(np.sum(m_1root))
                    n3 = int(np.sum(m_3roots))
                    z_0_vs_3 = (med_tam_0 - med_tam_3) / math.sqrt(
                        v0 / n0 + v3 / n3
                    )

                    bins_L_t1 = np.asarray(
                        pd.qcut(l_t1, q=100, labels=False, duplicates="drop"),
                        dtype=int,
                    )
                    bins_B10 = np.asarray(
                        pd.qcut(b_t1, q=10, labels=False, duplicates="drop"),
                        dtype=int,
                    )

                    rho_base, _ = partial_correlation_bins(
                        ap_t1_f, tam_t1, bins_L_t1
                    )
                    rho_leg2, _ = partial_correlation_bins(
                        ap_t1_f,
                        tam_t1,
                        bins_L_t1 * 2 + (leg_t1 == 1).astype(int),
                    )
                    rho_strat3, _ = partial_correlation_bins(
                        ap_t1_f, tam_t1, bins_L_t1 * 3 + state3
                    )
                    rho_b10, _ = partial_correlation_bins(
                        ap_t1_f, tam_t1, bins_L_t1 * 20 + bins_B10
                    )

                    rows_e92.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "p": p_int,
                            "n_0roots": n0,
                            "n_1root": n1,
                            "n_3roots": n3,
                            "mean_logtam_0roots": med_tam_0,
                            "mean_logtam_1root": med_tam_1,
                            "mean_logtam_3roots": med_tam_3,
                            "z_welch_0_vs_3": z_0_vs_3,
                            "rho_base_L": rho_base,
                            "rho_L_x_leg2": rho_leg2,
                            "rho_L_x_state3": rho_strat3,
                            "rho_L_x_B10": rho_b10,
                            "pct_red_leg2": 100.0
                            * (1.0 - abs(rho_leg2) / abs(rho_base)),
                            "pct_red_state3": 100.0
                            * (1.0 - abs(rho_strat3) / abs(rho_base)),
                            "pct_red_B10": 100.0
                            * (1.0 - abs(rho_b10) / abs(rho_base)),
                        }
                    )

            # E9.3: in the pure stratum (stratum_id == 0) at ranks 1 and 2, split of Cov(ap, log L^(r) | N)
            if rank in (1, 2):
                for p_int in (2, 3, 5, 7):
                    j = idx_primes[p_int]
                    m_pure_b = ((cond_sub % p_int) != 0) & (strat_sub == 0)
                    n_pb = int(np.sum(m_pure_b))
                    if n_pb < 200:
                        continue
                    q_n = min(100, n_pb // 20)
                    bins_N_pure = np.asarray(
                        pd.qcut(
                            cond_sub[m_pure_b],
                            q=q_n,
                            labels=False,
                            duplicates="drop",
                        ),
                        dtype=int,
                    )
                    ap_pb = ap_sub[m_pure_b, j].astype(np.float64)
                    log_L_pb = log_L[m_pure_b]
                    log_R_pb = log_reg[m_pure_b]
                    log_om_pb = log_omega[m_pure_b]

                    cov_L_N = covariance_in_bins(ap_pb, log_L_pb, bins_N_pure)
                    cov_R_N = covariance_in_bins(ap_pb, log_R_pb, bins_N_pure)
                    cov_om_N = covariance_in_bins(ap_pb, log_om_pb, bins_N_pure)
                    rho_R_N, nef_pb = partial_correlation_bins(
                        ap_pb, log_R_pb, bins_N_pure
                    )
                    rho_om_N, _ = partial_correlation_bins(
                        ap_pb, log_om_pb, bins_N_pure
                    )

                    rows_e93.append(
                        {
                            "rank": rank,
                            "window": win_name,
                            "p": p_int,
                            "n_pure": n_pb,
                            "cov_L_given_N": cov_L_N,
                            "cov_R_given_N": cov_R_N,
                            "cov_omega_given_N": cov_om_N,
                            "pct_R": 100.0 * cov_R_N / cov_L_N,
                            "pct_omega": 100.0 * cov_om_N / cov_L_N,
                            "rho_R_given_N": rho_R_N,
                            "rho_omega_given_N": rho_om_N,
                            "se": float(1.0 / np.sqrt(nef_pb)),
                        }
                    )

    df_e91 = pd.DataFrame(rows_e91)
    df_e91.to_csv(EXP_DIR / "e9_1_r0_strata_and_simpson.csv", index=False)

    df_e92 = pd.DataFrame(rows_e92)
    df_e92.to_csv(EXP_DIR / "e9_2_three_2torsion_states.csv", index=False)

    df_e93 = pd.DataFrame(rows_e93)
    df_e93.to_csv(EXP_DIR / "e9_3_pure_regulator_split.csv", index=False)

    this_path = Path(__file__).resolve()
    (EXP_DIR / "e09_exploratory.py").write_text(
        this_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    t_end = time.perf_counter()
    print(f"Total time e09_exploratory: {t_end - t_start:.2f} s")
    print("\n--- E9.1: STRATA AT RANK 0 AND SIMPSON AT RANKS 1 AND 2 ---")
    print(df_e91.to_string(index=False))
    print("\n--- E9.2: THREE 2-TORSION STATES VS COLLIDER log B ---")
    print(df_e92.to_string(index=False))
    print("\n--- E9.3: EXACT SPLIT IN THE PURE STRATUM (log L = log Ω + log R) ---")
    print(df_e93.to_string(index=False))

    return {"time_s": t_end - t_start}


if __name__ == "__main__":
    run_exploratory_09()
