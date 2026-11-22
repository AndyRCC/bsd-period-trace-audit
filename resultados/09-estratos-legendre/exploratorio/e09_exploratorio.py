from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import DIR_DERIVADOS, cargar_cache_ap, discriminante, simbolo_legendre
from lab.estadistica import (
    bins_estratificados,
    correlacion_parcial_bins,
    covarianza_en_bins,
)
from lab.experimentos.e09_estratos import PRIMOS_OBJ, VENTANAS, _codificar_estratos

N_PRIMOS = 25
RAIZ = Path(__file__).resolve().parents[2]
DIR_EXP = RAIZ / "resultados" / "09-estratos-legendre" / "exploratorio"
RUTA_EULER_MALOS = DIR_DERIVADOS / "euler_malos.csv"


def ejecutar_exploratorio_09() -> dict:
    t_inicio = time.perf_counter()
    DIR_EXP.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)
    df_eb = pd.read_csv(RUTA_EULER_MALOS)
    clases = clases.merge(df_eb, on="etiqueta", how="left")
    discs_py = [discriminante(ai) for ai in clases["ainvs"]]

    tam_all = clases["tamagawa"].to_numpy(dtype=int)
    tor_all = clases["torsion"].to_numpy(dtype=int)
    sha_all = np.rint(clases["sha_an"].to_numpy(dtype=float)).astype(int)
    est_id_all, _ = _codificar_estratos(tam_all, tor_all, sha_all)
    clases["estrato_id"] = est_id_all

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primos = {int(p): j for j, p in enumerate(primos)}

    # E9.1: Estratos exactos en Rango 0 y estratos puntuales (∏c in {1,2,3,4}, Sha=1) en Rangos 0, 1, 2
    filas_e91: list[dict[str, str | int | float]] = []
    # E9.2: 3 estados de 2-torsión en F_p (0_raices, 1_raiz, 3_raices) vs log B en T = 1
    filas_e92: list[dict[str, str | int | float]] = []
    # E9.3: Reparto exacto de Cov(ap, log L^(r) | N) entre log R y log Ω en el estrato puro de Rangos 1 y 2
    filas_e93: list[dict[str, str | int | float]] = []

    for rango in (0, 1, 2):
        for nombre_vent, n_min, n_max in VENTANAS:
            mask_rv = (
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = clases.loc[mask_rv].reset_index(drop=True)
            discs_rv = [d for d, m in zip(discs_py, mask_rv) if m]

            l_val = sub["l_valor"].to_numpy(dtype=np.float64)
            log_L = np.log(l_val)
            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
            log_reg = (
                np.zeros(len(sub), dtype=np.float64)
                if rango == 0
                else np.log(sub["regulador"].to_numpy(dtype=np.float64))
            )
            log_tam = np.log(sub["tamagawa"].to_numpy(dtype=np.float64))
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            tam_sub = sub["tamagawa"].to_numpy(dtype=int)
            tor_sub = sub["torsion"].to_numpy(dtype=int)
            sha_sub = np.rint(sub["sha_an"].to_numpy(dtype=float)).astype(int)
            est_sub = sub["estrato_id"].to_numpy(dtype=int)
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)

            for p_int in PRIMOS_OBJ:
                j = idx_primos[p_int]
                mask_b = (cond_sub % p_int) != 0
                ap_b = ap_sub[mask_b, j].astype(np.float64)
                l_b = l_val[mask_b]
                log_om_b = log_omega[mask_b]
                est_b = est_sub[mask_b]
                tam_b = tam_sub[mask_b]
                sha_b = sha_sub[mask_b]

                # E9.1
                bins_glob = np.asarray(
                    pd.qcut(l_b, q=100, labels=False, duplicates="drop"),
                    dtype=int,
                )
                rho_glob, nef_glob = correlacion_parcial_bins(
                    ap_b, log_om_b, bins_glob
                )

                bins_agrup = bins_estratificados(
                    l_b, est_b, q_max=100, n_min=200
                )
                m_val = bins_agrup >= 0
                rho_agrup, nef_agrup = correlacion_parcial_bins(
                    ap_b[m_val], log_om_b[m_val], bins_agrup[m_val]
                )

                # Estratos puntuales: ∏c in {1, 2, 3, 4} y Sha == 1 (donde ∏c y Sha no tienen variación intra-estrato)
                m_punt = (tam_b <= 4) & (sha_b == 1)
                bins_punt = bins_estratificados(
                    l_b[m_punt], est_b[m_punt], q_max=100, n_min=200
                )
                mv_p = bins_punt >= 0
                rho_punt, nef_punt = correlacion_parcial_bins(
                    ap_b[m_punt][mv_p], log_om_b[m_punt][mv_p], bins_punt[mv_p]
                )

                # Estratos puntuales con ∏c in {2, 3, 4} y Sha == 1
                m_c234 = (tam_b >= 2) & (tam_b <= 4) & (sha_b == 1)
                bins_c234 = bins_estratificados(
                    l_b[m_c234], est_b[m_c234], q_max=100, n_min=200
                )
                mv_234 = bins_c234 >= 0
                rho_c234, nef_234 = correlacion_parcial_bins(
                    ap_b[m_c234][mv_234],
                    log_om_b[m_c234][mv_234],
                    bins_c234[mv_234],
                )

                filas_e91.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "rho_global": rho_glob,
                        "rho_agrupada_48": rho_agrup,
                        "ratio_agrup_sobre_glob": rho_agrup / rho_glob,
                        "rho_puntual_c1234_sha1": rho_punt,
                        "rho_puntual_c234_sha1": rho_c234,
                        "se_agrup": float(1.0 / np.sqrt(nef_agrup)),
                        "se_c234": float(1.0 / np.sqrt(nef_234)),
                    }
                )

                # E9.2 (en rangos 0 y 1, solo T = 1)
                if rango in (0, 1):
                    m_t1 = mask_b & (tor_sub == 1)
                    ap_t1 = ap_sub[m_t1, j]
                    ap_t1_f = ap_t1.astype(np.float64)
                    l_t1 = l_val[m_t1]
                    tam_t1 = log_tam[m_t1]
                    b_t1 = log_B[m_t1]

                    tabla_leg = [
                        0 if r == 0 else simbolo_legendre(r, p_int)
                        for r in range(p_int)
                    ]
                    leg_all = np.array(
                        [tabla_leg[d % p_int] for d in discs_rv], dtype=np.int8
                    )
                    leg_t1 = leg_all[m_t1]

                    # 3 estados: 0 = 0_raices ((Δ/p)=+1, ap impar), 1 = 1_raiz ((Δ/p)=-1, ap par), 2 = 3_raices ((Δ/p)=+1, ap par)
                    es_par = (ap_t1 % 2) == 0
                    m_0raices = (leg_t1 == 1) & (~es_par)
                    m_1raiz = leg_t1 == -1
                    m_3raices = (leg_t1 == 1) & es_par

                    estado3 = np.zeros(len(ap_t1), dtype=int)
                    estado3[m_1raiz] = 1
                    estado3[m_3raices] = 2

                    med_tam_0 = float(np.mean(tam_t1[m_0raices]))
                    med_tam_1 = float(np.mean(tam_t1[m_1raiz]))
                    med_tam_3 = float(np.mean(tam_t1[m_3raices]))

                    v0 = float(np.var(tam_t1[m_0raices], ddof=1))
                    v3 = float(np.var(tam_t1[m_3raices], ddof=1))
                    n0 = int(np.sum(m_0raices))
                    n1 = int(np.sum(m_1raiz))
                    n3 = int(np.sum(m_3raices))
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

                    rho_base, _ = correlacion_parcial_bins(
                        ap_t1_f, tam_t1, bins_L_t1
                    )
                    rho_leg2, _ = correlacion_parcial_bins(
                        ap_t1_f,
                        tam_t1,
                        bins_L_t1 * 2 + (leg_t1 == 1).astype(int),
                    )
                    rho_est3, _ = correlacion_parcial_bins(
                        ap_t1_f, tam_t1, bins_L_t1 * 3 + estado3
                    )
                    rho_b10, _ = correlacion_parcial_bins(
                        ap_t1_f, tam_t1, bins_L_t1 * 20 + bins_B10
                    )

                    filas_e92.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "p": p_int,
                            "n_0raices": n0,
                            "n_1raiz": n1,
                            "n_3raices": n3,
                            "media_logtam_0raices": med_tam_0,
                            "media_logtam_1raiz": med_tam_1,
                            "media_logtam_3raices": med_tam_3,
                            "z_welch_0_vs_3": z_0_vs_3,
                            "rho_base_L": rho_base,
                            "rho_L_x_leg2": rho_leg2,
                            "rho_L_x_estado3": rho_est3,
                            "rho_L_x_B10": rho_b10,
                            "pct_red_leg2": 100.0
                            * (1.0 - abs(rho_leg2) / abs(rho_base)),
                            "pct_red_estado3": 100.0
                            * (1.0 - abs(rho_est3) / abs(rho_base)),
                            "pct_red_B10": 100.0
                            * (1.0 - abs(rho_b10) / abs(rho_base)),
                        }
                    )

            # E9.3: En el estrato puro (estrato_id == 0) de rangos 1 y 2, reparto de Cov(ap, log L^(r) | N)
            if rango in (1, 2):
                for p_int in (2, 3, 5, 7):
                    j = idx_primos[p_int]
                    m_puro_b = ((cond_sub % p_int) != 0) & (est_sub == 0)
                    n_pb = int(np.sum(m_puro_b))
                    if n_pb < 200:
                        continue
                    q_n = min(100, n_pb // 20)
                    bins_N_puro = np.asarray(
                        pd.qcut(
                            cond_sub[m_puro_b],
                            q=q_n,
                            labels=False,
                            duplicates="drop",
                        ),
                        dtype=int,
                    )
                    ap_pb = ap_sub[m_puro_b, j].astype(np.float64)
                    log_L_pb = log_L[m_puro_b]
                    log_R_pb = log_reg[m_puro_b]
                    log_om_pb = log_omega[m_puro_b]

                    cov_L_N = covarianza_en_bins(ap_pb, log_L_pb, bins_N_puro)
                    cov_R_N = covarianza_en_bins(ap_pb, log_R_pb, bins_N_puro)
                    cov_om_N = covarianza_en_bins(ap_pb, log_om_pb, bins_N_puro)
                    rho_R_N, nef_pb = correlacion_parcial_bins(
                        ap_pb, log_R_pb, bins_N_puro
                    )
                    rho_om_N, _ = correlacion_parcial_bins(
                        ap_pb, log_om_pb, bins_N_puro
                    )

                    filas_e93.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "p": p_int,
                            "n_puro": n_pb,
                            "cov_L_dado_N": cov_L_N,
                            "cov_R_dado_N": cov_R_N,
                            "cov_omega_dado_N": cov_om_N,
                            "pct_R": 100.0 * cov_R_N / cov_L_N,
                            "pct_omega": 100.0 * cov_om_N / cov_L_N,
                            "rho_R_dado_N": rho_R_N,
                            "rho_omega_dado_N": rho_om_N,
                            "se": float(1.0 / np.sqrt(nef_pb)),
                        }
                    )

    df_e91 = pd.DataFrame(filas_e91)
    df_e91.to_csv(DIR_EXP / "e9_1_estratos_r0_y_simpson.csv", index=False)

    df_e92 = pd.DataFrame(filas_e92)
    df_e92.to_csv(DIR_EXP / "e9_2_tres_estados_2torsion.csv", index=False)

    df_e93 = pd.DataFrame(filas_e93)
    df_e93.to_csv(DIR_EXP / "e9_3_reparto_puro_regulador.csv", index=False)

    ruta_este = Path(__file__).resolve()
    (DIR_EXP / "e09_exploratorio.py").write_text(
        ruta_este.read_text(encoding="utf-8"), encoding="utf-8"
    )

    t_fin = time.perf_counter()
    print(f"Tiempo total e09_exploratorio: {t_fin - t_inicio:.2f} s")
    print("\n--- E9.1: ESTRATOS EN RANGO 0 Y SIMPSON EN RANGOS 1 Y 2 ---")
    print(df_e91.to_string(index=False))
    print("\n--- E9.2: TRES ESTADOS DE 2-TORSIÓN VS COLISIONADOR log B ---")
    print(df_e92.to_string(index=False))
    print("\n--- E9.3: REPARTO EXACTO EN ESTRATO PURO (log L = log Ω + log R) ---")
    print(df_e93.to_string(index=False))

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_exploratorio_09()
