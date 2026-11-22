from __future__ import annotations

import math
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import DIR_DERIVADOS, cargar_cache_ap, discriminante
from lab.estadistica import (
    bins_estratificados,
    clave_estrato,
    correlacion_parcial_bins,
    correlacion_parcial_cov,
    residuos_parciales_cov,
)

N_PRIMOS = 25
RAIZ = Path(__file__).resolve().parents[2]
DIR_EXP = RAIZ / "resultados" / "10-cerrar-h16" / "exploratorio"
RUTA_EULER_MALOS = DIR_DERIVADOS / "euler_malos.csv"
RUTA_TIPOS_RED = DIR_DERIVADOS / "tipos_reduccion.csv"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]
PRIMOS_OBJ = (3, 5, 7)


def _clave_exacta_entera(cp: np.ndarray, T: np.ndarray, sha: np.ndarray) -> np.ndarray:
    cp_i = np.asarray(cp, dtype=np.int64)
    t_i = np.asarray(T, dtype=np.int64)
    s_i = np.rint(np.asarray(sha, dtype=np.float64)).astype(np.int64)
    # Código biyectivo sin agrupar colas (T <= 16, Sha <= 1000 en N < 100k)
    return (cp_i * 1000 + t_i) * 10000 + s_i


def _pendiente_en_bins(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> float:
    rx = residuos_parciales_cov(x, bins)
    ry = residuos_parciales_cov(y, bins)
    denom = float(np.sum(rx * rx))
    return float(np.sum(rx * ry) / denom) if denom > 0.0 else float("nan")


def ejecutar_exploratorio() -> dict:
    t_inicio = time.perf_counter()
    DIR_EXP.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)
    df_eb = pd.read_csv(RUTA_EULER_MALOS)
    df_tr = pd.read_csv(RUTA_TIPOS_RED)
    clases = clases.merge(df_eb, on="etiqueta", how="left")
    clases = clases.merge(df_tr, on="etiqueta", how="left")

    clases["log_disc"] = [
        math.log(abs(discriminante(ai))) for ai in clases["ainvs"]
    ]
    clases["log_N"] = np.log(clases["conductor"].to_numpy(dtype=np.float64))
    clases["log_exc"] = clases["log_disc"] - clases["log_N"]
    clases["log_omega"] = np.log(clases["omega"].to_numpy(dtype=np.float64))
    clases["log_R"] = np.log(clases["regulador"].to_numpy(dtype=np.float64))
    clases["log_L"] = np.log(clases["l_valor"].to_numpy(dtype=np.float64))

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primos = {int(p): j for j, p in enumerate(primos)}

    filas_e10_1: list[dict[str, str | int | float]] = []
    filas_e10_2a: list[dict[str, str | int | float]] = []
    filas_e10_2b: list[dict[str, str | int | float]] = []
    filas_e10_2c: list[dict[str, str | int | float]] = []
    filas_e10_3a: list[dict[str, str | int | float]] = []
    filas_e10_3b: list[dict[str, str | int | float]] = []

    for rango in (1, 2):
        for nombre_vent, n_min, n_max in VENTANAS:
            mask_rv = (
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = clases.loc[mask_rv].reset_index(drop=True)

            l_val = sub["l_valor"].to_numpy(dtype=np.float64)
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
            n_part = sub["n_partida"].to_numpy(dtype=np.int64)
            n_nopart = sub["n_no_partida"].to_numpy(dtype=np.int64)
            n_adit = sub["n_aditiva"].to_numpy(dtype=np.int64)
            n_malos = sub["n_malos"].to_numpy(dtype=np.int64)

            est_48 = clave_estrato(cp, T, sha)
            est_ex = _clave_exacta_entera(cp, T, sha)
            q_B = np.asarray(
                pd.qcut(log_B, q=5, labels=False, duplicates="drop"),
                dtype=np.int64,
            )
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)

            # ==================== E10.1 ====================
            for p_int in PRIMOS_OBJ:
                j = idx_primos[p_int]
                mb = (cond % p_int) != 0
                ap_b = ap_sub[mb, j].astype(np.float64)
                l_b = l_val[mb]
                logL_b = log_L[mb]
                om_b = log_om[mb]
                B_b = log_B[mb]
                disc_b = log_disc[mb]
                N_b = log_N[mb]
                exc_b = log_exc[mb]
                est_b = est_48[mb]

                bins_L = bins_estratificados(l_b, est_b, q_max=100, n_min=200)
                mv = bins_L >= 0
                ap_v = ap_b[mv]
                logL_v = logL_b[mv]
                om_v = om_b[mv]
                B_v = B_b[mv]
                disc_v = disc_b[mv]
                N_v = N_b[mv]
                exc_v = exc_b[mv]
                est_v = est_b[mv]
                bins_v = bins_L[mv]

                # Sin fijar L (solo dentro de estratos)
                rho_ap_om_sinL, _ = correlacion_parcial_bins(ap_v, om_v, est_v)
                rho_ap_disc_sinL, _ = correlacion_parcial_bins(ap_v, disc_v, est_v)
                rho_disc_L_sinL, _ = correlacion_parcial_bins(disc_v, logL_v, est_v)
                rho_disc_L_sinL_dadoB, _ = correlacion_parcial_cov(
                    disc_v, logL_v, est_v, B_v
                )

                # Fijando L dentro de estratos
                rho_base, _ = correlacion_parcial_bins(ap_v, om_v, bins_v)
                rho_ap_disc_L, _ = correlacion_parcial_bins(ap_v, disc_v, bins_v)
                r_D_om, _ = correlacion_parcial_bins(disc_v, om_v, bins_v)
                techo_pasivo_pct = 100.0 * (1.0 - math.sqrt(max(0.0, 1.0 - r_D_om * r_D_om)))

                rho_logB, _ = correlacion_parcial_cov(ap_v, om_v, bins_v, B_v)
                rho_logdisc, _ = correlacion_parcial_cov(ap_v, om_v, bins_v, disc_v)
                rho_logN, _ = correlacion_parcial_cov(ap_v, om_v, bins_v, N_v)
                rho_logexc, _ = correlacion_parcial_cov(ap_v, om_v, bins_v, exc_v)
                rho_B_N, _ = correlacion_parcial_cov(
                    ap_v, om_v, bins_v, np.column_stack([B_v, N_v])
                )
                rho_B_exc, _ = correlacion_parcial_cov(
                    ap_v, om_v, bins_v, np.column_stack([B_v, exc_v])
                )
                rho_B_disc, _ = correlacion_parcial_cov(
                    ap_v, om_v, bins_v, np.column_stack([B_v, disc_v])
                )
                pct_red_disc = 100.0 * (1.0 - abs(rho_logdisc) / abs(rho_base))

                filas_e10_1.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n": int(np.sum(mv)),
                        "rho_ap_om_sinL": rho_ap_om_sinL,
                        "rho_ap_disc_sinL": rho_ap_disc_sinL,
                        "rho_disc_L_sinL": rho_disc_L_sinL,
                        "rho_disc_L_sinL_dadoB": rho_disc_L_sinL_dadoB,
                        "rho_base_L": rho_base,
                        "rho_ap_disc_L": rho_ap_disc_L,
                        "r_D_omega": r_D_om,
                        "techo_pasivo_pct": techo_pasivo_pct,
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

            # ==================== E10.2a: Disección B vs f_partida, f_aditiva ====================
            n_mult = n_part + n_nopart
            m_mult = n_mult > 0
            f_part = n_part[m_mult].astype(np.float64) / n_mult[m_mult].astype(np.float64)
            f_adit = n_adit[m_mult].astype(np.float64) / n_malos[m_mult].astype(np.float64)
            Z_f = np.column_stack([f_part, f_adit])

            for esquema_nom, est_arr in [("estratos_48", est_48), ("estratos_enteros_exactos", est_ex)]:
                bins_m = bins_estratificados(l_val[m_mult], est_arr[m_mult], q_max=100, n_min=200)
                mv_m = bins_m >= 0
                B_m = log_B[m_mult][mv_m]
                fp_m = f_part[mv_m]
                Zf_m = Z_f[mv_m]
                bm = bins_m[mv_m]

                for obj_nom, obj_arr in [("log_disc", log_disc), ("log_exc", log_exc)]:
                    Y_m = obj_arr[m_mult][mv_m]
                    rho_B_Y, nef_m = correlacion_parcial_bins(B_m, Y_m, bm)
                    rho_fp_Y, _ = correlacion_parcial_bins(fp_m, Y_m, bm)
                    rho_B_Y_f, _ = correlacion_parcial_cov(B_m, Y_m, bm, Zf_m)
                    rho_fp_Y_B, _ = correlacion_parcial_cov(fp_m, Y_m, bm, B_m)
                    pct_red_B_f = 100.0 * (1.0 - abs(rho_B_Y_f) / abs(rho_B_Y))
                    filas_e10_2a.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "esquema": esquema_nom,
                            "objetivo": obj_nom,
                            "n": int(np.sum(mv_m)),
                            "se": float(1.0 / math.sqrt(nef_m)),
                            "rho_B_Y": rho_B_Y,
                            "rho_fp_Y": rho_fp_Y,
                            "rho_B_Y_dado_f": rho_B_Y_f,
                            "pct_red_B_por_f": pct_red_B_f,
                            "rho_fp_Y_dado_B": rho_fp_Y_B,
                        }
                    )

            # ==================== E10.2b: Por estrato individual (T=1, Sha=1) ====================
            grupos_c = [
                ("c=1", cp == 1),
                ("c=2", cp == 2),
                ("c=3", cp == 3),
                ("c=4", cp == 4),
                ("c=5-8", (cp >= 5) & (cp <= 8)),
                ("c>=9", cp >= 9),
                ("c=6_exacto", cp == 6),
                ("c=8_exacto", cp == 8),
                ("c=12_exacto", cp == 12),
            ]
            base_t1s1 = (T == 1) & (np.rint(sha) == 1)
            for c_nom, c_mask in grupos_c:
                m_ec = base_t1s1 & c_mask & m_mult
                n_ec = int(np.sum(m_ec))
                if n_ec < 200:
                    continue
                b_ec = bins_estratificados(
                    l_val[m_ec], np.zeros(n_ec, dtype=int), q_max=100, n_min=200
                )
                B_ec = log_B[m_ec]
                disc_ec = log_disc[m_ec]
                exc_ec = log_exc[m_ec]
                om_ec = log_om[m_ec]
                fp_ec = n_part[m_ec].astype(np.float64) / n_mult[m_ec].astype(np.float64)
                fa_ec = n_adit[m_ec].astype(np.float64) / n_malos[m_ec].astype(np.float64)

                r_B_d, nef_ec = correlacion_parcial_bins(B_ec, disc_ec, b_ec)
                r_fp_d, _ = correlacion_parcial_bins(fp_ec, disc_ec, b_ec)
                r_B_ex, _ = correlacion_parcial_bins(B_ec, exc_ec, b_ec)
                r_fp_ex, _ = correlacion_parcial_bins(fp_ec, exc_ec, b_ec)
                r_B_d_f, _ = correlacion_parcial_cov(
                    B_ec, disc_ec, b_ec, np.column_stack([fp_ec, fa_ec])
                )
                r_B_om, _ = correlacion_parcial_bins(B_ec, om_ec, b_ec)
                filas_e10_2b.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "estrato": f"{c_nom}|T=1|Sha=1",
                        "n": n_ec,
                        "se": float(1.0 / math.sqrt(nef_ec)),
                        "rho_B_logdisc": r_B_d,
                        "rho_fp_logdisc": r_fp_d,
                        "rho_B_logexc": r_B_ex,
                        "rho_fp_logexc": r_fp_ex,
                        "rho_B_logdisc_dado_f": r_B_d_f,
                        "rho_B_logom": r_B_om,
                    }
                )

            # ==================== E10.2c: P12 con estratos enteros exactos (sin colas 5-8, >=9) ====================
            for p_int in PRIMOS_OBJ:
                j = idx_primos[p_int]
                mb = (cond % p_int) != 0
                ap_b = ap_sub[mb, j].astype(np.float64)
                l_b = l_val[mb]
                om_b = log_om[mb]
                B_b = log_B[mb]
                disc_b = log_disc[mb]
                est_ex_b = est_ex[mb]
                qB_b = q_B[mb]

                bins_ex = bins_estratificados(l_b, est_ex_b, q_max=100, n_min=200)
                mv_ex = bins_ex >= 0
                r_ex_base, nef_ex = correlacion_parcial_bins(
                    ap_b[mv_ex], om_b[mv_ex], bins_ex[mv_ex]
                )
                est_ex_qB = est_ex_b * 10 + qB_b
                bins_ex_qB = bins_estratificados(l_b, est_ex_qB, q_max=100, n_min=200)
                mv_ex_qB = bins_ex_qB >= 0
                r_ex_qB, _ = correlacion_parcial_bins(
                    ap_b[mv_ex_qB], om_b[mv_ex_qB], bins_ex_qB[mv_ex_qB]
                )
                r_ex_logB, _ = correlacion_parcial_cov(
                    ap_b[mv_ex], om_b[mv_ex], bins_ex[mv_ex], B_b[mv_ex]
                )
                r_ex_logdisc, _ = correlacion_parcial_cov(
                    ap_b[mv_ex], om_b[mv_ex], bins_ex[mv_ex], disc_b[mv_ex]
                )
                filas_e10_2c.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n_exacto": int(np.sum(mv_ex)),
                        "se_exacto": float(1.0 / math.sqrt(nef_ex)),
                        "rho_exacto": r_ex_base,
                        "n_exacto_qB": int(np.sum(mv_ex_qB)),
                        "rho_exacto_qB": r_ex_qB,
                        "pct_red_qB": 100.0 * (1.0 - abs(r_ex_qB) / abs(r_ex_base)),
                        "rho_exacto_logB": r_ex_logB,
                        "pct_red_logB": 100.0 * (1.0 - abs(r_ex_logB) / abs(r_ex_base)),
                        "rho_exacto_logdisc": r_ex_logdisc,
                        "pct_red_logdisc": 100.0 * (1.0 - abs(r_ex_logdisc) / abs(r_ex_base)),
                    }
                )

            # ==================== E10.3a: Pendientes d(log Omega, log R, log L)/d(log|Delta|) ====================
            # Comparar puro (c=1, T=1, Sha=1) vs no_puro exacto (c>=2, T=1, Sha=1, por valor entero de c)
            m_puro = (cp == 1) & (T == 1) & (np.rint(sha) == 1)
            m_c2p = (cp >= 2) & (T == 1) & (np.rint(sha) == 1)
            for sub_nom, m_sub, est_sub in [
                ("puro_c1_T1_Sha1", m_puro, np.zeros(len(sub), dtype=int)),
                ("no_puro_c>=2_enteros_T1_Sha1", m_c2p, est_ex),
            ]:
                # Quedarse con estratos con n >= 200
                vals_e, cnts_e = np.unique(est_sub[m_sub], return_counts=True)
                validos = set(vals_e[cnts_e >= 200].tolist())
                m_use = m_sub & np.isin(est_sub, list(validos))
                b_use = est_sub[m_use]
                d_u = log_disc[m_use]
                om_u = log_om[m_use]
                R_u = log_R[m_use]
                L_u = log_L[m_use]
                B_u = log_B[m_use]

                rho_d_L, nef_u = correlacion_parcial_bins(d_u, L_u, b_use)
                rho_d_L_dadoB, _ = correlacion_parcial_cov(d_u, L_u, b_use, B_u)
                rho_d_om, _ = correlacion_parcial_bins(d_u, om_u, b_use)
                rho_d_R, _ = correlacion_parcial_bins(d_u, R_u, b_use)
                b_om = _pendiente_en_bins(d_u, om_u, b_use)
                b_R = _pendiente_en_bins(d_u, R_u, b_use)
                b_L = _pendiente_en_bins(d_u, L_u, b_use)
                filas_e10_3a.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "subconjunto": sub_nom,
                        "n": int(np.sum(m_use)),
                        "se": float(1.0 / math.sqrt(nef_u)),
                        "rho_logdisc_logL": rho_d_L,
                        "rho_logdisc_logL_dadoB": rho_d_L_dadoB,
                        "rho_logdisc_logom": rho_d_om,
                        "rho_logdisc_logR": rho_d_R,
                        "pend_logom_sobre_logdisc": b_om,
                        "pend_logR_sobre_logdisc": b_R,
                        "pend_logL_sobre_logdisc": b_L,
                        "error_suma_pend": abs((b_om + b_R) - b_L),
                    }
                )

            # ==================== E10.3b: Estrato puro en p = 2, 3, 5, 7 ====================
            for p_int in (2, 3, 5, 7):
                j = idx_primos[p_int]
                mp = m_puro & ((cond % p_int) != 0)
                n_p = int(np.sum(mp))
                if n_p < 200:
                    continue
                b_p = bins_estratificados(
                    l_val[mp], np.zeros(n_p, dtype=int), q_max=100, n_min=200
                )
                ap_p = ap_sub[mp, j].astype(np.float64)
                om_p = log_om[mp]
                B_p = log_B[mp]
                N_p = log_N[mp]
                disc_p = log_disc[mp]

                r_base, nef_p = correlacion_parcial_bins(ap_p, om_p, b_p)
                se_p = float(1.0 / math.sqrt(nef_p))
                r_B, _ = correlacion_parcial_cov(ap_p, om_p, b_p, B_p)
                r_N, _ = correlacion_parcial_cov(ap_p, om_p, b_p, N_p)
                r_d, _ = correlacion_parcial_cov(ap_p, om_p, b_p, disc_p)
                r_ap_B, _ = correlacion_parcial_bins(ap_p, B_p, b_p)
                r_B_om, _ = correlacion_parcial_bins(B_p, om_p, b_p)
                r_B_d, _ = correlacion_parcial_bins(B_p, disc_p, b_p)
                filas_e10_3b.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n_puro": n_p,
                        "se": se_p,
                        "rho_puro_base": r_base,
                        "z_base": r_base / se_p,
                        "rho_puro_logB": r_B,
                        "rho_puro_logN": r_N,
                        "rho_puro_logdisc": r_d,
                        "rho_ap_logB": r_ap_B,
                        "rho_logB_logom": r_B_om,
                        "rho_logB_logdisc": r_B_d,
                    }
                )

    df_e1 = pd.DataFrame(filas_e10_1)
    df_e2a = pd.DataFrame(filas_e10_2a)
    df_e2b = pd.DataFrame(filas_e10_2b)
    df_e2c = pd.DataFrame(filas_e10_2c)
    df_e3a = pd.DataFrame(filas_e10_3a)
    df_e3b = pd.DataFrame(filas_e10_3b)

    df_e1.to_csv(DIR_EXP / "e10_1_diagnostico_riesgo.csv", index=False)
    df_e2a.to_csv(DIR_EXP / "e10_2a_diseccion_B_disc.csv", index=False)
    df_e2b.to_csv(DIR_EXP / "e10_2b_por_estrato.csv", index=False)
    df_e2c.to_csv(DIR_EXP / "e10_2c_estratos_puntuales.csv", index=False)
    df_e3a.to_csv(DIR_EXP / "e10_3a_pendientes_puro_vs_c2.csv", index=False)
    df_e3b.to_csv(DIR_EXP / "e10_3b_puro_controles.csv", index=False)

    # Guardar copia del código en resultados/10-cerrar-h16/exploratorio/
    shutil.copy2(Path(__file__), DIR_EXP / "e10_exploratorio.py")

    t_fin = time.perf_counter()
    print(f"Tiempo total e10_exploratorio: {t_fin - t_inicio:.2f} s\n")

    print("=== E10.1: DIAGNÓSTICO DEL RIESGO REGISTRADO ===")
    print(df_e1.to_string(index=False))

    print("\n=== E10.2a: DISECCIÓN log B vs f_partida, f_aditiva ===")
    print(df_e2a.to_string(index=False))

    print("\n=== E10.2b: POR ESTRATO INDIVIDUAL (T=1, Sha=1) ===")
    print(df_e2b.to_string(index=False))

    print("\n=== E10.2c: P12 EN ESTRATOS ENTEROS EXACTOS (SIN COLAS) ===")
    print(df_e2c.to_string(index=False))

    print("\n=== E10.3a: PENDIENTES d(log Omega, log R, log L)/d(log|Delta|) ===")
    print(df_e3a.to_string(index=False))

    print("\n=== E10.3b: ESTRATO PURO (p=2,3,5,7) ===")
    print(df_e3b.to_string(index=False))

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_exploratorio()
