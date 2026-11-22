from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import cargar_cache_ap, discriminante, simbolo_legendre
from lab.estadistica import bins_estratificados, correlacion_parcial_bins

N_PRIMOS = 25
RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "09-estratos-legendre"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]
PRIMOS_OBJ = (3, 5, 7)


def _codificar_estratos(
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

    est_id = c_code * 8 + t_code * 2 + s_code

    c_names = ["1", "2", "3", "4", "5-8", ">=9"]
    t_names = ["1", "2", "3", ">=4"]
    s_names = ["1", ">=4"]
    mapa_nombres: dict[int, str] = {}
    for ic, cn in enumerate(c_names):
        for it, tn in enumerate(t_names):
            for is_, sn in enumerate(s_names):
                mapa_nombres[ic * 8 + it * 2 + is_] = f"c={cn}|T={tn}|Sha={sn}"

    return est_id, mapa_nombres


def ejecutar_estratos_legendre() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)
    discs_py = [discriminante(ai) for ai in clases["ainvs"]]

    tam_all = clases["tamagawa"].to_numpy(dtype=int)
    tor_all = clases["torsion"].to_numpy(dtype=int)
    sha_all = np.rint(clases["sha_an"].to_numpy(dtype=float)).astype(int)
    est_id_all, mapa_est = _codificar_estratos(tam_all, tor_all, sha_all)
    clases["estrato_id"] = est_id_all

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primos = {int(p): j for j, p in enumerate(primos)}

    filas_estratos: list[dict[str, str | int | float]] = []
    filas_leg: list[dict[str, str | int | float]] = []

    # Parte A: Composición en rangos 1 y 2
    for rango in (1, 2):
        for nombre_vent, n_min, n_max in VENTANAS:
            mask_rv = (
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = clases.loc[mask_rv].reset_index(drop=True)

            l_val = sub["l_valor"].to_numpy(dtype=np.float64)
            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            est_sub = sub["estrato_id"].to_numpy(dtype=int)
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)

            for p_int in PRIMOS_OBJ:
                j = idx_primos[p_int]
                mask_b = (cond_sub % p_int) != 0
                ap_b = ap_sub[mask_b, j].astype(np.float64)
                l_b = l_val[mask_b]
                log_om_b = log_omega[mask_b]
                est_b = est_sub[mask_b]

                # 1. Global (100 cuantiles de L sobre toda la muestra p ∤ N)
                bins_glob = np.asarray(
                    pd.qcut(l_b, q=100, labels=False, duplicates="drop"),
                    dtype=int,
                )
                rho_glob, nef_glob = correlacion_parcial_bins(
                    ap_b, log_om_b, bins_glob
                )
                filas_estratos.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "tipo": "global",
                        "n": int(len(ap_b)),
                        "rho": rho_glob,
                        "se": float(1.0 / np.sqrt(nef_glob)),
                    }
                )

                # 2. Agrupada dentro de estratos exactos con bins_estratificados
                bins_agrup = bins_estratificados(
                    l_b, est_b, q_max=100, n_min=200
                )
                m_val = bins_agrup >= 0
                rho_agrup, nef_agrup = correlacion_parcial_bins(
                    ap_b[m_val], log_om_b[m_val], bins_agrup[m_val]
                )
                filas_estratos.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "tipo": "agrupada",
                        "n": int(np.sum(m_val)),
                        "rho": rho_agrup,
                        "se": float(1.0 / np.sqrt(nef_agrup)),
                    }
                )

                # 3. Estrato puro (estrato_id == 0: c=1|T=1|Sha=1)
                m_puro = est_b == 0
                n_puro = int(np.sum(m_puro))
                bins_puro = bins_estratificados(
                    l_b[m_puro],
                    np.zeros(n_puro, dtype=int),
                    q_max=100,
                    n_min=200,
                )
                rho_puro, nef_puro = correlacion_parcial_bins(
                    ap_b[m_puro], log_om_b[m_puro], bins_puro
                )
                filas_estratos.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "tipo": "puro",
                        "n": n_puro,
                        "rho": rho_puro,
                        "se": float(1.0 / np.sqrt(nef_puro)),
                    }
                )

                # 4. Cada estrato individual que tenga n >= 200
                for eid in sorted(np.unique(est_b)):
                    m_e = est_b == eid
                    n_e = int(np.sum(m_e))
                    if n_e < 200:
                        continue
                    b_e = bins_estratificados(
                        l_b[m_e],
                        np.zeros(n_e, dtype=int),
                        q_max=100,
                        n_min=200,
                    )
                    rho_e, nef_e = correlacion_parcial_bins(
                        ap_b[m_e], log_om_b[m_e], b_e
                    )
                    filas_estratos.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "p": p_int,
                            "tipo": f"estrato:{mapa_est[int(eid)]}",
                            "n": n_e,
                            "rho": rho_e,
                            "se": float(1.0 / np.sqrt(nef_e)),
                        }
                    )

    # Parte B: Vía (Δ/p) en rangos 0 y 1, solo T = 1, p ∈ {3, 5, 7}
    for rango in (0, 1):
        for nombre_vent, n_min, n_max in VENTANAS:
            mask_rv_t1 = (
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
                & (clases["torsion"] == 1)
            ).to_numpy(dtype=bool)
            sub_t1 = clases.loc[mask_rv_t1].reset_index(drop=True)
            discs_t1 = [d for d, m in zip(discs_py, mask_rv_t1) if m]

            l_t1 = sub_t1["l_valor"].to_numpy(dtype=np.float64)
            log_tam_t1 = np.log(sub_t1["tamagawa"].to_numpy(dtype=np.float64))
            cond_t1 = sub_t1["conductor"].to_numpy(dtype=np.int32)
            ap_t1 = df_ap.loc[list(sub_t1["etiqueta"])].to_numpy(dtype=np.int16)

            for p_int in PRIMOS_OBJ:
                j = idx_primos[p_int]
                mask_b = (cond_t1 % p_int) != 0
                ap_b = ap_t1[mask_b, j]
                ap_f = ap_b.astype(np.float64)
                l_b = l_t1[mask_b]
                log_tam_b = log_tam_t1[mask_b]

                tabla_leg = [
                    0 if r == 0 else simbolo_legendre(r, p_int)
                    for r in range(p_int)
                ]
                leg_all = np.array(
                    [tabla_leg[d % p_int] for d in discs_t1], dtype=np.int8
                )
                leg_b = leg_all[mask_b]

                # (i) Cordura: % de clases con ap impar y (Δ/p) = -1
                es_impar = (ap_b % 2) != 0
                n_impar_leg_menos = int(np.sum(es_impar & (leg_b == -1)))
                pct_cordura = 100.0 * n_impar_leg_menos / len(ap_b)

                # (ii) Media de log ∏c según (Δ/p) = ±1, con z de Welch
                m_mas = leg_b == 1
                m_menos = leg_b == -1
                n_mas = int(np.sum(m_mas))
                n_menos = int(np.sum(m_menos))
                med_tam_mas = float(np.mean(log_tam_b[m_mas]))
                med_tam_menos = float(np.mean(log_tam_b[m_menos]))
                var_mas = float(np.var(log_tam_b[m_mas], ddof=1))
                var_menos = float(np.var(log_tam_b[m_menos], ddof=1))
                se_leg = math.sqrt(var_mas / n_mas + var_menos / n_menos)
                delta_leg = med_tam_mas - med_tam_menos
                z_welch_leg = delta_leg / se_leg if se_leg > 0 else 0.0

                # (iii) ρ(ap, log ∏c | L) con bins de L, y con bins L × (Δ/p)
                bins_L = np.asarray(
                    pd.qcut(l_b, q=100, labels=False, duplicates="drop"),
                    dtype=int,
                )
                bins_L_leg = bins_L * 2 + (leg_b == 1).astype(int)
                rho_tam_L, _ = correlacion_parcial_bins(ap_f, log_tam_b, bins_L)
                rho_tam_L_leg, _ = correlacion_parcial_bins(
                    ap_f, log_tam_b, bins_L_leg
                )
                pct_red_rho = 100.0 * (
                    1.0 - abs(rho_tam_L_leg) / abs(rho_tam_L)
                )

                # (iv) Dentro de (Δ/p) = +1: media de log ∏c con ap par frente a impar
                ap_mas = ap_b[m_mas]
                tam_mas = log_tam_b[m_mas]
                es_par_mas = (ap_mas % 2) == 0
                n_par_mas = int(np.sum(es_par_mas))
                n_imp_mas = int(np.sum(~es_par_mas))
                med_tam_par_mas = float(np.mean(tam_mas[es_par_mas]))
                med_tam_imp_mas = float(np.mean(tam_mas[~es_par_mas]))
                v_par_mas = float(np.var(tam_mas[es_par_mas], ddof=1))
                v_imp_mas = float(np.var(tam_mas[~es_par_mas], ddof=1))
                se_par_imp = math.sqrt(
                    v_par_mas / n_par_mas + v_imp_mas / n_imp_mas
                )
                delta_par_imp_mas = med_tam_par_mas - med_tam_imp_mas
                z_par_imp_mas = (
                    delta_par_imp_mas / se_par_imp if se_par_imp > 0 else 0.0
                )

                filas_leg.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n_T1": int(len(ap_b)),
                        "pct_ap_impar_leg_menos": pct_cordura,
                        "n_leg_mas": n_mas,
                        "n_leg_menos": n_menos,
                        "media_logtam_leg_mas": med_tam_mas,
                        "media_logtam_leg_menos": med_tam_menos,
                        "delta_logtam_leg": delta_leg,
                        "z_welch_leg": z_welch_leg,
                        "rho_ap_logtam_L": rho_tam_L,
                        "rho_ap_logtam_L_leg": rho_tam_L_leg,
                        "pct_reduccion_rho": pct_red_rho,
                        "n_legmas_ap_par": n_par_mas,
                        "n_legmas_ap_impar": n_imp_mas,
                        "media_logtam_legmas_ap_par": med_tam_par_mas,
                        "media_logtam_legmas_ap_impar": med_tam_imp_mas,
                        "delta_logtam_legmas_par_impar": delta_par_imp_mas,
                        "z_welch_legmas_par_impar": z_par_imp_mas,
                    }
                )

    df_est = pd.DataFrame(filas_estratos)
    df_est.to_csv(DIR_SALIDA / "estratos.csv", index=False)

    df_leg = pd.DataFrame(filas_leg)
    df_leg.to_csv(DIR_SALIDA / "legendre_tamagawa.csv", index=False)

    # Figura estratos.png: para rangos 1 y 2, ρ global, agrupada y estrato puro en p = 3, 5, 7 con barras ±2 EE
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=150, sharey=True)
    tipos_plot = [
        ("global", "o-", "#1f5a99", -0.15, "Global (bins de L)"),
        ("agrupada", "s-", "#c0392b", 0.0, "Agrupada dentro de estratos exactos"),
        ("puro", "^--", "#27ae60", 0.15, "Estrato puro (∏c=1, T=1, Sha=1)"),
    ]
    etiquetas_x = [
        "p=3\n(1–50k)",
        "p=5\n(1–50k)",
        "p=7\n(1–50k)",
        "p=3\n(50k–100k)",
        "p=5\n(50k–100k)",
        "p=7\n(50k–100k)",
    ]
    pares_xp = [
        ("1-49999", 3),
        ("1-49999", 5),
        ("1-49999", 7),
        ("50000-99999", 3),
        ("50000-99999", 5),
        ("50000-99999", 7),
    ]
    x_base = np.arange(len(pares_xp), dtype=float)

    for ax, rango, titulo in [
        (axes[0], 1, "A. Rango 1: ρ(aₚ, log Ω | L′(1)) ± 2 EE"),
        (axes[1], 2, "B. Rango 2: ρ(aₚ, log Ω | L″(1)/2) ± 2 EE"),
    ]:
        sub_r = df_est.loc[df_est["rango"] == rango]
        for tipo_key, fmt, col, dx, label in tipos_plot:
            ys = []
            errs = []
            for vent, p_val in pares_xp:
                fila = sub_r.loc[
                    (sub_r["ventana"] == vent)
                    & (sub_r["p"] == p_val)
                    & (sub_r["tipo"] == tipo_key)
                ].iloc[0]
                ys.append(float(fila["rho"]))
                errs.append(2.0 * float(fila["se"]))
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
        ax.set_xticklabels(etiquetas_x)
        ax.set_ylabel("Correlación parcial ρ(aₚ, log Ω | L)")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.8)

    fig.suptitle(
        "Efecto de composición entre estratos exactos de (∏c, T, Sha) en rangos 1 y 2",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "estratos.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"Tiempo total e09_estratos: {t_fin - t_inicio:.2f} s")

    print("\n--- PARTE A: RESUMEN DE ESTRATOS (global, agrupada, puro) ---")
    print(
        df_est.loc[df_est["tipo"].isin(["global", "agrupada", "puro"])].to_string(
            index=False
        )
    )

    print("\n--- PARTE B: VÍA LEGENDRE-TAMAGAWA (T = 1) ---")
    print(df_leg.to_string(index=False))

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_estratos_legendre()
