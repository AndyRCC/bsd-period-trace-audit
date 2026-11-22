from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import DIR_DERIVADOS, cargar_cache_ap, discriminante
from lab.estadistica import correlacion_parcial_bins, correlacion_parcial_cov

N_PRIMOS = 25
RAIZ = Path(__file__).resolve().parents[2]
DIR_EXP = RAIZ / "resultados" / "08-rango1" / "exploratorio"
RUTA_EULER_MALOS = DIR_DERIVADOS / "euler_malos.csv"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def ejecutar_exploratorio() -> dict:
    t_inicio = time.perf_counter()
    DIR_EXP.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)
    df_eb = pd.read_csv(RUTA_EULER_MALOS)
    clases = clases.merge(df_eb, on="etiqueta", how="left")
    clases["log_disc"] = [
        math.log(abs(discriminante(ai))) for ai in clases["ainvs"]
    ]

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    # E1: Con L vs Sin L (condicionando solo por 100 cuantiles de N)
    filas_e1: list[dict[str, str | int | float]] = []
    # E2: Subpoblación pura (Tamagawa=1, Torsion=1, Sha=1)
    filas_e2: list[dict[str, str | int | float]] = []
    # E3: Paridad de ap (p+1-ap par vs impar) sobre log ∏c en curvas con T=1
    filas_e3: list[dict[str, str | int | float]] = []

    for rango in (0, 1, 2):
        for nombre_vent, n_min, n_max in VENTANAS:
            sub = clases.loc[
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ].reset_index(drop=True)

            bins_L = pd.qcut(
                sub["l_valor"], q=100, labels=False, duplicates="drop"
            ).to_numpy(dtype=int)
            bins_N = pd.qcut(
                sub["conductor"], q=100, labels=False, duplicates="drop"
            ).to_numpy(dtype=int)

            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
            log_L = np.log(sub["l_valor"].to_numpy(dtype=np.float64))
            log_reg = (
                np.zeros(len(sub), dtype=np.float64)
                if rango == 0
                else np.log(sub["regulador"].to_numpy(dtype=np.float64))
            )
            log_tam = np.log(sub["tamagawa"].to_numpy(dtype=np.float64))
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            tam_int = sub["tamagawa"].to_numpy(dtype=int)
            tor_int = sub["torsion"].to_numpy(dtype=int)
            sha_int = np.rint(sub["sha_an"].to_numpy(dtype=float)).astype(int)

            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)
            buenos_sub = (cond_sub[:, None] % primos[None, :]) != 0

            mask_pura = (tam_int == 1) & (tor_int == 1) & (sha_int == 1)
            mask_t1_c1 = (tam_int == 1) & (tor_int == 1)
            mask_t1 = tor_int == 1

            for j, p_primo in enumerate(primos):
                p_int = int(p_primo)
                if p_int not in (2, 3, 5, 7, 11, 13):
                    continue
                mask_b = buenos_sub[:, j]
                ap_p = ap_sub[mask_b, j].astype(np.float64)
                b_L = bins_L[mask_b]
                b_N = bins_N[mask_b]

                # E1
                rho_om_L, n_ef_L = correlacion_parcial_bins(
                    ap_p, log_omega[mask_b], b_L
                )
                rho_om_N, n_ef_N = correlacion_parcial_bins(
                    ap_p, log_omega[mask_b], b_N
                )
                rho_L_N, _ = correlacion_parcial_bins(ap_p, log_L[mask_b], b_N)
                rho_tam_N, _ = correlacion_parcial_bins(
                    ap_p, log_tam[mask_b], b_N
                )
                rho_tam_L, _ = correlacion_parcial_bins(
                    ap_p, log_tam[mask_b], b_L
                )
                rho_reg_N = (
                    0.0
                    if rango == 0
                    else correlacion_parcial_bins(ap_p, log_reg[mask_b], b_N)[0]
                )
                rho_reg_L = (
                    0.0
                    if rango == 0
                    else correlacion_parcial_bins(ap_p, log_reg[mask_b], b_L)[0]
                )

                filas_e1.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n": int(np.sum(mask_b)),
                        "rho_omega_dado_N": rho_om_N,
                        "rho_omega_dado_L": rho_om_L,
                        "rho_L_dado_N": rho_L_N,
                        "rho_tam_dado_N": rho_tam_N,
                        "rho_tam_dado_L": rho_tam_L,
                        "rho_reg_dado_N": rho_reg_N,
                        "rho_reg_dado_L": rho_reg_L,
                        "se": float(1.0 / np.sqrt(n_ef_L)),
                    }
                )

                # E2: Subpoblación pura (∏c=1, T=1, Sha=1)
                m_p = mask_b & mask_pura
                n_pura = int(np.sum(m_p))
                if n_pura > 200:
                    b_L_pura = pd.qcut(
                        sub.loc[m_p, "l_valor"],
                        q=min(100, n_pura // 20),
                        labels=False,
                        duplicates="drop",
                    ).to_numpy(dtype=int)
                    ap_pura = ap_sub[m_p, j].astype(np.float64)
                    rho_pura, nef_p = correlacion_parcial_bins(
                        ap_pura, log_omega[m_p], b_L_pura
                    )
                    rho_pura_disc, _ = correlacion_parcial_cov(
                        ap_pura, log_omega[m_p], b_L_pura, log_disc[m_p]
                    )
                    z_pura = float(rho_pura * np.sqrt(nef_p))

                    # También para (∏c=1, T=1) con Sha libre
                    m_tc = mask_b & mask_t1_c1
                    b_L_tc = pd.qcut(
                        sub.loc[m_tc, "l_valor"],
                        q=min(100, int(np.sum(m_tc)) // 20),
                        labels=False,
                        duplicates="drop",
                    ).to_numpy(dtype=int)
                    rho_tc, nef_tc = correlacion_parcial_bins(
                        ap_sub[m_tc, j].astype(np.float64),
                        log_omega[m_tc],
                        b_L_tc,
                    )
                    z_tc = float(rho_tc * np.sqrt(nef_tc))

                    filas_e2.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "p": p_int,
                            "n_pura": n_pura,
                            "rho_pura_dado_L": rho_pura,
                            "z_pura": z_pura,
                            "rho_pura_dado_L_logdisc": rho_pura_disc,
                            "n_t1_c1": int(np.sum(m_tc)),
                            "rho_t1_c1_dado_L": rho_tc,
                            "z_t1_c1": z_tc,
                        }
                    )

                # E3: Paridad de ap en curvas con T=1 (p impar)
                if p_int > 2:
                    m_t = mask_b & mask_t1
                    ap_t = ap_sub[m_t, j]
                    tam_t = log_tam[m_t]
                    es_par = (ap_t % 2) == 0
                    n_par = int(np.sum(es_par))
                    n_impar = int(np.sum(~es_par))
                    if n_par > 50 and n_impar > 50:
                        m_tam_par = float(np.mean(tam_t[es_par]))
                        m_tam_imp = float(np.mean(tam_t[~es_par]))
                        v_par = float(np.var(tam_t[es_par], ddof=1))
                        v_imp = float(np.var(tam_t[~es_par], ddof=1))
                        se_diff = math.sqrt(v_par / n_par + v_imp / n_impar)
                        delta_tam = m_tam_par - m_tam_imp
                        z_tam = delta_tam / se_diff if se_diff > 0 else 0.0

                        filas_e3.append(
                            {
                                "rango": rango,
                                "ventana": nombre_vent,
                                "p": p_int,
                                "n_T1": int(np.sum(m_t)),
                                "n_ap_par": n_par,
                                "n_ap_impar": n_impar,
                                "media_ap_par": float(np.mean(ap_t[es_par])),
                                "media_ap_impar": float(np.mean(ap_t[~es_par])),
                                "media_logtam_ap_par": m_tam_par,
                                "media_logtam_ap_impar": m_tam_imp,
                                "delta_logtam_par_menos_impar": delta_tam,
                                "z_welch": z_tam,
                            }
                        )

    df_e1 = pd.DataFrame(filas_e1)
    df_e1.to_csv(DIR_EXP / "e1_colisionador_vs_intrinsico.csv", index=False)

    df_e2 = pd.DataFrame(filas_e2)
    df_e2.to_csv(DIR_EXP / "e2_subpoblacion_pura.csv", index=False)

    df_e3 = pd.DataFrame(filas_e3)
    df_e3.to_csv(DIR_EXP / "e3_paridad_ap_tamagawa.csv", index=False)

    # Guardar copia del código exploratorio en resultados/08-rango1/exploratorio/
    ruta_este = Path(__file__).resolve()
    (DIR_EXP / "e08_exploratorio.py").write_text(
        ruta_este.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Figura exploratoria
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)

    # Panel A: E1 en Rango 0 y Rango 1 (N < 50 000): ρ(ap, log Ω | N) vs ρ(ap, log Ω | L)
    for r, col, mk in [(0, "#1f5a99", "o"), (1, "#c0392b", "s")]:
        s1 = df_e1.loc[(df_e1["rango"] == r) & (df_e1["ventana"] == "1-49999")]
        ax1.plot(
            s1["p"],
            s1["rho_omega_dado_N"],
            f"{mk}--",
            color=col,
            alpha=0.65,
            label=f"Rango {r}: ρ(aₚ, log Ω | N) [sin fijar L]",
        )
        ax1.plot(
            s1["p"],
            s1["rho_omega_dado_L"],
            f"{mk}-",
            color=col,
            linewidth=1.8,
            label=f"Rango {r}: ρ(aₚ, log Ω | L)",
        )
    ax1.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax1.set_xlabel("Primo p")
    ax1.set_ylabel("Correlación parcial ρ")
    ax1.set_title("E1. Efecto de fijar L vs fijar solo N (N < 50 000)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=8.5)

    # Panel B: E2 Subpoblación pura (∏c=1, T=1, Sha=1) en rangos 0, 1 y 2
    for r, col, mk in [
        (0, "#1f5a99", "o"),
        (1, "#c0392b", "s"),
        (2, "#27ae60", "^"),
    ]:
        s2 = df_e2.loc[(df_e2["rango"] == r) & (df_e2["ventana"] == "1-49999")]
        ax2.plot(
            s2["p"],
            s2["rho_pura_dado_L"],
            f"{mk}-",
            color=col,
            linewidth=1.8,
            label=f"Rango {r} puro (∏c=1, T=1, Sha=1)",
        )
    ax2.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
    ax2.set_xlabel("Primo p")
    ax2.set_ylabel("ρ(aₚ, log Ω | L) en subpoblación pura")
    ax2.set_title("E2. Subpoblación pura ∏c=1, T=1, Sha=1 (N < 50 000)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        "Análisis exploratorios E1 y E2 (EXPLORATORIO)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_EXP / "exploratorio.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"Tiempo total e08_exploratorio: {t_fin - t_inicio:.2f} s")
    print("\n--- E1: CON L VS SIN L (p in {3, 5, 7}) ---")
    print(df_e1.loc[df_e1["p"].isin([3, 5, 7])].to_string(index=False))
    print("\n--- E2: SUBPOBLACIÓN PURA (p in {3, 5, 7}) ---")
    print(df_e2.loc[df_e2["p"].isin([3, 5, 7])].to_string(index=False))
    print("\n--- E3: PARIDAD DE ap SOBRE log ∏c EN CURVAS CON T=1 (p in {3, 5, 7}) ---")
    print(
        df_e3.loc[
            (df_e3["rango"].isin([0, 1])) & (df_e3["p"].isin([3, 5, 7]))
        ].to_string(index=False)
    )
    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_exploratorio()
