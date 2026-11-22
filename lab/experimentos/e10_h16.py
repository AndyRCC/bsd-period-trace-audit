from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import DIR_DERIVADOS, TAMANO_LOTE_MAX, cargar_cache_ap, discriminante
from lab.estadistica import (
    bins_estratificados,
    clave_estrato,
    correlacion_parcial_bins,
    correlacion_parcial_cov,
)
from lab.gp import tipos_reduccion

N_PRIMOS = 25
RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "10-cerrar-h16"
RUTA_EULER_MALOS = DIR_DERIVADOS / "euler_malos.csv"
RUTA_TIPOS_RED = DIR_DERIVADOS / "tipos_reduccion.csv"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]
PRIMOS_OBJ = (3, 5, 7)


def construir_o_cargar_tipos_reduccion(clases: pd.DataFrame) -> pd.DataFrame:
    DIR_DERIVADOS.mkdir(parents=True, exist_ok=True)
    total = len(clases)
    cols_esp = ["etiqueta", "n_partida", "n_no_partida", "n_aditiva"]
    if RUTA_TIPOS_RED.exists():
        df_ex = pd.read_csv(RUTA_TIPOS_RED)
        if len(df_ex) == total and list(df_ex.columns) == cols_esp:
            return df_ex

    curvas = [(fila["etiqueta"], fila["ainvs"]) for _, fila in clases.iterrows()]
    partes: list[pd.DataFrame] = []
    for inicio in range(0, total, TAMANO_LOTE_MAX):
        fin = min(inicio + TAMANO_LOTE_MAX, total)
        t0 = time.perf_counter()
        df_lote = tipos_reduccion(curvas[inicio:fin])
        dt = time.perf_counter() - t0
        print(
            f"  tipos_reduccion lote {inicio}-{fin} ({fin - inicio} curvas): {dt:.2f} s",
            flush=True,
        )
        partes.append(df_lote)

    df_total = pd.concat(partes, ignore_index=True)
    temporal = RUTA_TIPOS_RED.with_suffix(".tmp.csv")
    df_total.to_csv(temporal, index=False)
    temporal.replace(RUTA_TIPOS_RED)
    return df_total


def ejecutar_h16() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)

    if not RUTA_EULER_MALOS.exists():
        raise FileNotFoundError(f"Falta caché requerida: {RUTA_EULER_MALOS}")
    df_eb = pd.read_csv(RUTA_EULER_MALOS)
    if len(df_eb) != len(clases):
        raise RuntimeError(
            f"euler_malos.csv tiene {len(df_eb)} filas, esperadas {len(clases)}"
        )

    t_tr0 = time.perf_counter()
    df_tr = construir_o_cargar_tipos_reduccion(clases)
    t_tr1 = time.perf_counter()
    print(f"Caché tipos_reduccion lista en {t_tr1 - t_tr0:.2f} s", flush=True)

    clases = clases.merge(df_eb, on="etiqueta", how="left")
    clases = clases.merge(df_tr, on="etiqueta", how="left")

    # Control de cordura exacto: n_partida + n_no_partida + n_aditiva == n_malos en todas las clases
    suma_tipos = (
        clases["n_partida"].to_numpy(dtype=int)
        + clases["n_no_partida"].to_numpy(dtype=int)
        + clases["n_aditiva"].to_numpy(dtype=int)
    )
    n_malos = clases["n_malos"].to_numpy(dtype=int)
    desacuerdos = int(np.sum(suma_tipos != n_malos))
    if desacuerdos > 0:
        raise RuntimeError(
            f"Inconsistencia entre tipos_reduccion y euler_malos en {desacuerdos} clases"
        )
    print(
        f"Verificación tipos_reduccion vs euler_malos: 0 desacuerdos en {len(clases)} clases.",
        flush=True,
    )

    clases["log_disc"] = [
        math.log(abs(discriminante(ai))) for ai in clases["ainvs"]
    ]
    clases["log_omega"] = np.log(clases["omega"].to_numpy(dtype=np.float64))

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)
    idx_primos = {int(p): j for j, p in enumerate(primos)}

    filas_p12: list[dict[str, str | int | float]] = []
    filas_cadena: list[dict[str, str | int | float]] = []

    for rango in (1, 2):
        for nombre_vent, n_min, n_max in VENTANAS:
            mask_rv = (
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ).to_numpy(dtype=bool)
            sub = clases.loc[mask_rv].reset_index(drop=True)

            l_val = sub["l_valor"].to_numpy(dtype=np.float64)
            log_om = sub["log_omega"].to_numpy(dtype=np.float64)
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            cond = sub["conductor"].to_numpy(dtype=np.int32)
            cp = sub["tamagawa"].to_numpy(dtype=np.int64)
            T = sub["torsion"].to_numpy(dtype=np.int64)
            sha = sub["sha_an"].to_numpy(dtype=np.float64)
            n_part = sub["n_partida"].to_numpy(dtype=np.int64)
            n_nopart = sub["n_no_partida"].to_numpy(dtype=np.int64)

            # Quintiles de B calculados dentro de cada (rango, ventana)
            q_B = np.asarray(
                pd.qcut(log_B, q=5, labels=False, duplicates="drop"),
                dtype=np.int64,
            )
            n_qB_bins = int(len(np.unique(q_B)))
            print(
                f"Rango {rango}, ventana {nombre_vent}: n={len(sub)}, quintiles B={n_qB_bins}, conteos={np.bincount(q_B).tolist()}",
                flush=True,
            )

            # ---- P14: dentro de los estratos, dado L (4 casos de rango x ventana) ----
            est_rv = clave_estrato(cp, T, sha)
            bins_rv = bins_estratificados(l_val, est_rv, q_max=100, n_min=200)
            m_rv = bins_rv >= 0

            rho_B_disc, nef_B_disc = correlacion_parcial_bins(
                log_B[m_rv], log_disc[m_rv], bins_rv[m_rv]
            )
            se_B_disc = float(1.0 / np.sqrt(nef_B_disc))
            filas_cadena.append(
                {
                    "prueba": "P14",
                    "rango": rango,
                    "ventana": nombre_vent,
                    "p": "todos",
                    "par": "logB_vs_logdisc",
                    "n": int(np.sum(m_rv)),
                    "rho": rho_B_disc,
                    "se": se_B_disc,
                    "z": rho_B_disc / se_B_disc,
                }
            )

            n_mult = n_part + n_nopart
            m_mult = n_mult > 0
            frac_part = n_part[m_mult].astype(np.float64) / n_mult[m_mult].astype(
                np.float64
            )
            bins_mult = bins_estratificados(
                l_val[m_mult], est_rv[m_mult], q_max=100, n_min=200
            )
            m_mult_v = bins_mult >= 0
            rho_fp_disc, nef_fp_disc = correlacion_parcial_bins(
                frac_part[m_mult_v],
                log_disc[m_mult][m_mult_v],
                bins_mult[m_mult_v],
            )
            se_fp_disc = float(1.0 / np.sqrt(nef_fp_disc))
            filas_cadena.append(
                {
                    "prueba": "P14",
                    "rango": rango,
                    "ventana": nombre_vent,
                    "p": "todos",
                    "par": "frac_partida_vs_logdisc",
                    "n": int(np.sum(m_mult_v)),
                    "rho": rho_fp_disc,
                    "se": se_fp_disc,
                    "z": rho_fp_disc / se_fp_disc,
                }
            )

            # ---- P12 y P13: por primo p in {3, 5, 7}, solo p ∤ N ----
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)

            for p_int in PRIMOS_OBJ:
                j = idx_primos[p_int]
                mask_b = (cond % p_int) != 0
                ap_b = ap_sub[mask_b, j].astype(np.float64)
                l_b = l_val[mask_b]
                log_om_b = log_om[mask_b]
                log_B_b = log_B[mask_b]
                log_disc_b = log_disc[mask_b]
                cp_b = cp[mask_b]
                T_b = T[mask_b]
                sha_b = sha[mask_b]
                q_B_b = q_B[mask_b]

                # Estratos exactos (Tarea 9)
                est_base = clave_estrato(cp_b, T_b, sha_b)
                bins_base = bins_estratificados(
                    l_b, est_base, q_max=100, n_min=200
                )
                m_base = bins_base >= 0
                rho_est, nef_est = correlacion_parcial_bins(
                    ap_b[m_base], log_om_b[m_base], bins_base[m_base]
                )
                se_est = float(1.0 / np.sqrt(nef_est))

                # Estratos exactos x quintil de B
                est_qB = clave_estrato(cp_b, T_b, sha_b, extra=q_B_b)
                bins_qB = bins_estratificados(
                    l_b, est_qB, q_max=100, n_min=200
                )
                m_qB = bins_qB >= 0
                rho_qB, nef_qB = correlacion_parcial_bins(
                    ap_b[m_qB], log_om_b[m_qB], bins_qB[m_qB]
                )
                se_qB = float(1.0 / np.sqrt(nef_qB))

                # Estratos exactos + log B lineal
                rho_logB, nef_logB = correlacion_parcial_cov(
                    ap_b[m_base],
                    log_om_b[m_base],
                    bins_base[m_base],
                    log_B_b[m_base],
                )
                se_logB = float(1.0 / np.sqrt(nef_logB))

                # Estratos exactos + log|Δ| lineal
                rho_logdisc, nef_logdisc = correlacion_parcial_cov(
                    ap_b[m_base],
                    log_om_b[m_base],
                    bins_base[m_base],
                    log_disc_b[m_base],
                )
                se_logdisc = float(1.0 / np.sqrt(nef_logdisc))

                pct_red_qB = 100.0 * (1.0 - abs(rho_qB) / abs(rho_est))
                pct_red_logB = 100.0 * (1.0 - abs(rho_logB) / abs(rho_est))
                pct_red_logdisc = 100.0 * (
                    1.0 - abs(rho_logdisc) / abs(rho_est)
                )

                filas_p12.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n_estrato": int(np.sum(m_base)),
                        "rho_estrato": rho_est,
                        "se_estrato": se_est,
                        "n_qB": int(np.sum(m_qB)),
                        "rho_qB": rho_qB,
                        "se_qB": se_qB,
                        "pct_red_qB": pct_red_qB,
                        "rho_logB": rho_logB,
                        "se_logB": se_logB,
                        "pct_red_logB": pct_red_logB,
                        "rho_logdisc": rho_logdisc,
                        "se_logdisc": se_logdisc,
                        "pct_red_logdisc": pct_red_logdisc,
                    }
                )

                # P13: dentro de estratos y dado L: ρ(ap, log B) y ρ(log B, log Ω)
                rho_ap_B, nef_ap_B = correlacion_parcial_bins(
                    ap_b[m_base], log_B_b[m_base], bins_base[m_base]
                )
                se_ap_B = float(1.0 / np.sqrt(nef_ap_B))
                filas_cadena.append(
                    {
                        "prueba": "P13",
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": str(p_int),
                        "par": "ap_vs_logB",
                        "n": int(np.sum(m_base)),
                        "rho": rho_ap_B,
                        "se": se_ap_B,
                        "z": rho_ap_B / se_ap_B,
                    }
                )

                rho_B_om, nef_B_om = correlacion_parcial_bins(
                    log_B_b[m_base], log_om_b[m_base], bins_base[m_base]
                )
                se_B_om = float(1.0 / np.sqrt(nef_B_om))
                filas_cadena.append(
                    {
                        "prueba": "P13",
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": str(p_int),
                        "par": "logB_vs_logomega",
                        "n": int(np.sum(m_base)),
                        "rho": rho_B_om,
                        "se": se_B_om,
                        "z": rho_B_om / se_B_om,
                    }
                )

    df_p12 = pd.DataFrame(filas_p12)
    df_p12.to_csv(DIR_SALIDA / "p12_estratos_B.csv", index=False)

    df_cadena = pd.DataFrame(filas_cadena)
    df_cadena.to_csv(DIR_SALIDA / "p13_p14_cadena.csv", index=False)

    # Figura h16.png: rangos 1 y 2, p = 3, 5, 7, las 4 series con barras ±2 EE
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), dpi=150, sharey=True)
    series_plot = [
        ("rho_estrato", "se_estrato", "o-", "#c0392b", -0.21, "Estratos exactos (base H16)"),
        ("rho_qB", "se_qB", "s-", "#2980b9", -0.07, "Estratos × quintiles de B"),
        ("rho_logB", "se_logB", "^--", "#16a085", 0.07, "Estratos + log B lineal"),
        ("rho_logdisc", "se_logdisc", "D-", "#8e44ad", 0.21, "Estratos + log|Δ| lineal"),
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
        (axes[0], 1, "A. Rango 1: ρ(aₚ, log Ω | L′(1), estrato) ± 2 EE"),
        (axes[1], 2, "B. Rango 2: ρ(aₚ, log Ω | L″(1)/2, estrato) ± 2 EE"),
    ]:
        sub_r = df_p12.loc[df_p12["rango"] == rango]
        for col_rho, col_se, fmt, color, dx, label in series_plot:
            ys = []
            errs = []
            for vent, p_val in pares_xp:
                fila = sub_r.loc[
                    (sub_r["ventana"] == vent) & (sub_r["p"] == p_val)
                ].iloc[0]
                ys.append(float(fila[col_rho]))
                errs.append(2.0 * float(fila[col_se]))
            ax.errorbar(
                x_base + dx,
                ys,
                yerr=errs,
                fmt=fmt,
                color=color,
                ecolor=color,
                elinewidth=1.25,
                capsize=3.2,
                markersize=5.6,
                linewidth=1.5,
                label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.axvline(2.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(etiquetas_x)
        ax.set_ylabel("Correlación parcial ρ(aₚ, log Ω | L, estrato)")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.6)

    fig.suptitle(
        "Cierre de H16 (P12): control por quintiles de B, log B lineal y log|Δ| lineal dentro de estratos exactos",
        fontsize=11.8,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "h16.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"\nTiempo total e10_h16: {t_fin - t_inicio:.2f} s")

    print("\n--- P12: ESTRATOS EXACTOS VS B Y log|Δ| ---")
    print(df_p12.to_string(index=False))

    print("\n--- P13 Y P14: ESLABONES DE LA CADENA H17 ---")
    print(df_cadena.to_string(index=False))

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_h16()
