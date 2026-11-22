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
    correlacion_parcial_bins,
    correlacion_parcial_cov,
    covarianza_en_bins,
)
from lab.gp import factor_euler_malos

SEMILLA = 20261121
N_PRIMOS = 25

RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "07-mecanismo-h8"
RUTA_EULER_MALOS = DIR_DERIVADOS / "euler_malos.csv"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def _construir_o_cargar_euler_malos(clases: pd.DataFrame) -> pd.DataFrame:
    DIR_DERIVADOS.mkdir(parents=True, exist_ok=True)
    total = len(clases)
    if RUTA_EULER_MALOS.exists():
        df_ex = pd.read_csv(RUTA_EULER_MALOS)
        if len(df_ex) == total and list(df_ex.columns) == ["etiqueta", "log_B", "n_malos"]:
            return df_ex

    curvas = [(fila["etiqueta"], fila["ainvs"]) for _, fila in clases.iterrows()]
    partes: list[pd.DataFrame] = []
    for inicio in range(0, total, TAMANO_LOTE_MAX):
        fin = min(inicio + TAMANO_LOTE_MAX, total)
        df_lote = factor_euler_malos(curvas[inicio:fin])
        partes.append(df_lote)

    df_total = pd.concat(partes, ignore_index=True)
    df_total.to_csv(RUTA_EULER_MALOS, index=False)
    return df_total


def ejecutar_mecanismo_h8() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)

    t_eb0 = time.perf_counter()
    df_eb = _construir_o_cargar_euler_malos(clases)
    t_eb1 = time.perf_counter()

    clases = clases.merge(df_eb, on="etiqueta", how="left")
    clases["log_disc"] = [
        math.log(abs(discriminante(ai))) for ai in clases["ainvs"]
    ]

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    filas_desc: list[dict[str, str | int | float]] = []
    filas_col: list[dict[str, str | int | float]] = []

    rng_placebo = np.random.default_rng(SEMILLA)

    for rango in (0, 1):
        for nombre_vent, n_min, n_max in VENTANAS:
            sub = clases.loc[
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ].reset_index(drop=True)

            bins_L = pd.qcut(
                sub["l_valor"], q=100, labels=False, duplicates="drop"
            ).to_numpy(dtype=int)

            omega = sub["omega"].to_numpy(dtype=np.float64)
            log_omega = np.log(omega)
            log_L = np.log(sub["l_valor"].to_numpy(dtype=np.float64))
            log_reg = (
                np.zeros(len(sub), dtype=np.float64)
                if rango == 0
                else np.log(sub["regulador"].to_numpy(dtype=np.float64))
            )
            log_tam = np.log(sub["tamagawa"].to_numpy(dtype=np.float64))
            log_sha = np.log(sub["sha_an"].to_numpy(dtype=np.float64))
            log_tor = np.log(sub["torsion"].to_numpy(dtype=np.float64))
            log_B = sub["log_B"].to_numpy(dtype=np.float64)
            log_disc = sub["log_disc"].to_numpy(dtype=np.float64)
            z_placebo = rng_placebo.normal(0.0, 1.0, size=len(sub))

            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)
            buenos_sub = (cond_sub[:, None] % primos[None, :]) != 0

            for j, p_primo in enumerate(primos):
                p_int = int(p_primo)
                mask_b = buenos_sub[:, j]
                n_b = int(np.sum(mask_b))
                ap_p = ap_sub[mask_b, j].astype(np.float64)
                b_L = bins_L[mask_b]

                # Parte A: descomposición exacta de Cov(ap, log Ω | L)
                cov_total = covarianza_en_bins(ap_p, log_omega[mask_b], b_L)
                cov_L = covarianza_en_bins(ap_p, log_L[mask_b], b_L)
                cov_reg = (
                    0.0
                    if rango == 0
                    else -covarianza_en_bins(ap_p, log_reg[mask_b], b_L)
                )
                cov_tam = -covarianza_en_bins(ap_p, log_tam[mask_b], b_L)
                cov_sha = -covarianza_en_bins(ap_p, log_sha[mask_b], b_L)
                cov_tor = 2.0 * covarianza_en_bins(ap_p, log_tor[mask_b], b_L)

                suma = cov_L + cov_reg + cov_tam + cov_sha + cov_tor
                error_id = abs(cov_total - suma)
                if error_id > 1e-6:
                    raise RuntimeError(
                        f"Fallo de identidad en rango={rango}, ventana={nombre_vent}, p={p_int}: "
                        f"error_identidad={error_id:.3e} > 1e-6"
                    )

                filas_desc.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n": n_b,
                        "cov_total": cov_total,
                        "cov_L": cov_L,
                        "cov_reg": cov_reg,
                        "cov_tam": cov_tam,
                        "cov_sha": cov_sha,
                        "cov_tor": cov_tor,
                        "suma": suma,
                        "error_identidad": error_id,
                        "pct_L": 100.0 * cov_L / cov_total,
                        "pct_reg": 100.0 * cov_reg / cov_total,
                        "pct_tam": 100.0 * cov_tam / cov_total,
                        "pct_sha": 100.0 * cov_sha / cov_total,
                        "pct_tor": 100.0 * cov_tor / cov_total,
                    }
                )

                # Parte B: prueba del colisionador
                rho_omega, n_ef = correlacion_parcial_bins(
                    ap_p, omega[mask_b], b_L
                )
                rho_logomega, _ = correlacion_parcial_bins(
                    ap_p, log_omega[mask_b], b_L
                )
                rho_logB, _ = correlacion_parcial_cov(
                    ap_p, log_omega[mask_b], b_L, log_B[mask_b]
                )
                rho_logtam, _ = correlacion_parcial_cov(
                    ap_p, log_omega[mask_b], b_L, log_tam[mask_b]
                )
                rho_logdisc, _ = correlacion_parcial_cov(
                    ap_p, log_omega[mask_b], b_L, log_disc[mask_b]
                )
                rho_placebo, _ = correlacion_parcial_cov(
                    ap_p, log_omega[mask_b], b_L, z_placebo[mask_b]
                )
                rho_ap_logB, _ = correlacion_parcial_bins(
                    ap_p, log_B[mask_b], b_L
                )
                se_val = float(1.0 / np.sqrt(n_ef)) if n_ef > 0 else float("nan")

                filas_col.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "n": n_b,
                        "rho_omega": rho_omega,
                        "rho_logomega": rho_logomega,
                        "rho_logB": rho_logB,
                        "rho_logtam": rho_logtam,
                        "rho_logdisc": rho_logdisc,
                        "rho_placebo": rho_placebo,
                        "rho_ap_logB": rho_ap_logB,
                        "se": se_val,
                    }
                )

    df_desc = pd.DataFrame(filas_desc)
    df_desc.to_csv(DIR_SALIDA / "descomposicion.csv", index=False)

    df_col = pd.DataFrame(filas_col)
    df_col.to_csv(DIR_SALIDA / "colisionador.csv", index=False)

    # Figura mecanismo.png: dos paneles (rango 0 y rango 1 en N < 50 000)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=150, sharey=True)

    series_cfg = [
        ("rho_logomega", "o-", "#1f5a99", "Sin covariable: ρ(aₚ, log Ω | L)"),
        ("rho_logB", "s--", "#e67e22", "Controlando Z = log B"),
        ("rho_logtam", "^-", "#c0392b", "Controlando Z = log ∏c"),
        ("rho_logdisc", "d-", "#27ae60", "Controlando Z = log |Δ|"),
        ("rho_placebo", "x:", "#7f8c8d", "Controlando Z = ruido placebo"),
    ]

    for ax, rango, titulo in [
        (ax0, 0, "A. Rango 0 (N < 50 000): ρ(aₚ, log Ω | L, Z)"),
        (ax1, 1, "B. Rango 1 (N < 50 000): ρ(aₚ, log Ω | L′(1), Z)"),
    ]:
        sub_c = df_col.loc[
            (df_col["rango"] == rango) & (df_col["ventana"] == "1-49999")
        ].reset_index(drop=True)
        for col_y, fmt, color, label in series_cfg:
            ax.plot(
                sub_c["p"],
                sub_c[col_y],
                fmt,
                color=color,
                markersize=4.5,
                linewidth=1.5,
                label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Primo p (p ≤ 97)")
        ax.set_ylabel("Correlación parcial ρ")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Efecto de covariables candidatas sobre ρ(aₚ, log Ω | L) en N < 50 000 (una curva por clase)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "mecanismo.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"Tiempo caché euler_malos.csv: {t_eb1 - t_eb0:.2f} s")
    print(f"Tiempo total e07_mecanismo: {t_fin - t_inicio:.2f} s")
    print(f"Máximo error_identidad en los 100 casos: {df_desc['error_identidad'].max():.3e}")

    print("\n--- DESCOMPOSICIÓN (p in {2, 3, 5, 7}) ---")
    print(
        df_desc.loc[df_desc["p"].isin([2, 3, 5, 7])][
            [
                "rango",
                "ventana",
                "p",
                "cov_total",
                "pct_L",
                "pct_reg",
                "pct_tam",
                "pct_sha",
                "pct_tor",
                "error_identidad",
            ]
        ].to_string(index=False)
    )

    print("\n--- COLISIONADOR (p in {2, 3, 5, 7}) ---")
    print(
        df_col.loc[df_col["p"].isin([2, 3, 5, 7])][
            [
                "rango",
                "ventana",
                "p",
                "rho_omega",
                "rho_logomega",
                "rho_logB",
                "rho_logtam",
                "rho_logdisc",
                "rho_placebo",
                "rho_ap_logB",
                "se",
            ]
        ].to_string(index=False)
    )

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_mecanismo_h8()
