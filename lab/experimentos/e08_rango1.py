from __future__ import annotations

import math
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import DIR_DERIVADOS, cargar_cache_ap, discriminante
from lab.estadistica import (
    correlacion_parcial_bins,
    correlacion_parcial_cov,
    covarianza_en_bins,
)

N_PRIMOS = 25
RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "08-rango1"
RUTA_EULER_MALOS = DIR_DERIVADOS / "euler_malos.csv"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def ejecutar_rango1() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd).reset_index(drop=True)

    if not RUTA_EULER_MALOS.exists():
        raise FileNotFoundError(f"No existe la caché {RUTA_EULER_MALOS}")
    df_eb = pd.read_csv(RUTA_EULER_MALOS)
    clases = clases.merge(df_eb, on="etiqueta", how="left")

    clases["log_disc"] = [
        math.log(abs(discriminante(ai))) for ai in clases["ainvs"]
    ]
    clases["log_N"] = np.log(clases["conductor"].to_numpy(dtype=np.float64))
    clases["sigma"] = clases["log_disc"] / clases["log_N"]

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    filas_cov: list[dict[str, str | int | float]] = []
    filas_dir: list[dict[str, str | int | float]] = []
    filas_desc_r2: list[dict[str, str | int | float]] = []

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

            log_omega = np.log(sub["omega"].to_numpy(dtype=np.float64))
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
            sigma = sub["sigma"].to_numpy(dtype=np.float64)
            log_omega_sobre_R = log_omega - log_reg

            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)
            buenos_sub = (cond_sub[:, None] % primos[None, :]) != 0

            conjuntos_a: list[tuple[str, np.ndarray | None]] = [
                ("ninguno", None),
                ("log_B", log_B),
                ("log_tam", log_tam),
                ("log_disc", log_disc),
                ("sigma", sigma),
                ("log_B+log_tam", np.column_stack([log_B, log_tam])),
                ("log_B+log_disc", np.column_stack([log_B, log_disc])),
                ("log_tam+log_disc", np.column_stack([log_tam, log_disc])),
                (
                    "log_B+log_tam+log_disc",
                    np.column_stack([log_B, log_tam, log_disc]),
                ),
            ]

            Z0_full = np.column_stack([log_tam, log_tor, log_sha])
            conjuntos_b: list[tuple[str, np.ndarray]] = [
                ("Z0", Z0_full),
                ("Z0+log_disc", np.column_stack([Z0_full, log_disc])),
                ("Z0+log_B", np.column_stack([Z0_full, log_B])),
                (
                    "Z0+log_disc+log_B",
                    np.column_stack([Z0_full, log_disc, log_B]),
                ),
            ]

            for j, p_primo in enumerate(primos):
                p_int = int(p_primo)
                mask_b = buenos_sub[:, j]
                n_b = int(np.sum(mask_b))
                ap_p = ap_sub[mask_b, j].astype(np.float64)
                b_L = bins_L[mask_b]
                log_om_b = log_omega[mask_b]

                # Parte A: Covariables
                for nombre_conj, z_mat in conjuntos_a:
                    if z_mat is None:
                        rho_val, n_ef = correlacion_parcial_bins(
                            ap_p, log_om_b, b_L
                        )
                    else:
                        z_sub = z_mat[mask_b]
                        rho_val, n_ef = correlacion_parcial_cov(
                            ap_p, log_om_b, b_L, z_sub
                        )
                    se_val = (
                        float(1.0 / np.sqrt(n_ef)) if n_ef > 0 else float("nan")
                    )
                    filas_cov.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "p": p_int,
                            "n": n_b,
                            "conjunto": nombre_conj,
                            "rho": rho_val,
                            "se": se_val,
                        }
                    )

                # Parte B: Dirección libre (rangos 1 y 2)
                if rango in (1, 2):
                    log_om_r_b = log_omega_sobre_R[mask_b]
                    for nombre_conj_b, z_mat_b in conjuntos_b:
                        rho_b, n_ef_b = correlacion_parcial_cov(
                            ap_p, log_om_r_b, b_L, z_mat_b[mask_b]
                        )
                        se_b = (
                            float(1.0 / np.sqrt(n_ef_b))
                            if n_ef_b > 0
                            else float("nan")
                        )
                        filas_dir.append(
                            {
                                "rango": rango,
                                "ventana": nombre_vent,
                                "p": p_int,
                                "n": n_b,
                                "conjunto": nombre_conj_b,
                                "rho": rho_b,
                                "se": se_b,
                            }
                        )

                    # Comprobación del mecanismo de tamaño: ρ(log|Δ|, log(Ω/R) | L, Z0)
                    rho_tam_mec, n_ef_mec = correlacion_parcial_cov(
                        log_disc[mask_b], log_om_r_b, b_L, Z0_full[mask_b]
                    )
                    se_mec = (
                        float(1.0 / np.sqrt(n_ef_mec))
                        if n_ef_mec > 0
                        else float("nan")
                    )
                    filas_dir.append(
                        {
                            "rango": rango,
                            "ventana": nombre_vent,
                            "p": p_int,
                            "n": n_b,
                            "conjunto": "log_disc_vs_log_omega_R|Z0",
                            "rho": rho_tam_mec,
                            "se": se_mec,
                        }
                    )

                # Parte C: Descomposición exacta en rango 2
                if rango == 2:
                    cov_total = covarianza_en_bins(ap_p, log_om_b, b_L)
                    cov_L = covarianza_en_bins(ap_p, log_L[mask_b], b_L)
                    cov_reg = -covarianza_en_bins(ap_p, log_reg[mask_b], b_L)
                    cov_tam = -covarianza_en_bins(ap_p, log_tam[mask_b], b_L)
                    cov_sha = -covarianza_en_bins(ap_p, log_sha[mask_b], b_L)
                    cov_tor = 2.0 * covarianza_en_bins(ap_p, log_tor[mask_b], b_L)

                    suma = cov_L + cov_reg + cov_tam + cov_sha + cov_tor
                    error_id = abs(cov_total - suma)
                    if error_id > 1e-6:
                        raise RuntimeError(
                            f"Fallo de identidad en rango 2, ventana={nombre_vent}, p={p_int}: {error_id}"
                        )
                    filas_desc_r2.append(
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

    df_cov = pd.DataFrame(filas_cov)
    df_cov.to_csv(DIR_SALIDA / "covariables.csv", index=False)

    df_dir = pd.DataFrame(filas_dir)
    df_dir.to_csv(DIR_SALIDA / "direccion_libre.csv", index=False)

    df_desc_r2 = pd.DataFrame(filas_desc_r2)
    df_desc_r2.to_csv(DIR_SALIDA / "descomposicion_r2.csv", index=False)

    # Figura rango1.png: para rango 1 y N < 50 000, ρ(aₚ, log Ω | L') contra p con cada conjunto de covariables
    sub_fig = df_cov.loc[
        (df_cov["rango"] == 1) & (df_cov["ventana"] == "1-49999")
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)

    estilos = [
        ("ninguno", "o-", "#1f5a99", 2.0, "Sin covariables"),
        ("log_B", "s--", "#e67e22", 1.5, "{log B}"),
        ("log_tam", "^-", "#c0392b", 1.6, "{log ∏c}"),
        ("log_disc", "d-", "#27ae60", 1.6, "{log |Δ|}"),
        ("sigma", "v:", "#16a085", 1.4, "{σ = log|Δ|/log N}"),
        ("log_B+log_tam", "P--", "#8e44ad", 1.4, "{log B, log ∏c}"),
        ("log_B+log_disc", "X--", "#d35400", 1.4, "{log B, log |Δ|}"),
        ("log_tam+log_disc", "*:", "#2c3e50", 1.5, "{log ∏c, log |Δ|}"),
        (
            "log_B+log_tam+log_disc",
            "h-",
            "#111111",
            2.2,
            "{log B, log ∏c, log |Δ|}",
        ),
    ]

    for conj, fmt, col, lw, etiqueta in estilos:
        sc = sub_fig.loc[sub_fig["conjunto"] == conj].sort_values("p")
        ax.plot(
            sc["p"],
            sc["rho"],
            fmt,
            color=col,
            linewidth=lw,
            markersize=5,
            label=etiqueta,
        )

    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Primo p (25 primeros primos, p ≤ 97)")
    ax.set_ylabel("Correlación parcial ρ(aₚ, log Ω | L′(1), Z)")
    ax.set_title(
        "Rango 1, N < 50 000: ρ(aₚ, log Ω | L′(1), Z) para los 9 conjuntos de covariables"
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", ncol=2, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "rango1.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"Tiempo total e08_rango1: {t_fin - t_inicio:.2f} s")

    print("\n--- PARTE A: COVARIABLES EN RANGO 1 (p in {3, 5, 7}) ---")
    piv_a = df_cov.loc[
        (df_cov["rango"] == 1) & (df_cov["p"].isin([3, 5, 7]))
    ].pivot(index=["ventana", "p"], columns="conjunto", values="rho")
    print(piv_a.to_string())

    print("\n--- PARTE B: DIRECCIÓN LIBRE EN RANGO 1 Y 2 (p in {3, 5, 7}) ---")
    piv_b = df_dir.loc[df_dir["p"].isin([3, 5, 7])].pivot(
        index=["rango", "ventana", "p"], columns="conjunto", values="rho"
    )
    print(piv_b.to_string())

    print("\n--- PARTE C: DESCOMPOSICIÓN RANGO 2 (p in {2, 3, 5, 7}) ---")
    print(
        df_desc_r2.loc[df_desc_r2["p"].isin([2, 3, 5, 7])][
            [
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

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_rango1()
