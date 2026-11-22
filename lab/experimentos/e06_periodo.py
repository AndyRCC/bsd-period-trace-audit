from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import cargar_cache_ap, discriminante, simbolo_legendre
from lab.estadistica import (
    correlacion_parcial_bins,
    diferencia_perfiles,
    distribucion_nula_perfiles,
)

SEMILLA = 20261121
N_PERM = 1000
N_PRIMOS = 25

RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "06-periodo-componentes"

VENTANAS = [
    ("1-49999", 1, 49_999),
    ("50000-99999", 50_000, 99_999),
]


def _etiqueta_tamano(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return ">=3"


def ejecutar_periodo_componentes() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)

    # Estadísticas a nivel de clase completa para la Parte D
    grp_clase = df_bsd.groupby(["conductor", "clase"], as_index=False).agg(
        sha_max_clase=("sha_an", "max"),
        n_curvas_clase=("numero", "count"),
    )

    clases = (
        una_por_clase(df_bsd)
        .merge(grp_clase, on=["conductor", "clase"], how="left")
        .reset_index(drop=True)
    )

    # Discriminante exacto y su signo para cada clase (curva 1)
    discs_py = [discriminante(ai) for ai in clases["ainvs"]]
    signo_pos_all = np.array([d > 0 for d in discs_py], dtype=bool)
    clases["signo_pos"] = signo_pos_all
    clases["omega_mas"] = np.where(
        signo_pos_all, clases["omega"] / 2.0, clases["omega"]
    )

    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    filas_corr: list[dict[str, str | int | float]] = []
    filas_signo: list[dict[str, str | int | float]] = []

    # Partes A y B
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
            signo_pos = sub["signo_pos"].to_numpy(dtype=bool)
            bins_L_signo = bins_L * 2 + signo_pos.astype(int)

            omega = sub["omega"].to_numpy(dtype=np.float64)
            omega_mas = sub["omega_mas"].to_numpy(dtype=np.float64)
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)

            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)
            buenos_sub = (cond_sub[:, None] % primos[None, :]) != 0

            # Correlaciones parciales por primo (Parte A y Parte B.i/ii)
            for j, p_primo in enumerate(primos):
                p_int = int(p_primo)
                p_mod4 = int(p_int % 4)
                mask_b = buenos_sub[:, j]
                n_b = int(np.sum(mask_b))
                ap_p = ap_sub[mask_b, j].astype(np.float64)

                rho_L, n_ef_L = correlacion_parcial_bins(
                    ap_p, omega[mask_b], bins_L[mask_b]
                )
                rho_L_signo, _ = correlacion_parcial_bins(
                    ap_p, omega[mask_b], bins_L_signo[mask_b]
                )
                rho_L_om_mas, _ = correlacion_parcial_bins(
                    ap_p, omega_mas[mask_b], bins_L[mask_b]
                )
                se_val = float(1.0 / np.sqrt(n_ef_L)) if n_ef_L > 0 else float("nan")

                filas_corr.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "p_mod4": p_mod4,
                        "n": n_b,
                        "rho_L": rho_L,
                        "rho_L_signo": rho_L_signo,
                        "rho_L_omega_mas": rho_L_om_mas,
                        "se": se_val,
                    }
                )

            # Diferencia de ap medio entre Δ > 0 y Δ < 0 dentro de los bins de L (Parte B)
            delta_obs, _ = diferencia_perfiles(ap_sub, buenos_sub, signo_pos)
            deltas_nulos, _ = distribucion_nula_perfiles(
                ap_sub,
                buenos_sub,
                signo_pos,
                bins_L,
                n_perm=N_PERM,
                semilla=SEMILLA,
            )
            media_nula = deltas_nulos.mean(axis=0)
            desv_nula = deltas_nulos.std(axis=0, ddof=1)
            delta_ajustado_L = delta_obs - media_nula
            z_signo = delta_ajustado_L / desv_nula

            for j, p_primo in enumerate(primos):
                p_int = int(p_primo)
                filas_signo.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": p_int,
                        "p_mod4": int(p_int % 4),
                        "delta_signo": float(delta_ajustado_L[j]),
                        "z": float(z_signo[j]),
                    }
                )

    df_corr = pd.DataFrame(filas_corr)
    df_corr.to_csv(DIR_SALIDA / "correlaciones.csv", index=False)

    df_signo = pd.DataFrame(filas_signo)
    df_signo.to_csv(DIR_SALIDA / "signo.csv", index=False)

    # Parte C: Mecanismo del símbolo de Legendre (solo p impar, rango 0, todo N < 100 000)
    mask_r0 = (clases["rango"] == 0).to_numpy(dtype=bool)
    sub_r0 = clases.loc[mask_r0].reset_index(drop=True)
    discs_r0 = [d for d, m in zip(discs_py, mask_r0) if m]
    abs_discs_r0 = [abs(d) for d in discs_r0]
    signo_r0 = np.array([1 if d > 0 else -1 for d in discs_r0], dtype=np.int8)
    cond_r0 = sub_r0["conductor"].to_numpy(dtype=np.int32)
    ap_r0 = df_ap.loc[list(sub_r0["etiqueta"])].to_numpy(dtype=np.int16)

    filas_leg: list[dict[str, int | float]] = []
    for j, p_primo in enumerate(primos):
        p_int = int(p_primo)
        if p_int == 2:
            continue
        p_mod4 = int(p_int % 4)
        tabla_leg = [0 if r == 0 else simbolo_legendre(r, p_int) for r in range(p_int)]
        leg_minus1 = -1 if p_mod4 == 3 else 1

        mask_b = (cond_r0 % p_int) != 0
        leg_abs_all = np.array(
            [tabla_leg[ad % p_int] for ad in abs_discs_r0], dtype=np.int8
        )
        leg_abs = leg_abs_all[mask_b]
        sig_b = signo_r0[mask_b]
        leg_disc = np.where(sig_b > 0, leg_abs, leg_minus1 * leg_abs)
        ap_p = ap_r0[mask_b, j].astype(np.float64)

        prop_abs_res = float(np.mean(leg_abs == 1))
        corr_sig_leg = float(np.corrcoef(sig_b, leg_disc)[0, 1])

        mask_mas = leg_disc == 1
        mask_menos = leg_disc == -1
        n_mas = int(np.sum(mask_mas))
        n_menos = int(np.sum(mask_menos))
        media_mas = float(np.mean(ap_p[mask_mas]))
        media_menos = float(np.mean(ap_p[mask_menos]))

        filas_leg.append(
            {
                "p": p_int,
                "p_mod4": p_mod4,
                "prop_absdisc_residuo": prop_abs_res,
                "corr_signo_legendre": corr_sig_leg,
                "media_ap_leg_mas": media_mas,
                "media_ap_leg_menos": media_menos,
                "n_mas": n_mas,
                "n_menos": n_menos,
            }
        )

    df_leg = pd.DataFrame(filas_leg)
    df_leg.to_csv(DIR_SALIDA / "legendre.csv", index=False)

    # Parte D: H7 descriptivo por rango (0, 1, 2), ventana y tamaño de clase
    clases["cat_tamano"] = [
        _etiqueta_tamano(int(c)) for c in clases["n_curvas_clase"]
    ]
    filas_iso: list[dict[str, str | int | float]] = []

    for rango in (0, 1, 2):
        for nombre_vent, n_min, n_max in VENTANAS:
            base = clases.loc[
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ]
            for tam_et in ("todas", "1", "2", ">=3"):
                if tam_et == "todas":
                    sub_d = base
                else:
                    sub_d = base.loc[base["cat_tamano"] == tam_et]

                n_cls = len(sub_d)
                if n_cls == 0:
                    continue
                sha1_ge4 = np.rint(sub_d["sha_an"].to_numpy(dtype=float)).astype(int) >= 4
                shamax_ge4 = (
                    np.rint(sub_d["sha_max_clase"].to_numpy(dtype=float)).astype(int) >= 4
                )
                n_c1 = int(np.sum(sha1_ge4))
                n_max_c = int(np.sum(shamax_ge4))
                pct_c1 = 100.0 * n_c1 / n_cls
                pct_max = 100.0 * n_max_c / n_cls
                cociente = (n_max_c / n_c1) if n_c1 > 0 else float("nan")

                filas_iso.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "tamano_clase": tam_et,
                        "n_clases": n_cls,
                        "n_sha4_curva1": n_c1,
                        "pct_sha4_curva1": pct_c1,
                        "n_sha4_clase": n_max_c,
                        "pct_sha4_clase": pct_max,
                        "cociente": cociente,
                    }
                )

    df_iso = pd.DataFrame(filas_iso)
    df_iso.to_csv(DIR_SALIDA / "isogenia_sha.csv", index=False)

    # Figura correlaciones.png: rango 0, ventana N < 50 000
    sub_plot = df_corr.loc[
        (df_corr["rango"] == 0) & (df_corr["ventana"] == "1-49999")
    ].reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=150, sharey=True)
    colores_mod4 = {1: "#1f5a99", 3: "#c0392b", 2: "#7f8c8d"}
    etiquetas_mod4 = {
        3: "p ≡ 3 (mod 4)",
        1: "p ≡ 1 (mod 4)",
        2: "p = 2",
    }

    for ax, col_y, titulo in [
        (ax1, "rho_L", "A. Sin controlar signo(Δ): ρ(aₚ, Ω | L)"),
        (ax2, "rho_L_signo", "B. Controlando signo(Δ): ρ(aₚ, Ω | L, signo(Δ))"),
    ]:
        for mod_val in (3, 1, 2):
            m = sub_plot["p_mod4"] == mod_val
            sub_m = sub_plot.loc[m]
            ax.errorbar(
                sub_m["p"],
                sub_m[col_y],
                yerr=1.96 * sub_m["se"],
                fmt="o",
                color=colores_mod4[mod_val],
                ecolor=colores_mod4[mod_val],
                elinewidth=1.2,
                capsize=3,
                markersize=6,
                label=etiquetas_mod4[mod_val],
            )
        ax.plot(
            sub_plot["p"],
            sub_plot["rho_L_omega_mas"],
            "x--",
            color="#27ae60",
            linewidth=1.1,
            markersize=5,
            alpha=0.8,
            label="ρ(aₚ, Ω⁺ | L)",
        )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Primo p (p ≤ 97)")
        ax.set_ylabel("Correlación parcial ρₚ (±1.96 EE)")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    fig.suptitle(
        "Correlación parcial entre aₚ y el período Ω en rango 0, N < 50 000 (una curva por clase)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "correlaciones.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"Tiempo total e06_periodo: {t_fin - t_inicio:.2f} s")

    print("\n--- CORRELACIONES (Rango 0, primeros 10 primos en ambas ventanas) ---")
    print(
        df_corr.loc[(df_corr["rango"] == 0) & (df_corr["p"] <= 29)].to_string(
            index=False
        )
    )

    print("\n--- RESUMEN POR p mod 4 (sin p = 2) ---")
    for rango in (0, 1):
        for vent, _, _ in VENTANAS:
            sc = df_corr.loc[
                (df_corr["rango"] == rango)
                & (df_corr["ventana"] == vent)
                & (df_corr["p"] > 2)
            ]
            ss = df_signo.loc[
                (df_signo["rango"] == rango)
                & (df_signo["ventana"] == vent)
                & (df_signo["p"] > 2)
            ]
            for mod_val in (1, 3):
                mc = sc.loc[sc["p_mod4"] == mod_val]
                ms = ss.loc[ss["p_mod4"] == mod_val]
                print(
                    f"rango={rango} vent={vent} p≡{mod_val}(mod4): "
                    f"mean(rho_L)={mc['rho_L'].mean():+.5f}, "
                    f"mean(rho_L_signo)={mc['rho_L_signo'].mean():+.5f}, "
                    f"mean(rho_L_omega_mas)={mc['rho_L_omega_mas'].mean():+.5f}, "
                    f"mean(delta_signo)={ms['delta_signo'].mean():+.5f}, "
                    f"mean(z)={ms['z'].mean():+.2f}"
                )

    print("\n--- SIGNO (Rango 0, p <= 29) ---")
    print(
        df_signo.loc[(df_signo["rango"] == 0) & (df_signo["p"] <= 29)].to_string(
            index=False
        )
    )

    print("\n--- LEGENDRE (Rango 0, N < 100 000, p <= 31) ---")
    print(df_leg.loc[df_leg["p"] <= 31].to_string(index=False))

    print("\n--- ISOGENIA SHA (Parte D) ---")
    print(df_iso.to_string(index=False))

    return {"tiempo_s": t_fin - t_inicio}


if __name__ == "__main__":
    ejecutar_periodo_componentes()
