from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import cargar_cache_ap
from lab.estadistica import diferencia_perfiles, distribucion_nula_perfiles

SEMILLA = 20261121
N_PERM = 1000
N_PRIMOS = 25

RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "05-sha-murmuraciones"

VENTANAS = [
    ("10000-50000", 10_000, 50_000),
    ("50000-99999", 50_000, 99_999),
]


def _cat_tamagawa(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(len(arr), dtype=int)
    res[arr == 2] = 1
    res[(arr >= 3) & (arr <= 4)] = 2
    res[arr >= 5] = 3
    return res


def _cat_torsion(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(len(arr), dtype=int)
    res[arr == 2] = 1
    res[arr >= 3] = 2
    return res


def _construir_nulos(sub: pd.DataFrame, rango: int) -> dict[str, np.ndarray]:
    n = len(sub)
    b_l100 = pd.qcut(sub["l_valor"], q=100, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_l10 = pd.qcut(sub["l_valor"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_cond10 = pd.qcut(sub["conductor"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_om10 = pd.qcut(sub["omega"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_tam = _cat_tamagawa(sub["tamagawa"].to_numpy(dtype=int))
    b_tor = _cat_torsion(sub["torsion"].to_numpy(dtype=int))

    nulos: dict[str, np.ndarray] = {
        "sin_condicionar": np.zeros(n, dtype=int),
        "cond_L_100": b_l100,
        "cond_L10_conductor10": b_l10 * 100 + b_cond10,
        "cond_L10_omega10": b_l10 * 100 + b_om10,
    }
    if rango == 1:
        b_reg10 = pd.qcut(sub["regulador"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
        nulos["cond_L10_regulador10"] = b_l10 * 100 + b_reg10

    nulos["cond_L10_tamagawa"] = b_l10 * 10 + b_tam
    nulos["cond_L10_torsion"] = b_l10 * 10 + b_tor
    return nulos


def ejecutar_sha_murmuraciones() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=99_999)
    sha_max_df = (
        df_bsd.groupby(["conductor", "clase"], as_index=False)["sha_an"]
        .max()
        .rename(columns={"sha_an": "sha_max_clase"})
    )

    clases = una_por_clase(df_bsd).merge(
        sha_max_df, on=["conductor", "clase"], how="left"
    )
    df_ap = cargar_cache_ap(n_primos=N_PRIMOS)
    primos = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    filas_resumen: list[dict[str, str | int | float]] = []
    filas_delta: list[dict[str, str | int | float]] = []
    filas_isogenia: list[dict[str, str | int | float]] = []

    datos_grafico: dict[tuple[int, str], dict[str, np.ndarray]] = {}

    for rango in (0, 1):
        for nombre_vent, n_min, n_max in VENTANAS:
            sub = clases.loc[
                (clases["rango"] == rango)
                & (clases["conductor"] >= n_min)
                & (clases["conductor"] <= n_max)
            ].reset_index(drop=True)

            sha_c1 = np.rint(sub["sha_an"].to_numpy(dtype=float)).astype(int)
            mask_valid = (sha_c1 == 1) | (sha_c1 >= 4)
            sub = sub.loc[mask_valid].reset_index(drop=True)
            sha_c1 = sha_c1[mask_valid]

            grupo = sha_c1 >= 4
            n_sha1 = int(np.sum(~grupo))
            n_sha4 = int(np.sum(grupo))

            ap_sub = df_ap.loc[list(sub["etiqueta"])].to_numpy(dtype=np.int16)
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            buenos_sub = (cond_sub[:, None] % primos[None, :]) != 0

            delta_obs, rms_obs = diferencia_perfiles(ap_sub, buenos_sub, grupo)

            nulos_bins = _construir_nulos(sub, rango=rango)

            deltas_sin: np.ndarray | None = None
            deltas_L: np.ndarray | None = None

            for nombre_nulo, bins in nulos_bins.items():
                deltas_nulos, rms_nulos = distribucion_nula_perfiles(
                    ap_sub,
                    buenos_sub,
                    grupo,
                    bins,
                    n_perm=N_PERM,
                    semilla=SEMILLA,
                )
                media_nula = float(np.mean(rms_nulos))
                desv_nula = float(np.std(rms_nulos, ddof=1))
                p_val = float((1 + int(np.sum(rms_nulos >= rms_obs))) / (N_PERM + 1))

                filas_resumen.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "nulo": nombre_nulo,
                        "n_sha1": n_sha1,
                        "n_sha4": n_sha4,
                        "rms_obs": rms_obs,
                        "media_nula": media_nula,
                        "desv_nula": desv_nula,
                        "p": p_val,
                    }
                )

                if nombre_nulo == "sin_condicionar":
                    deltas_sin = deltas_nulos
                elif nombre_nulo == "cond_L_100":
                    deltas_L = deltas_nulos

            assert deltas_sin is not None and deltas_L is not None
            media_d_sin = deltas_sin.mean(axis=0)
            desv_d_sin = deltas_sin.std(axis=0, ddof=1)
            z_sin = (delta_obs - media_d_sin) / desv_d_sin

            media_d_L = deltas_L.mean(axis=0)
            desv_d_L = deltas_L.std(axis=0, ddof=1)
            z_L = (delta_obs - media_d_L) / desv_d_L

            for j, p_primo in enumerate(primos):
                filas_delta.append(
                    {
                        "rango": rango,
                        "ventana": nombre_vent,
                        "p": int(p_primo),
                        "delta": float(delta_obs[j]),
                        "z_sin_condicionar": float(z_sin[j]),
                        "z_cond_L": float(z_L[j]),
                    }
                )

            datos_grafico[(rango, nombre_vent)] = {
                "primos": primos,
                "delta": delta_obs,
                "media_sin": media_d_sin,
                "desv_sin": desv_d_sin,
                "media_L": media_d_L,
                "desv_L": desv_d_L,
            }

            # Sensibilidad por isogenia: máximo de Sha en la clase
            sha_max = np.rint(sub["sha_max_clase"].to_numpy(dtype=float)).astype(int)
            grupo_max = sha_max >= 4
            n_cambian = int(np.sum(grupo_max != grupo))
            n_sha1_max = int(np.sum(~grupo_max))
            n_sha4_max = int(np.sum(grupo_max))

            _, rms_obs_max = diferencia_perfiles(ap_sub, buenos_sub, grupo_max)
            _, rms_nulos_max = distribucion_nula_perfiles(
                ap_sub,
                buenos_sub,
                grupo_max,
                nulos_bins["cond_L_100"],
                n_perm=N_PERM,
                semilla=SEMILLA,
            )
            media_nula_max = float(np.mean(rms_nulos_max))
            desv_nula_max = float(np.std(rms_nulos_max, ddof=1))
            p_val_max = float(
                (1 + int(np.sum(rms_nulos_max >= rms_obs_max))) / (N_PERM + 1)
            )

            filas_isogenia.append(
                {
                    "rango": rango,
                    "ventana": nombre_vent,
                    "n_cambian": n_cambian,
                    "n_sha1_max": n_sha1_max,
                    "n_sha4_max": n_sha4_max,
                    "rms_obs_c1": rms_obs,
                    "rms_obs_max": rms_obs_max,
                    "media_nula_max": media_nula_max,
                    "desv_nula_max": desv_nula_max,
                    "p_cond_L_max": p_val_max,
                }
            )

    df_resumen = pd.DataFrame(filas_resumen)
    df_resumen.to_csv(DIR_SALIDA / "resumen.csv", index=False)

    df_delta = pd.DataFrame(filas_delta)
    df_delta.to_csv(DIR_SALIDA / "delta_p.csv", index=False)

    # Figura delta_p.png: un panel por rango (mostrando ventana principal y réplica con bandas ±2σ)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=150)

    for ax, rango, titulo in [
        (ax0, 0, "A. Rango 0: Δ(p) = ⟨aₚ⟩(Sha≥4) − ⟨aₚ⟩(Sha=1)"),
        (ax1, 1, "B. Rango 1: Δ(p) = ⟨aₚ⟩(Sha≥4) − ⟨aₚ⟩(Sha=1)"),
    ]:
        d_prin = datos_grafico[(rango, "10000-50000")]
        d_rep = datos_grafico[(rango, "50000-99999")]
        p_arr = d_prin["primos"]

        ax.fill_between(
            p_arr,
            d_prin["media_sin"] - 2.0 * d_prin["desv_sin"],
            d_prin["media_sin"] + 2.0 * d_prin["desv_sin"],
            color="#95a5a6",
            alpha=0.28,
            label="Nulo sin condicionar ±2σ (10k–50k)",
        )
        ax.fill_between(
            p_arr,
            d_prin["media_L"] - 2.0 * d_prin["desv_L"],
            d_prin["media_L"] + 2.0 * d_prin["desv_L"],
            color="#f39c12",
            alpha=0.28,
            label="Nulo condicionado por L ±2σ (10k–50k)",
        )
        ax.plot(
            p_arr,
            d_prin["media_L"],
            "--",
            color="#d68910",
            linewidth=1.2,
            label="Media nula condicionada por L (10k–50k)",
        )
        ax.plot(
            p_arr,
            d_prin["delta"],
            "o-",
            color="#1f5a99",
            markersize=4.5,
            linewidth=1.8,
            label="Δ(p) observado N ∈ [10 000, 50 000]",
        )
        ax.plot(
            p_arr,
            d_rep["delta"],
            "s-",
            color="#c0392b",
            markersize=4.0,
            linewidth=1.5,
            alpha=0.85,
            label="Δ(p) observado N ∈ [50 000, 99 999]",
        )
        ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Primo p (25 primeros primos, p ≤ 97)")
        ax.set_ylabel("Δ(p) = ⟨aₚ⟩(Sha ≥ 4) − ⟨aₚ⟩(Sha = 1)")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        "Efecto de Sha analítico sobre el perfil de aₚ (una curva por clase, p ∤ N)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_SALIDA / "delta_p.png", bbox_inches="tight")
    plt.close(fig)

    t_fin = time.perf_counter()
    print(f"Tiempo total e05_sha: {t_fin - t_inicio:.2f} s")
    print("\n--- RESUMEN ---")
    print(df_resumen.to_string(index=False))
    print("\n--- SENSIBILIDAD POR ISOGENIA (máximo Sha de la clase) ---")
    df_iso = pd.DataFrame(filas_isogenia)
    print(df_iso.to_string(index=False))

    return {
        "tiempo_s": t_fin - t_inicio,
        "resumen": df_resumen,
        "isogenia": df_iso,
    }


if __name__ == "__main__":
    ejecutar_sha_murmuraciones()
