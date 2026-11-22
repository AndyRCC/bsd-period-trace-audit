from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.gp import coeficientes_ap, promedios_ap

SEMILLA = 20261121
N_PRIMOS_PRINCIPAL = 1000
N_BARAJADOS = 100
P_MAX_FUERA = 20_360
N_BINS_X = 50
X_MAX = 0.9

RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "04-murmuraciones"


def control_barajado(
    matriz_ap: np.ndarray,
    rangos: np.ndarray,
    n_barajados: int = N_BARAJADOS,
    semilla: int = SEMILLA,
) -> np.ndarray:
    ap = np.asarray(matriz_ap, dtype=np.float64)
    r = np.asarray(rangos)
    n0 = int(np.sum(r == 0))
    n1 = int(np.sum(r == 1))
    if n0 == 0 or n1 == 0:
        raise ValueError("Se requieren curvas de rango 0 y de rango 1")

    suma_total = ap.sum(axis=0)
    rng = np.random.default_rng(semilla)
    s_nulos = np.empty(n_barajados, dtype=np.float64)

    for b in range(n_barajados):
        perm = rng.permutation(r)
        suma0 = ap[perm == 0].sum(axis=0)
        f0 = suma0 / n0
        f1 = (suma_total - suma0) / n1
        d = f0 - f1
        s_nulos[b] = float(np.sqrt(np.mean(d * d)))

    return s_nulos


def _binar_en_x(
    primos: np.ndarray,
    valores: np.ndarray,
    n_medio: float,
    n_bins: int = N_BINS_X,
    x_max: float = X_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    x = primos / float(n_medio)
    bordes = np.linspace(0.0, x_max, n_bins + 1)
    centros = 0.5 * (bordes[:-1] + bordes[1:])
    medias = np.full(n_bins, np.nan, dtype=np.float64)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (x >= bordes[i]) & (x < bordes[i + 1])
        else:
            mask = (x >= bordes[i]) & (x <= bordes[i + 1])
        if np.any(mask):
            medias[i] = float(np.mean(valores[mask]))

    return centros, medias


def ejecutar_murmuraciones() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd(n_max=30_000)
    clases = una_por_clase(df_bsd)

    # 1. Rango principal [7500, 10000], solo rangos 0 y 1
    sub_prin = clases.loc[
        (clases["conductor"] >= 7_500)
        & (clases["conductor"] <= 10_000)
        & (clases["rango"].isin([0, 1]))
    ].reset_index(drop=True)

    n0_prin = int((sub_prin["rango"] == 0).sum())
    n1_prin = int((sub_prin["rango"] == 1).sum())
    n_medio_prin = float(sub_prin["conductor"].mean())

    curvas_prin_ap = [
        (fila["etiqueta"], fila["ainvs"]) for _, fila in sub_prin.iterrows()
    ]
    t_ap0 = time.perf_counter()
    df_ap = coeficientes_ap(curvas_prin_ap, n_primos=N_PRIMOS_PRINCIPAL)
    t_ap1 = time.perf_counter()

    primos_prin = np.array(df_ap.columns, dtype=np.int32)
    matriz_ap = df_ap.to_numpy(dtype=np.int16)
    rangos_prin = sub_prin["rango"].to_numpy(dtype=np.int8)
    conductores_prin = sub_prin["conductor"].to_numpy(dtype=np.int32)

    # Versión principal: todos los primos
    ap_f64 = matriz_ap.astype(np.float64)
    f0_todos = ap_f64[rangos_prin == 0].mean(axis=0)
    f1_todos = ap_f64[rangos_prin == 1].mean(axis=0)
    d_todos = f0_todos - f1_todos
    s_real_todos = float(np.sqrt(np.mean(d_todos * d_todos)))
    corr_f0_f1_todos = float(np.corrcoef(f0_todos, f1_todos)[0, 1])

    # Variante: sin los primos que dividen al conductor
    buenos = (conductores_prin[:, None] % primos_prin[None, :]) != 0
    mask0 = buenos[rangos_prin == 0]
    mask1 = buenos[rangos_prin == 1]
    n0_buenos = mask0.sum(axis=0)
    n1_buenos = mask1.sum(axis=0)
    suma0_buenos = np.where(mask0, ap_f64[rangos_prin == 0], 0.0).sum(axis=0)
    suma1_buenos = np.where(mask1, ap_f64[rangos_prin == 1], 0.0).sum(axis=0)
    f0_sin_malos = suma0_buenos / n0_buenos
    f1_sin_malos = suma1_buenos / n1_buenos
    d_sin_malos = f0_sin_malos - f1_sin_malos
    s_real_sin_malos = float(np.sqrt(np.mean(d_sin_malos * d_sin_malos)))
    corr_f0_f1_sin_malos = float(np.corrcoef(f0_sin_malos, f1_sin_malos)[0, 1])

    # 2. Control: 100 barajados con semilla 20261121
    s_nulos = control_barajado(
        matriz_ap, rangos_prin, n_barajados=N_BARAJADOS, semilla=SEMILLA
    )
    media_nulos = float(np.mean(s_nulos))
    desv_nulos = float(np.std(s_nulos, ddof=1))
    max_nulos = float(np.max(s_nulos))
    z_score = (s_real_todos - media_nulos) / desv_nulos

    # Calcular también correlaciones nulas entre f0 y f1 como descripción
    rng_corr = np.random.default_rng(SEMILLA)
    suma_tot = ap_f64.sum(axis=0)
    corrs_nulas = np.empty(N_BARAJADOS, dtype=np.float64)
    for b in range(N_BARAJADOS):
        perm = rng_corr.permutation(rangos_prin)
        s0 = ap_f64[perm == 0].sum(axis=0)
        f0_b = s0 / n0_prin
        f1_b = (suma_tot - s0) / n1_prin
        corrs_nulas[b] = float(np.corrcoef(f0_b, f1_b)[0, 1])

    df_nulos = pd.DataFrame(
        {
            "iteracion": np.arange(1, N_BARAJADOS + 1),
            "s_nulo": s_nulos,
            "corr_f0_f1_nula": corrs_nulas,
        }
    )
    df_nulos.to_csv(DIR_SALIDA / "nulos.csv", index=False)

    # 3. Fuera de muestra: medir tiempo con 500 curvas y extrapolar
    sub_fuera_cand = clases.loc[
        (clases["conductor"] >= 20_000)
        & (clases["conductor"] <= 25_000)
        & (clases["rango"].isin([0, 1]))
    ].reset_index(drop=True)

    lote_bench = [
        (fila["etiqueta"], fila["ainvs"], int(fila["rango"]))
        for _, fila in sub_fuera_cand.iloc[:500].iterrows()
    ]
    t_b0 = time.perf_counter()
    _ = promedios_ap(lote_bench, p_max=P_MAX_FUERA, excluir_malos=False)
    t_b1 = time.perf_counter()
    t_bench_500 = t_b1 - t_b0
    t_extrapolado = t_bench_500 * (len(sub_fuera_cand) / 500.0)

    if t_extrapolado > 600.0:
        n_max_fuera = 22_500
        etiqueta_rango_fuera = "20000-22500"
        sub_fuera = clases.loc[
            (clases["conductor"] >= 20_000)
            & (clases["conductor"] <= 22_500)
            & (clases["rango"].isin([0, 1]))
        ].reset_index(drop=True)
    else:
        n_max_fuera = 25_000
        etiqueta_rango_fuera = "20000-25000"
        sub_fuera = sub_fuera_cand

    n0_fuera = int((sub_fuera["rango"] == 0).sum())
    n1_fuera = int((sub_fuera["rango"] == 1).sum())
    n_medio_fuera = float(sub_fuera["conductor"].mean())

    lote_fuera = [
        (fila["etiqueta"], fila["ainvs"], int(fila["rango"]))
        for _, fila in sub_fuera.iterrows()
    ]
    t_f0 = time.perf_counter()
    df_prom_fuera = promedios_ap(lote_fuera, p_max=P_MAX_FUERA, excluir_malos=False)
    t_f1 = time.perf_counter()
    t_gp_fuera = t_f1 - t_f0

    f0_fuera_df = df_prom_fuera.loc[df_prom_fuera["grupo"] == 0].sort_values("p")
    f1_fuera_df = df_prom_fuera.loc[df_prom_fuera["grupo"] == 1].sort_values("p")
    primos_fuera = f0_fuera_df["p"].to_numpy(dtype=np.int32)
    f0_fuera = f0_fuera_df["media"].to_numpy(dtype=np.float64)
    f1_fuera = f1_fuera_df["media"].to_numpy(dtype=np.float64)
    d_fuera = f0_fuera - f1_fuera
    s_fuera = float(np.sqrt(np.mean(d_fuera * d_fuera)))

    # 4. Guardar promedios.csv: columnas rango_N, variante, grupo, p, n, media
    filas_prom: list[pd.DataFrame] = []
    for g, n_arr, f_arr in [
        (0, np.full(len(primos_prin), n0_prin, dtype=int), f0_todos),
        (1, np.full(len(primos_prin), n1_prin, dtype=int), f1_todos),
    ]:
        filas_prom.append(
            pd.DataFrame(
                {
                    "rango_N": "7500-10000",
                    "variante": "todos",
                    "grupo": g,
                    "p": primos_prin,
                    "n": n_arr,
                    "media": f_arr,
                }
            )
        )
    for g, n_arr, f_arr in [
        (0, n0_buenos, f0_sin_malos),
        (1, n1_buenos, f1_sin_malos),
    ]:
        filas_prom.append(
            pd.DataFrame(
                {
                    "rango_N": "7500-10000",
                    "variante": "sin_malos",
                    "grupo": g,
                    "p": primos_prin,
                    "n": n_arr,
                    "media": f_arr,
                }
            )
        )
    df_fuera_out = df_prom_fuera[["grupo", "p", "n", "media"]].copy()
    df_fuera_out.insert(0, "variante", "todos")
    df_fuera_out.insert(0, "rango_N", etiqueta_rango_fuera)
    filas_prom.append(df_fuera_out)

    df_promedios_total = pd.concat(filas_prom, ignore_index=True)
    df_promedios_total.to_csv(DIR_SALIDA / "promedios.csv", index=False)

    # 5. Binado en 50 intervalos de x = p / N_medio en [0, 0.9]
    centros_x, d_bin_prin_bruto = _binar_en_x(primos_prin, d_todos, n_medio_prin)
    _, d_bin_fuera_bruto = _binar_en_x(primos_fuera, d_fuera, n_medio_fuera)

    d_norm_prin = d_todos / np.sqrt(primos_prin)
    d_norm_fuera = d_fuera / np.sqrt(primos_fuera)
    _, d_bin_prin_norm = _binar_en_x(primos_prin, d_norm_prin, n_medio_prin)
    _, d_bin_fuera_norm = _binar_en_x(primos_fuera, d_norm_fuera, n_medio_fuera)

    mask_bins_bruto = np.isfinite(d_bin_prin_bruto) & np.isfinite(d_bin_fuera_bruto)
    corr_escala_bruto = float(
        np.corrcoef(
            d_bin_prin_bruto[mask_bins_bruto], d_bin_fuera_bruto[mask_bins_bruto]
        )[0, 1]
    )
    mask_bins_norm = np.isfinite(d_bin_prin_norm) & np.isfinite(d_bin_fuera_norm)
    corr_escala_norm = float(
        np.corrcoef(
            d_bin_prin_norm[mask_bins_norm], d_bin_fuera_norm[mask_bins_norm]
        )[0, 1]
    )

    # 6. Figuras
    # Figura 1: murmuraciones.png
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150, sharey=True)
    ax1.scatter(primos_prin, f0_todos, s=8, alpha=0.55, color="#1f5a99", label=f"Rango 0 (n={n0_prin})")
    ax1.scatter(primos_prin, f1_todos, s=8, alpha=0.55, color="#c0392b", label=f"Rango 1 (n={n1_prin})")
    ax1.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax1.set_xlabel("Primo p (primeros 1 000 primos, p ≤ 7 919)")
    ax1.set_ylabel("Promedio de aₚ")
    ax1.set_title(f"A. Todos los primos (S = {s_real_todos:.4f})")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    ax2.scatter(primos_prin, f0_sin_malos, s=8, alpha=0.55, color="#1f5a99", label="Rango 0 (p ∤ N)")
    ax2.scatter(primos_prin, f1_sin_malos, s=8, alpha=0.55, color="#c0392b", label="Rango 1 (p ∤ N)")
    ax2.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax2.set_xlabel("Primo p (primeros 1 000 primos, p ≤ 7 919)")
    ax2.set_title(f"B. Excluyendo primos p | N (S = {s_real_sin_malos:.4f})")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    fig1.suptitle(
        "Murmuraciones de curvas elípticas en N ∈ [7 500, 10 000] (una curva por clase)",
        fontsize=12,
        fontweight="bold",
    )
    fig1.tight_layout()
    fig1.savefig(DIR_SALIDA / "murmuraciones.png", bbox_inches="tight")
    plt.close(fig1)

    # Figura 2: control.png
    fig2, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    ax.hist(
        s_nulos,
        bins=15,
        color="#7f8c8d",
        edgecolor="#2c3e50",
        alpha=0.8,
        label=f"Distribución nula (100 barajados, μ={media_nulos:.4f}, σ={desv_nulos:.4f})",
    )
    ax.axvline(
        s_real_todos,
        color="#c0392b",
        linewidth=2.2,
        linestyle="-",
        label=f"S real = {s_real_todos:.4f} (z = {z_score:.2f})",
    )
    ax.set_xlabel("Estadístico S = √(promedio de (f₀(p) − f₁(p))²)")
    ax.set_ylabel("Frecuencia (de 100 barajados)")
    ax.set_title("Control por permutación de etiquetas de rango en N ∈ [7 500, 10 000]")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper center")
    fig2.tight_layout()
    fig2.savefig(DIR_SALIDA / "control.png", bbox_inches="tight")
    plt.close(fig2)

    # Figura 3: escala.png
    fig3, (ax_b, ax_n) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)
    ax_b.plot(
        centros_x,
        d_bin_prin_bruto,
        "o-",
        color="#1f5a99",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [7 500, 10 000] (N̄={n_medio_prin:.0f})",
    )
    ax_b.plot(
        centros_x,
        d_bin_fuera_bruto,
        "s-",
        color="#d35400",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [{etiqueta_rango_fuera.replace('-', ', ')}] (N̄={n_medio_fuera:.0f})",
    )
    ax_b.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_b.set_xlabel("Escala adimensional x = p / N̄")
    ax_b.set_ylabel("Promedio por bin de D(p) = f₀(p) − f₁(p)")
    ax_b.set_title(f"A. D(p) en bruto (r = {corr_escala_bruto:.4f})")
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper left")

    ax_n.plot(
        centros_x,
        d_bin_prin_norm,
        "o-",
        color="#1f5a99",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [7 500, 10 000] (N̄={n_medio_prin:.0f})",
    )
    ax_n.plot(
        centros_x,
        d_bin_fuera_norm,
        "s-",
        color="#d35400",
        markersize=4,
        linewidth=1.6,
        label=f"N ∈ [{etiqueta_rango_fuera.replace('-', ', ')}] (N̄={n_medio_fuera:.0f})",
    )
    ax_n.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_n.set_xlabel("Escala adimensional x = p / N̄")
    ax_n.set_ylabel("Promedio por bin de D(p) / √p")
    ax_n.set_title(f"B. D(p) / √p normalizado (r = {corr_escala_norm:.4f})")
    ax_n.grid(True, linestyle=":", alpha=0.6)
    ax_n.legend(loc="upper left")

    fig3.suptitle(
        "Invariancia de escala de las murmuraciones en 50 intervalos de x = p / N̄ ∈ [0, 0.9]",
        fontsize=12,
        fontweight="bold",
    )
    fig3.tight_layout()
    fig3.savefig(DIR_SALIDA / "escala.png", bbox_inches="tight")
    plt.close(fig3)

    t_fin = time.perf_counter()

    resumen = {
        "n0_prin": n0_prin,
        "n1_prin": n1_prin,
        "n_medio_prin": n_medio_prin,
        "t_ap_prin_s": t_ap1 - t_ap0,
        "s_real_todos": s_real_todos,
        "s_real_sin_malos": s_real_sin_malos,
        "corr_f0_f1_todos": corr_f0_f1_todos,
        "corr_f0_f1_sin_malos": corr_f0_f1_sin_malos,
        "media_nulos": media_nulos,
        "desv_nulos": desv_nulos,
        "max_nulos": max_nulos,
        "z_score": z_score,
        "corr_nula_media": float(np.mean(corrs_nulas)),
        "t_bench_500_s": t_bench_500,
        "t_extrapolado_s": t_extrapolado,
        "etiqueta_rango_fuera": etiqueta_rango_fuera,
        "n0_fuera": n0_fuera,
        "n1_fuera": n1_fuera,
        "n_medio_fuera": n_medio_fuera,
        "n_primos_fuera": len(primos_fuera),
        "p_ult_fuera": int(primos_fuera[-1]),
        "t_gp_fuera_s": t_gp_fuera,
        "s_fuera": s_fuera,
        "corr_escala_bruto": corr_escala_bruto,
        "corr_escala_norm": corr_escala_norm,
        "t_total_s": t_fin - t_inicio,
    }

    for k, v in resumen.items():
        print(f"{k}: {v}")
    return resumen


if __name__ == "__main__":
    ejecutar_murmuraciones()
