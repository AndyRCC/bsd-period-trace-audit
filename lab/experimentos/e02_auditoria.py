from __future__ import annotations

import math
from pathlib import Path
import time
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, cargar_generadores
from lab.gp import invariantes_pari

SEMILLA = 20261121
N_MUESTRA = 1000
N_RANGO3 = 20
TOL_REAL = 1e-6

RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "02-auditoria-datos"


def _cuadrado_mas_cercano(x: float) -> tuple[int, float]:
    if not np.isfinite(x) or x <= 0:
        return 0, float("nan")
    raiz = max(1, int(round(math.sqrt(x))))
    mejor_c = raiz * raiz
    mejor_d = abs(x - mejor_c)
    for k in (max(1, raiz - 1), raiz + 1):
        c = k * k
        d = abs(x - c)
        if d < mejor_d:
            mejor_c = c
            mejor_d = d
    return mejor_c, mejor_d


def ejecutar_auditoria() -> dict:
    t_inicio = time.perf_counter()
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)

    df_bsd = cargar_bsd()
    df_gens = cargar_generadores()
    df_completo = df_bsd.merge(df_gens, on="etiqueta", how="inner")

    rng = np.random.default_rng(SEMILLA)
    indices_1000 = np.sort(rng.choice(len(df_completo), size=N_MUESTRA, replace=False))
    muestra_1000 = df_completo.iloc[indices_1000]

    etiquetas_ya = set(muestra_1000["etiqueta"])
    candidatas_r3 = df_completo.loc[
        (df_completo["rango"] == 3) & (~df_completo["etiqueta"].isin(etiquetas_ya))
    ].reset_index(drop=True)

    rng_r3 = np.random.default_rng(SEMILLA)
    indices_r3 = np.sort(rng_r3.choice(len(candidatas_r3), size=N_RANGO3, replace=False))
    muestra_r3 = candidatas_r3.iloc[indices_r3]

    muestra = pd.concat([muestra_1000, muestra_r3], ignore_index=True)

    lote_curvas = [
        (fila["etiqueta"], fila["ainvs"], fila["gens"])
        for _, fila in muestra.iterrows()
    ]

    t_gp0 = time.perf_counter()
    df_pari = invariantes_pari(lote_curvas)
    t_gp1 = time.perf_counter()

    comp = pd.DataFrame(
        {
            "etiqueta": muestra["etiqueta"],
            "conductor": muestra["conductor"],
            "rango_tabla": muestra["rango"],
            "rango_pari": df_pari["rango_an"],
            "l_valor_tabla": muestra["l_valor"],
            "l_valor_pari": df_pari["l_valor"],
            "omega_tabla": muestra["omega"],
            "omega_pari": df_pari["omega"],
            "tamagawa_tabla": muestra["tamagawa"],
            "tamagawa_pari": df_pari["tamagawa"],
            "torsion_tabla": muestra["torsion"],
            "torsion_pari": df_pari["torsion"],
            "regulador_tabla": muestra["regulador"],
            "regulador_pari": df_pari["regulador"],
            "sha_an_tabla": muestra["sha_an"],
            "raiz_pari": df_pari["raiz"],
        }
    )

    comp["coincide_rango"] = comp["rango_tabla"] == comp["rango_pari"]
    comp["coincide_tamagawa"] = comp["tamagawa_tabla"] == comp["tamagawa_pari"]
    comp["coincide_torsion"] = comp["torsion_tabla"] == comp["torsion_pari"]

    comp["err_rel_l_valor"] = (
        np.abs(comp["l_valor_pari"] - comp["l_valor_tabla"]) / comp["l_valor_tabla"]
    )
    comp["coincide_l_valor"] = comp["err_rel_l_valor"] <= TOL_REAL

    comp["err_rel_omega"] = (
        np.abs(comp["omega_pari"] - comp["omega_tabla"]) / comp["omega_tabla"]
    )
    comp["coincide_omega"] = comp["err_rel_omega"] <= TOL_REAL

    comp["err_rel_regulador"] = (
        np.abs(comp["regulador_pari"] - comp["regulador_tabla"]) / comp["regulador_tabla"]
    )
    comp["coincide_regulador"] = comp["err_rel_regulador"] <= TOL_REAL

    comp["coincide_paridad"] = ((-1) ** comp["rango_pari"]) == comp["raiz_pari"]

    denom = comp["omega_pari"] * comp["regulador_pari"] * comp["tamagawa_pari"]
    comp["sha_calc"] = comp["l_valor_pari"] * (comp["torsion_pari"] ** 2) / denom

    cuad_lista = [_cuadrado_mas_cercano(v) for v in comp["sha_calc"]]
    comp["cuadrado_mas_cercano"] = [c for c, _ in cuad_lista]
    comp["dist_cuadrado_perfecto"] = [d for _, d in cuad_lista]

    ruta_csv = DIR_SALIDA / "comparacion.csv"
    comp.to_csv(ruta_csv, index=False)

    t_fin = time.perf_counter()

    resumen = {
        "n_muestra": len(comp),
        "tiempo_total_s": t_fin - t_inicio,
        "tiempo_gp_s": t_gp1 - t_gp0,
        "por_rango": comp["rango_tabla"].value_counts().sort_index().to_dict(),
        "pct_rango": 100.0 * float(comp["coincide_rango"].mean()),
        "pct_l_valor": 100.0 * float(comp["coincide_l_valor"].mean()),
        "pct_omega": 100.0 * float(comp["coincide_omega"].mean()),
        "pct_tamagawa": 100.0 * float(comp["coincide_tamagawa"].mean()),
        "pct_torsion": 100.0 * float(comp["coincide_torsion"].mean()),
        "pct_regulador": 100.0 * float(comp["coincide_regulador"].mean()),
        "pct_paridad": 100.0 * float(comp["coincide_paridad"].mean()),
    }

    print(f"Muestra: {resumen['n_muestra']} curvas (1000 al azar + 20 de rango 3, semilla={SEMILLA})")
    print(f"Tiempo total: {resumen['tiempo_total_s']:.2f} s (GP lote 1020: {resumen['tiempo_gp_s']:.2f} s)")
    print(f"Distribución por rango en la muestra: {resumen['por_rango']}")
    print(
        f"Coincidencias (%): rango={resumen['pct_rango']:.1f}%, "
        f"l_valor={resumen['pct_l_valor']:.1f}%, omega={resumen['pct_omega']:.1f}%, "
        f"tamagawa={resumen['pct_tamagawa']:.1f}%, torsion={resumen['pct_torsion']:.1f}%, "
        f"regulador={resumen['pct_regulador']:.1f}%, paridad={resumen['pct_paridad']:.1f}%"
    )

    for col_err, nombre in [
        ("err_rel_l_valor", "l_valor"),
        ("err_rel_omega", "omega"),
        ("err_rel_regulador", "regulador"),
    ]:
        idx_max = int(comp[col_err].idxmax())
        fila_max = comp.iloc[idx_max]
        print(
            f"Peor discrepancia {nombre}: etiqueta={fila_max['etiqueta']}, "
            f"err_rel={fila_max[col_err]:.3e}"
        )

    print(
        f"sha_calc válido en {comp['dist_cuadrado_perfecto'].notna().sum()}/{len(comp)} curvas; "
        f"max dist a cuadrado perfecto: {comp['dist_cuadrado_perfecto'].max():.3e}"
    )
    return resumen


if __name__ == "__main__":
    ejecutar_auditoria()
