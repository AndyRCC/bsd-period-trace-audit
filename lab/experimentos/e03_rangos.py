from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase

RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "resultados" / "03-distribucion-rangos"


def ejecutar_rangos() -> pd.DataFrame:
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    df_bsd = cargar_bsd(n_max=99_999)
    clases = una_por_clase(df_bsd)

    filas: list[dict[str, str | int | float]] = []
    for b in range(10):
        n_min = 1 if b == 0 else b * 10_000
        n_max = b * 10_000 + 9_999
        etiqueta_bloque = f"{n_min}-{n_max}"

        sub = clases.loc[(clases["conductor"] >= n_min) & (clases["conductor"] <= n_max)]
        n_clases = len(sub)
        conteos = sub["rango"].value_counts().to_dict()
        r0 = int(conteos.get(0, 0))
        r1 = int(conteos.get(1, 0))
        r2 = int(conteos.get(2, 0))
        r3 = int(conteos.get(3, 0))

        if b == 0:
            esperado_b0 = (16_450, 19_622, 1_969, 1)
            if (r0, r1, r2, r3) != esperado_b0 or n_clases != 38_042:
                raise RuntimeError(
                    f"Conteos del bloque 1-9999 no coinciden con los publicados: "
                    f"obtenido={(r0, r1, r2, r3)} (n={n_clases}), esperado={esperado_b0}"
                )

        rango_vals = sub["rango"].to_numpy(dtype=float)
        rango_medio = float(np.mean(rango_vals))
        error_estandar = float(np.std(rango_vals, ddof=1) / np.sqrt(n_clases))

        impar_vals = (sub["rango"].to_numpy(dtype=int) % 2 == 1).astype(float)
        prop_impar = float(np.mean(impar_vals))
        err_impar = float(np.sqrt(prop_impar * (1.0 - prop_impar) / n_clases))

        filas.append(
            {
                "bloque": etiqueta_bloque,
                "n_min": n_min,
                "n_max": n_max,
                "n_clases": n_clases,
                "r0": r0,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "prop_r0": r0 / n_clases,
                "prop_r1": r1 / n_clases,
                "prop_r2": r2 / n_clases,
                "prop_r3": r3 / n_clases,
                "rango_medio": rango_medio,
                "error_estandar": error_estandar,
                "prop_paridad_impar": prop_impar,
                "error_estandar_paridad": err_impar,
            }
        )

    tabla = pd.DataFrame(filas)
    ruta_csv = DIR_SALIDA / "tabla.csv"
    tabla.to_csv(ruta_csv, index=False)

    # Figura de dos paneles
    x = np.arange(len(tabla))
    etiquetas_x = [f"{b * 10}k–{b * 10 + 10}k" if b > 0 else "1–10k" for b in range(10)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    ax1.errorbar(
        x,
        tabla["rango_medio"],
        yerr=tabla["error_estandar"],
        fmt="o-",
        color="#1f5a99",
        ecolor="#1f5a99",
        elinewidth=1.3,
        capsize=4,
        markersize=6,
        linewidth=1.8,
        label="Rango medio ± EE",
    )
    ax1.axhline(0.5, color="#888888", linestyle="--", linewidth=1.1, label="Predicción asintótica (0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(etiquetas_x, rotation=35, ha="right")
    ax1.set_xlabel("Bloque de conductor N")
    ax1.set_ylabel("Rango medio por clase de isogenia")
    ax1.set_title("A. Rango medio por bloque de 10 000 conductores")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.errorbar(
        x,
        tabla["prop_paridad_impar"],
        yerr=tabla["error_estandar_paridad"],
        fmt="s-",
        color="#b8332a",
        ecolor="#b8332a",
        elinewidth=1.3,
        capsize=4,
        markersize=6,
        linewidth=1.8,
        label="Proporción rango impar (ε = −1) ± EE",
    )
    ax2.axhline(0.5, color="#222222", linestyle="--", linewidth=1.2, label="Equidistribución (0.5)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiquetas_x, rotation=35, ha="right")
    ax2.set_xlabel("Bloque de conductor N")
    ax2.set_ylabel("Proporción de clases con rango impar")
    ax2.set_title("B. Proporción de paridad impar (ε = −1) por bloque")
    ax2.set_ylim(0.48, 0.56)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Distribución de rangos y paridad en las tablas de Cremona (N ≤ 99 999, una curva por clase)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    ruta_fig = DIR_SALIDA / "figura.png"
    fig.savefig(ruta_fig, bbox_inches="tight")
    plt.close(fig)

    print(tabla.to_string(index=False))
    return tabla


if __name__ == "__main__":
    ejecutar_rangos()
