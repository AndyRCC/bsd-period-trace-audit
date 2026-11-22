from __future__ import annotations

import functools
from pathlib import Path
import numpy as np
import pandas as pd

from lab.datos import cargar_bsd, una_por_clase
from lab.gp import coeficientes_ap

RAIZ = Path(__file__).resolve().parents[1]
DIR_DERIVADOS = RAIZ / "datos" / "derivados"
TAMANO_LOTE_MAX = 50_000


def discriminante(ainvs: tuple[int, ...]) -> int:
    if len(ainvs) != 5:
        raise ValueError("Se esperaban 5 coeficientes de Weierstrass (a1, a2, a3, a4, a6)")
    a1, a2, a3, a4, a6 = (int(x) for x in ainvs)
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = (
        a1 * a1 * a6
        + 4 * a2 * a6
        - a1 * a3 * a4
        + a2 * a3 * a3
        - a4 * a4
    )
    return -b2 * b2 * b8 - 8 * (b4**3) - 27 * (b6 * b6) + 9 * b2 * b4 * b6


def construir_cache_ap(n_primos: int = 25) -> Path:
    if n_primos <= 0:
        raise ValueError("n_primos debe ser positivo")
    DIR_DERIVADOS.mkdir(parents=True, exist_ok=True)
    ruta = DIR_DERIVADOS / f"ap_{int(n_primos)}.npz"

    clases = una_por_clase(cargar_bsd(n_max=99_999)).reset_index(drop=True)
    total = len(clases)

    if ruta.exists():
        with np.load(ruta, allow_pickle=False) as datos:
            if (
                "etiquetas" in datos
                and "primos" in datos
                and "ap" in datos
                and datos["ap"].shape == (total, int(n_primos))
            ):
                return ruta

    curvas = [(fila["etiqueta"], fila["ainvs"]) for _, fila in clases.iterrows()]
    bloques_ap: list[np.ndarray] = []
    primos_arr: np.ndarray | None = None

    for inicio in range(0, total, TAMANO_LOTE_MAX):
        fin = min(inicio + TAMANO_LOTE_MAX, total)
        lote = curvas[inicio:fin]
        df_lote = coeficientes_ap(lote, n_primos=int(n_primos))
        if primos_arr is None:
            primos_arr = np.array([int(p) for p in df_lote.columns], dtype=np.int32)
        bloques_ap.append(df_lote.to_numpy(dtype=np.int16))

    if primos_arr is None:
        raise RuntimeError("No se pudieron calcular coeficientes ap")

    matriz_ap = np.vstack(bloques_ap).astype(np.int16, copy=False)
    etiquetas_arr = clases["etiqueta"].to_numpy(dtype="U16")

    temporal = ruta.with_suffix(".tmp.npz")
    np.savez(
        temporal,
        etiquetas=etiquetas_arr,
        primos=primos_arr,
        ap=matriz_ap,
    )
    temporal.replace(ruta)
    _cargar_cache_ap_mem.cache_clear()
    return ruta


@functools.lru_cache(maxsize=4)
def _cargar_cache_ap_mem(n_primos: int) -> pd.DataFrame:
    ruta = construir_cache_ap(n_primos=n_primos)
    with np.load(ruta, allow_pickle=False) as datos:
        etiquetas = datos["etiquetas"]
        primos = [int(p) for p in datos["primos"]]
        ap = datos["ap"].astype(np.int16, copy=True)
    return pd.DataFrame(ap, index=pd.Index(etiquetas, name="etiqueta"), columns=primos)


def cargar_cache_ap(n_primos: int = 25) -> pd.DataFrame:
    return _cargar_cache_ap_mem(int(n_primos)).copy()


def simbolo_legendre(a: int, p: int) -> int:
    p_int = int(p)
    if p_int <= 2 or p_int % 2 == 0:
        raise ValueError("p debe ser un primo impar")
    r = int(a) % p_int
    if r == 0:
        return 0
    e = pow(r, (p_int - 1) // 2, p_int)
    return 1 if e == 1 else -1

