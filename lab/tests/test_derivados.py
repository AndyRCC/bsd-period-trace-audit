import numpy as np

from lab.datos import cargar_bsd, una_por_clase
from lab.derivados import (
    cargar_cache_ap,
    construir_cache_ap,
    discriminante,
    simbolo_legendre,
)
from lab.gp import coeficientes_ap


def test_discriminante_referencia():
    assert discriminante((0, -1, 1, -10, -20)) == -161051
    assert discriminante((0, 0, 1, -1, 0)) == 37
    assert discriminante((0, 1, 1, -2, 0)) == 389


def test_cache_completo():
    ruta = construir_cache_ap(n_primos=25)
    assert ruta.exists()
    df_cache = cargar_cache_ap(n_primos=25)
    assert df_cache.index.name == "etiqueta"
    assert df_cache.shape == (437_226, 25)
    assert int(df_cache.columns[0]) == 2
    assert int(df_cache.columns[-1]) == 97


def test_cache_contra_coeficientes():
    clases = una_por_clase(cargar_bsd(n_max=99_999)).reset_index(drop=True)
    rng = np.random.default_rng(20261121)
    idxs = np.sort(rng.choice(len(clases), size=20, replace=False))
    muestra = clases.iloc[idxs]

    df_cache = cargar_cache_ap(n_primos=25)
    sub_cache = df_cache.loc[list(muestra["etiqueta"])]

    curvas = [(fila["etiqueta"], fila["ainvs"]) for _, fila in muestra.iterrows()]
    df_gp = coeficientes_ap(curvas, n_primos=25)

    assert list(sub_cache.index) == list(df_gp.index)
    assert list(sub_cache.columns) == list(df_gp.columns)
    assert np.array_equal(sub_cache.to_numpy(), df_gp.to_numpy())


def test_simbolo_legendre_casos():
    assert simbolo_legendre(-1, 3) == -1
    assert simbolo_legendre(-1, 5) == 1
    assert simbolo_legendre(2, 7) == 1
    assert simbolo_legendre(3, 7) == -1
    assert simbolo_legendre(14, 7) == 0

