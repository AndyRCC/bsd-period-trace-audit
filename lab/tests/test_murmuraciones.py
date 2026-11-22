import numpy as np

from lab.datos import cargar_bsd, una_por_clase
from lab.experimentos.e04_murmuraciones import control_barajado
from lab.gp import coeficientes_ap


def test_conteos_7500_10000():
    clases = una_por_clase(cargar_bsd(n_max=10_000))
    cerrado = clases.loc[(clases["conductor"] >= 7_500) & (clases["conductor"] <= 10_000)]
    semiabierto = clases.loc[(clases["conductor"] >= 7_500) & (clases["conductor"] < 10_000)]

    conteos_cerrado = cerrado["rango"].value_counts().to_dict()
    conteos_semi = semiabierto["rango"].value_counts().to_dict()

    assert conteos_cerrado.get(0, 0) == 4_328
    assert conteos_cerrado.get(1, 0) == 5_194
    assert conteos_semi.get(0, 0) == 4_328
    assert conteos_semi.get(1, 0) == 5_194


def test_ultimo_primo():
    df_ap = coeficientes_ap([("11a1", (0, -1, 1, -10, -20))], n_primos=1000)
    assert len(df_ap.columns) == 1000
    assert int(df_ap.columns[-1]) == 7_919


def test_control_reproducible():
    rng = np.random.default_rng(12345)
    matriz_ap = rng.integers(-10, 11, size=(100, 50), dtype=np.int16)
    rangos = np.array([0] * 45 + [1] * 55, dtype=np.int8)

    s1 = control_barajado(matriz_ap, rangos, n_barajados=15, semilla=20261121)
    s2 = control_barajado(matriz_ap, rangos, n_barajados=15, semilla=20261121)

    assert s1.shape == (15,)
    assert np.array_equal(s1, s2)
