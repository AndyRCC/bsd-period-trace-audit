import numpy as np
import pytest

from lab.estadistica import (
    bins_estratificados,
    clave_estrato,
    correlacion_parcial_bins,
    correlacion_parcial_cov,
    covarianza_en_bins,
    diferencia_perfiles,
    permutar_en_bins,
    residuos_parciales_cov,
)


def test_permutar_en_bins_conserva_conteos():
    rng = np.random.default_rng(20261121)
    bins = np.repeat(np.arange(5), 40)
    etiquetas = np.array([(i % 3 == 0) for i in range(200)], dtype=bool)

    perm = permutar_en_bins(etiquetas, bins, rng)
    assert perm.shape == etiquetas.shape
    assert not np.array_equal(perm, etiquetas)

    for b in range(5):
        mask = bins == b
        assert int(np.sum(perm[mask])) == int(np.sum(etiquetas[mask]))
        assert int(np.sum(~perm[mask])) == int(np.sum(~etiquetas[mask]))


def test_diferencia_perfiles_manual():
    # Matriz 4x2: filas 0,1 en grupo True; filas 2,3 en grupo False
    # Celda (0, 0) no es buena (p | N)
    ap = np.array(
        [
            [99, 2],
            [4, 6],
            [1, 3],
            [3, 5],
        ],
        dtype=np.int16,
    )
    buenos = np.array(
        [
            [False, True],
            [True, True],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    grupo = np.array([True, True, False, False], dtype=bool)

    # Columna 0: True tiene solo [4] -> media 4.0; False tiene [1, 3] -> media 2.0 => delta[0] = 2.0
    # Columna 1: True tiene [2, 6] -> media 4.0; False tiene [3, 5] -> media 4.0 => delta[1] = 0.0
    # RMS = sqrt((2.0^2 + 0.0^2) / 2) = sqrt(2)
    delta, rms = diferencia_perfiles(ap, buenos, grupo)
    assert delta.shape == (2,)
    assert float(delta[0]) == pytest.approx(2.0, rel=1e-12, abs=1e-12)
    assert float(delta[1]) == pytest.approx(0.0, rel=1e-12, abs=1e-12)
    assert float(rms) == pytest.approx(np.sqrt(2.0), rel=1e-12, abs=1e-12)


def test_correlacion_parcial_elimina_efecto_de_bin():
    rng = np.random.default_rng(20261121)
    n = 20_000
    bins = rng.integers(0, 50, size=n)
    x = 10.0 * bins + rng.normal(0.0, 1.0, size=n)
    y = -5.0 * bins + rng.normal(0.0, 1.0, size=n)

    rho, n_ef = correlacion_parcial_bins(x, y, bins)
    assert abs(rho) < 0.02
    assert n_ef == n - 50 - 1


def test_correlacion_parcial_caso_manual():
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([2.0, 4.0, 6.0, 30.0, 29.0, 28.0])
    bins = np.array([0, 0, 0, 1, 1, 1])

    rho, n_ef = correlacion_parcial_bins(x, y, bins)
    assert float(rho) == pytest.approx(1.0 / np.sqrt(10.0), rel=1e-9, abs=1e-9)
    assert int(n_ef) == 3


def test_correlacion_parcial_cov_quita_covariable():
    rng = np.random.default_rng(20261121)
    n = 20_000
    bins = rng.integers(0, 20, size=n)
    z = rng.normal(0.0, 1.0, size=n)
    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = rng.normal(0.0, 1.0, size=n)
    x = z + e1
    y = z + e2

    rho_sin, _ = correlacion_parcial_bins(x, y, bins)
    assert rho_sin > 0.4

    rho_con, n_ef = correlacion_parcial_cov(x, y, bins, z)
    assert abs(rho_con) < 0.02
    assert n_ef == n - 20 - 1 - 1


def test_identidad_descomposicion():
    rng = np.random.default_rng(20261121)
    n = 5_000
    bins = rng.integers(0, 25, size=n)
    x = rng.normal(0.0, 2.0, size=n)
    u = rng.normal(1.0, 1.5, size=n)
    v = rng.normal(-1.0, 0.8, size=n)
    t = rng.normal(0.5, 1.2, size=n)
    w = u - v + 2.0 * t

    cov_xw = covarianza_en_bins(x, w, bins)
    cov_xu = covarianza_en_bins(x, u, bins)
    cov_xv = covarianza_en_bins(x, v, bins)
    cov_xt = covarianza_en_bins(x, t, bins)

    rhs = cov_xu - cov_xv + 2.0 * cov_xt
    assert abs(cov_xw - rhs) < 1e-12


def test_correlacion_parcial_cov_dos_covariables():
    rng = np.random.default_rng(20261121)
    n = 20_000
    bins = rng.integers(0, 20, size=n)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = rng.normal(0.0, 1.0, size=n)
    x = z1 + z2 + e1
    y = z1 + z2 + e2

    rho_z1, n_ef1 = correlacion_parcial_cov(x, y, bins, z1)
    assert rho_z1 > 0.2
    assert n_ef1 == n - 20 - 1 - 1

    z12 = np.column_stack([z1, z2])
    rho_z12, n_ef12 = correlacion_parcial_cov(x, y, bins, z12)
    assert abs(rho_z12) < 0.02
    assert n_ef12 == n - 20 - 2 - 1


def test_residuos_parciales_cov_identidad_direccion_libre():
    rng = np.random.default_rng(20261121)
    n = 5_000
    bins = rng.integers(0, 25, size=n)
    x = rng.normal(0.0, 1.0, size=n)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    u = 0.4 * x + rng.normal(0.0, 1.0, size=n)
    # Si u + v = f(bins) - z1 + 2*z2 exacto, entonces (u - v) proyectado sobre (bins, z1, z2) es 2*u proyectado
    v = (0.5 * bins - z1 + 2.0 * z2) - u
    Z = np.column_stack([z1, z2])

    ru = residuos_parciales_cov(u, bins, Z)
    ruv = residuos_parciales_cov(u - v, bins, Z)
    assert np.allclose(ruv, 2.0 * ru, atol=1e-12)

    rho_u, _ = correlacion_parcial_cov(x, u, bins, Z)
    rho_uv, _ = correlacion_parcial_cov(x, u - v, bins, Z)
    assert abs(rho_u - rho_uv) < 1e-12


def test_bins_estratificados():
    rng = np.random.default_rng(20261121)
    L = rng.uniform(0.1, 10.0, size=1400)
    estrato = np.array([0] * 1000 + [1] * 300 + [2] * 100, dtype=int)

    bins = bins_estratificados(L, estrato, q_max=10, n_min=200)
    assert bins.shape == (1400,)

    # Estrato 2 (100 elementos < n_min=200) queda en -1
    assert np.all(bins[estrato == 2] == -1)

    # Estrato 0 (1000 elementos) tiene 10 bins; estrato 1 (300 elementos) tiene 300//50 = 6 bins
    b0 = bins[estrato == 0]
    b1 = bins[estrato == 1]
    assert np.all(b0 >= 0)
    assert np.all(b1 >= 0)
    u0 = set(np.unique(b0))
    u1 = set(np.unique(b1))
    assert len(u0) == 10
    assert len(u1) == 6
    # Los ids no se repiten entre estratos
    assert len(u0.intersection(u1)) == 0


def test_clave_estrato():
    cp = np.array([1, 1, 6, 7, 4, 5, 9, 12], dtype=int)
    T = np.array([1, 1, 2, 2, 1, 1, 4, 6], dtype=int)
    sha = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 9.0], dtype=float)

    claves = clave_estrato(cp, T, sha)
    assert claves.shape == (8,)
    assert np.issubdtype(claves.dtype, np.integer)
    # Dos filas con los mismos valores dan la misma clave
    assert claves[0] == claves[1]
    # cp = 6 y cp = 7 caen en la misma categoría (5-8)
    assert claves[2] == claves[3]
    # cp = 4 y cp = 5 NO caen en la misma categoría
    assert claves[4] != claves[5]
    # cp >= 9, T >= 4, Sha >= 4 caen en la misma categoría
    assert claves[6] == claves[7]
    # Categorías distintas dan claves distintas
    assert len({int(claves[0]), int(claves[2]), int(claves[4]), int(claves[5]), int(claves[6])}) == 5

    # Con extra: extra distinto da clave distinta; mismo extra da misma clave
    extra = np.array([0, 1, 2, 2, 0, 0, 3, 4], dtype=int)
    claves_ex = clave_estrato(cp, T, sha, extra=extra)
    assert claves_ex.shape == (8,)
    assert np.issubdtype(claves_ex.dtype, np.integer)
    assert claves_ex[0] != claves_ex[1]
    assert claves_ex[2] == claves_ex[3]
    assert claves_ex[4] != claves_ex[5]
    assert claves_ex[6] != claves_ex[7]





