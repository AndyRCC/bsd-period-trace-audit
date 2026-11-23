import numpy as np
import pytest

from lab.stats import (
    stratified_bins,
    stratum_key,
    partial_correlation_bins,
    partial_correlation_cov,
    covariance_in_bins,
    profile_difference,
    permute_in_bins,
    partial_residuals_cov,
)


def test_permute_in_bins_preserves_counts():
    rng = np.random.default_rng(20261121)
    bins = np.repeat(np.arange(5), 40)
    labels = np.array([(i % 3 == 0) for i in range(200)], dtype=bool)

    perm = permute_in_bins(labels, bins, rng)
    assert perm.shape == labels.shape
    assert not np.array_equal(perm, labels)

    for b in range(5):
        mask = bins == b
        assert int(np.sum(perm[mask])) == int(np.sum(labels[mask]))
        assert int(np.sum(~perm[mask])) == int(np.sum(~labels[mask]))


def test_profile_difference_manual():
    # 4x2 matrix: rows 0,1 in group True; rows 2,3 in group False
    # Cell (0, 0) is not good (p | N)
    ap = np.array(
        [
            [99, 2],
            [4, 6],
            [1, 3],
            [3, 5],
        ],
        dtype=np.int16,
    )
    good = np.array(
        [
            [False, True],
            [True, True],
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    group = np.array([True, True, False, False], dtype=bool)

    # Column 0: True has only [4] -> mean 4.0; False has [1, 3] -> mean 2.0 => delta[0] = 2.0
    # Column 1: True has [2, 6] -> mean 4.0; False has [3, 5] -> mean 4.0 => delta[1] = 0.0
    # RMS = sqrt((2.0^2 + 0.0^2) / 2) = sqrt(2)
    delta, rms = profile_difference(ap, good, group)
    assert delta.shape == (2,)
    assert float(delta[0]) == pytest.approx(2.0, rel=1e-12, abs=1e-12)
    assert float(delta[1]) == pytest.approx(0.0, rel=1e-12, abs=1e-12)
    assert float(rms) == pytest.approx(np.sqrt(2.0), rel=1e-12, abs=1e-12)


def test_partial_correlation_removes_bin_effect():
    rng = np.random.default_rng(20261121)
    n = 20_000
    bins = rng.integers(0, 50, size=n)
    x = 10.0 * bins + rng.normal(0.0, 1.0, size=n)
    y = -5.0 * bins + rng.normal(0.0, 1.0, size=n)

    rho, n_ef = partial_correlation_bins(x, y, bins)
    assert abs(rho) < 0.02
    assert n_ef == n - 50 - 1


def test_partial_correlation_manual_case():
    x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    y = np.array([2.0, 4.0, 6.0, 30.0, 29.0, 28.0])
    bins = np.array([0, 0, 0, 1, 1, 1])

    rho, n_ef = partial_correlation_bins(x, y, bins)
    assert float(rho) == pytest.approx(1.0 / np.sqrt(10.0), rel=1e-9, abs=1e-9)
    assert int(n_ef) == 3


def test_partial_correlation_cov_removes_covariate():
    rng = np.random.default_rng(20261121)
    n = 20_000
    bins = rng.integers(0, 20, size=n)
    z = rng.normal(0.0, 1.0, size=n)
    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = rng.normal(0.0, 1.0, size=n)
    x = z + e1
    y = z + e2

    rho_uncond, _ = partial_correlation_bins(x, y, bins)
    assert rho_uncond > 0.4

    rho_with, n_ef = partial_correlation_cov(x, y, bins, z)
    assert abs(rho_with) < 0.02
    assert n_ef == n - 20 - 1 - 1


def test_decomposition_identity():
    rng = np.random.default_rng(20261121)
    n = 5_000
    bins = rng.integers(0, 25, size=n)
    x = rng.normal(0.0, 2.0, size=n)
    u = rng.normal(1.0, 1.5, size=n)
    v = rng.normal(-1.0, 0.8, size=n)
    t = rng.normal(0.5, 1.2, size=n)
    w = u - v + 2.0 * t

    cov_xw = covariance_in_bins(x, w, bins)
    cov_xu = covariance_in_bins(x, u, bins)
    cov_xv = covariance_in_bins(x, v, bins)
    cov_xt = covariance_in_bins(x, t, bins)

    rhs = cov_xu - cov_xv + 2.0 * cov_xt
    assert abs(cov_xw - rhs) < 1e-12


def test_partial_correlation_cov_two_covariates():
    rng = np.random.default_rng(20261121)
    n = 20_000
    bins = rng.integers(0, 20, size=n)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = rng.normal(0.0, 1.0, size=n)
    x = z1 + z2 + e1
    y = z1 + z2 + e2

    rho_z1, n_ef1 = partial_correlation_cov(x, y, bins, z1)
    assert rho_z1 > 0.2
    assert n_ef1 == n - 20 - 1 - 1

    z12 = np.column_stack([z1, z2])
    rho_z12, n_ef12 = partial_correlation_cov(x, y, bins, z12)
    assert abs(rho_z12) < 0.02
    assert n_ef12 == n - 20 - 2 - 1


def test_partial_residuals_cov_free_direction_identity():
    rng = np.random.default_rng(20261121)
    n = 5_000
    bins = rng.integers(0, 25, size=n)
    x = rng.normal(0.0, 1.0, size=n)
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    u = 0.4 * x + rng.normal(0.0, 1.0, size=n)
    # If u + v = f(bins) - z1 + 2*z2 exactly, then (u - v) projected on (bins, z1, z2) is 2*u projected
    v = (0.5 * bins - z1 + 2.0 * z2) - u
    Z = np.column_stack([z1, z2])

    ru = partial_residuals_cov(u, bins, Z)
    ruv = partial_residuals_cov(u - v, bins, Z)
    assert np.allclose(ruv, 2.0 * ru, atol=1e-12)

    rho_u, _ = partial_correlation_cov(x, u, bins, Z)
    rho_uv, _ = partial_correlation_cov(x, u - v, bins, Z)
    assert abs(rho_u - rho_uv) < 1e-12


def test_stratified_bins():
    rng = np.random.default_rng(20261121)
    L = rng.uniform(0.1, 10.0, size=1400)
    stratum = np.array([0] * 1000 + [1] * 300 + [2] * 100, dtype=int)

    bins = stratified_bins(L, stratum, q_max=10, n_min=200)
    assert bins.shape == (1400,)

    # Stratum 2 (100 elements < n_min=200) stays at -1
    assert np.all(bins[stratum == 2] == -1)

    # Stratum 0 (1000 elements) has 10 bins; stratum 1 (300 elements) has 300//50 = 6 bins
    b0 = bins[stratum == 0]
    b1 = bins[stratum == 1]
    assert np.all(b0 >= 0)
    assert np.all(b1 >= 0)
    u0 = set(np.unique(b0))
    u1 = set(np.unique(b1))
    assert len(u0) == 10
    assert len(u1) == 6
    # Ids are not repeated across strata
    assert len(u0.intersection(u1)) == 0


def test_stratum_key():
    cp = np.array([1, 1, 6, 7, 4, 5, 9, 12], dtype=int)
    T = np.array([1, 1, 2, 2, 1, 1, 4, 6], dtype=int)
    sha = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 9.0], dtype=float)

    keys = stratum_key(cp, T, sha)
    assert keys.shape == (8,)
    assert np.issubdtype(keys.dtype, np.integer)
    # Two rows with the same values give the same key
    assert keys[0] == keys[1]
    # cp = 6 and cp = 7 fall in the same category (5-8)
    assert keys[2] == keys[3]
    # cp = 4 and cp = 5 do NOT fall in the same category
    assert keys[4] != keys[5]
    # cp >= 9, T >= 4, Sha >= 4 fall in the same category
    assert keys[6] == keys[7]
    # Different categories give different keys
    assert len({int(keys[0]), int(keys[2]), int(keys[4]), int(keys[5]), int(keys[6])}) == 5

    # With extra: a different extra gives a different key; the same extra gives the same key
    extra = np.array([0, 1, 2, 2, 0, 0, 3, 4], dtype=int)
    keys_ex = stratum_key(cp, T, sha, extra=extra)
    assert keys_ex.shape == (8,)
    assert np.issubdtype(keys_ex.dtype, np.integer)
    assert keys_ex[0] != keys_ex[1]
    assert keys_ex[2] == keys_ex[3]
    assert keys_ex[4] != keys_ex[5]
    assert keys_ex[6] != keys_ex[7]





