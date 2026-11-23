import numpy as np

from lab.data import load_bsd, one_per_class
from lab.derived import (
    load_ap_cache,
    build_ap_cache,
    discriminant,
    legendre_symbol,
)
from lab.gp import ap_coefficients


def test_discriminant_reference():
    assert discriminant((0, -1, 1, -10, -20)) == -161051
    assert discriminant((0, 0, 1, -1, 0)) == 37
    assert discriminant((0, 1, 1, -2, 0)) == 389


def test_cache_full():
    path = build_ap_cache(n_primes=25)
    assert path.exists()
    df_cache = load_ap_cache(n_primes=25)
    assert df_cache.index.name == "label"
    assert df_cache.shape == (437_226, 25)
    assert int(df_cache.columns[0]) == 2
    assert int(df_cache.columns[-1]) == 97


def test_cache_vs_coefficients():
    classes = one_per_class(load_bsd(n_max=99_999)).reset_index(drop=True)
    rng = np.random.default_rng(20261121)
    idxs = np.sort(rng.choice(len(classes), size=20, replace=False))
    sample = classes.iloc[idxs]

    df_cache = load_ap_cache(n_primes=25)
    sub_cache = df_cache.loc[list(sample["label"])]

    curves = [(row["label"], row["ainvs"]) for _, row in sample.iterrows()]
    df_gp = ap_coefficients(curves, n_primes=25)

    assert list(sub_cache.index) == list(df_gp.index)
    assert list(sub_cache.columns) == list(df_gp.columns)
    assert np.array_equal(sub_cache.to_numpy(), df_gp.to_numpy())


def test_legendre_symbol_cases():
    assert legendre_symbol(-1, 3) == -1
    assert legendre_symbol(-1, 5) == 1
    assert legendre_symbol(2, 7) == 1
    assert legendre_symbol(3, 7) == -1
    assert legendre_symbol(14, 7) == 0

