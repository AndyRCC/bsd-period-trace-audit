import numpy as np

from lab.data import load_bsd, one_per_class
from lab.experiments.e04_murmurations import control_shuffled
from lab.gp import ap_coefficients


def test_counts_7500_10000():
    classes = one_per_class(load_bsd(n_max=10_000))
    closed = classes.loc[(classes["conductor"] >= 7_500) & (classes["conductor"] <= 10_000)]
    half_open = classes.loc[(classes["conductor"] >= 7_500) & (classes["conductor"] < 10_000)]

    counts_closed = closed["rank"].value_counts().to_dict()
    counts_semi = half_open["rank"].value_counts().to_dict()

    assert counts_closed.get(0, 0) == 4_328
    assert counts_closed.get(1, 0) == 5_194
    assert counts_semi.get(0, 0) == 4_328
    assert counts_semi.get(1, 0) == 5_194


def test_last_prime():
    df_ap = ap_coefficients([("11a1", (0, -1, 1, -10, -20))], n_primes=1000)
    assert len(df_ap.columns) == 1000
    assert int(df_ap.columns[-1]) == 7_919


def test_control_reproducible():
    rng = np.random.default_rng(12345)
    matrix_ap = rng.integers(-10, 11, size=(100, 50), dtype=np.int16)
    ranks = np.array([0] * 45 + [1] * 55, dtype=np.int8)

    s1 = control_shuffled(matrix_ap, ranks, n_shuffled=15, seed=20261121)
    s2 = control_shuffled(matrix_ap, ranks, n_shuffled=15, seed=20261121)

    assert s1.shape == (15,)
    assert np.array_equal(s1, s2)
