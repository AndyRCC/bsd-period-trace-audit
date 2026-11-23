import numpy as np
import pytest

from lab.data import load_bsd, load_generators
from lab.gp import (
    _projective_point_to_gp,
    ap_coefficients,
    run_gp,
    bad_euler_factor,
    pari_invariants,
    ap_averages,
    reduction_types,
)

INVARIANT_COLUMNS = [
    "label",
    "rank_an",
    "l_value",
    "omega",
    "tamagawa",
    "torsion",
    "regulator",
    "root_number",
]

REFERENCE_CURVES = [
    ("11a1", (0, -1, 1, -10, -20), []),
    ("37a1", (0, 0, 1, -1, 0), ["[0:0:1]"]),
    ("389a1", (0, 1, 1, -2, 0), ["[0:0:1]", "[1:0:1]"]),
]

EXPECTED_REFERENCE = {
    "11a1": {
        "rank_an": 0,
        "torsion": 5,
        "tamagawa": 5,
        "omega": 1.26920930427955,
        "l_value": 0.253841860855911,
        "regulator": 1.0,
        "root_number": 1,
    },
    "37a1": {
        "rank_an": 1,
        "torsion": 1,
        "tamagawa": 1,
        "omega": 5.98691729246392,
        "l_value": 0.305999773834052,
        "regulator": 0.0511114082399688,
        "root_number": -1,
    },
    "389a1": {
        "rank_an": 2,
        "torsion": 1,
        "tamagawa": 1,
        "omega": 4.98042512171011,
        "l_value": 0.759316500288427,
        "regulator": 0.152460177943144,
        "root_number": 1,
    },
}


@pytest.fixture(scope="module")
def df_invariants_ref():
    return pari_invariants(REFERENCE_CURVES)


@pytest.mark.parametrize("label", ["11a1", "37a1", "389a1"])
def test_invariants_reference(df_invariants_ref, label):
    assert list(df_invariants_ref.columns) == INVARIANT_COLUMNS
    assert len(df_invariants_ref) == 3
    row = df_invariants_ref.loc[df_invariants_ref["label"] == label].iloc[0]
    exp = EXPECTED_REFERENCE[label]
    assert int(row["rank_an"]) == exp["rank_an"]
    assert int(row["torsion"]) == exp["torsion"]
    assert int(row["tamagawa"]) == exp["tamagawa"]
    assert int(row["root_number"]) == exp["root_number"]
    assert float(row["omega"]) == pytest.approx(exp["omega"], rel=1e-8)
    assert float(row["l_value"]) == pytest.approx(exp["l_value"], rel=1e-8)
    assert float(row["regulator"]) == pytest.approx(exp["regulator"], rel=1e-8)


def test_5077a1_vs_table():
    df_bsd = load_bsd(n_max=9_999)
    df_gens = load_generators(n_max=9_999)
    row_bsd = df_bsd.loc[df_bsd["label"] == "5077a1"].iloc[0]
    row_gens = df_gens.loc[df_gens["label"] == "5077a1"].iloc[0]

    assert int(row_bsd["rank"]) == 3
    assert int(row_bsd["torsion"]) == 1
    assert int(row_bsd["tamagawa"]) == 1

    df_pari = pari_invariants([("5077a1", row_bsd["ainvs"], row_gens["gens"])])
    row_pari = df_pari.iloc[0]

    assert int(row_pari["rank_an"]) == 3
    assert int(row_pari["torsion"]) == 1
    assert int(row_pari["tamagawa"]) == 1
    assert int(row_pari["root_number"]) == -1
    assert float(row_pari["omega"]) == pytest.approx(float(row_bsd["omega"]), rel=1e-8)
    assert float(row_pari["l_value"]) == pytest.approx(float(row_bsd["l_value"]), rel=1e-8)
    assert float(row_pari["regulator"]) == pytest.approx(float(row_bsd["regulator"]), rel=1e-8)


def test_generators_on_curve():
    labels = ["37a1", "389a1", "5077a1", "65a2"]
    df_bsd = load_bsd(n_max=9_999)
    df_gens = load_generators(n_max=9_999)

    bsd_sub = df_bsd.loc[df_bsd["label"].isin(labels)].set_index("label")
    gens_sub = df_gens.loc[df_gens["label"].isin(labels)].set_index("label")

    lines_gp = []
    total_points = 0
    for tag in labels:
        ai = bsd_sub.loc[tag, "ainvs"]
        gens = gens_sub.loc[tag, "gens"]
        assert len(gens) == int(bsd_sub.loc[tag, "rank"])
        ai_gp = f"[{ai[0]},{ai[1]},{ai[2]},{ai[3]},{ai[4]}]"
        for p in gens:
            p_gp = _projective_point_to_gp(p)
            lines_gp.append(f"E = ellinit({ai_gp}); print(ellisoncurve(E, {p_gp}));")
            total_points += 1

    output = run_gp("\n".join(lines_gp))
    results = [int(x.strip()) for x in output.splitlines() if x.strip()]
    assert len(results) == total_points
    assert all(r == 1 for r in results)


def test_ap_11a1():
    df_ap = ap_coefficients([("11a1", (0, -1, 1, -10, -20))], n_primes=1000)
    assert df_ap.index.name == "label"
    assert df_ap.shape == (1, 1000)
    assert int(df_ap.loc["11a1", 2]) == -2
    assert int(df_ap.loc["11a1", 3]) == -1
    assert int(df_ap.loc["11a1", 5]) == 1
    assert int(df_ap.loc["11a1", 7]) == -2
    assert int(df_ap.loc["11a1", 11]) == 1
    assert int(df_ap.loc["11a1", 13]) == 4


def test_gp_timeout():
    with pytest.raises(TimeoutError):
        run_gp("while(1, )", timeout_s=2)


def test_gp_error():
    with pytest.raises(RuntimeError):
        run_gp("1/0;")


def test_gp_path_with_space(tmp_path, monkeypatch):
    dir_with_space = tmp_path / "directory with space"
    dir_with_space.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(dir_with_space)
    output = run_gp("print(2 + 2);")
    assert output.strip() == "4"


def test_ap_averages_vs_coefficients():
    df_bsd = load_bsd(n_max=200)
    sample = df_bsd.iloc[:5]
    groups = [0, 0, 1, 1, 0]
    curves_avg = [
        (row["label"], row["ainvs"], g)
        for (_, row), g in zip(sample.iterrows(), groups)
    ]
    curves_coef = [
        (row["label"], row["ainvs"]) for _, row in sample.iterrows()
    ]

    df_avg = ap_averages(curves_avg, p_max=100, exclude_bad=False)
    assert list(df_avg.columns) == ["group", "p", "n", "acc", "mean"]

    df_coef = ap_coefficients(curves_coef, n_primes=25)
    assert list(df_coef.columns)[-1] == 97

    for g in (0, 1):
        idxs = [i for i, val in enumerate(groups) if val == g]
        sub_coef = df_coef.iloc[idxs]
        sub_avg = df_avg.loc[df_avg["group"] == g].set_index("p")
        assert len(sub_avg) == 25
        for p in df_coef.columns:
            assert int(sub_avg.loc[p, "n"]) == len(idxs)
            assert int(sub_avg.loc[p, "acc"]) == int(sub_coef[p].sum())
            assert float(sub_avg.loc[p, "mean"]) == pytest.approx(
                float(sub_coef[p].mean()), rel=1e-12, abs=1e-12
            )


def test_ap_averages_exclude_bad():
    curves = [("11a1", (0, -1, 1, -10, -20), 0)]
    df_avg = ap_averages(curves, p_max=20, exclude_bad=True)
    by_p = df_avg.set_index("p")
    assert int(by_p.loc[11, "n"]) == 0
    assert int(by_p.loc[11, "acc"]) == 0
    assert np.isnan(float(by_p.loc[11, "mean"]))
    for p in (2, 3, 5, 7, 13, 17, 19):
        assert int(by_p.loc[p, "n"]) == 1


def test_bad_euler_factor_reference():
    curves = [
        ("11a1", (0, -1, 1, -10, -20)),
        ("37a1", (0, 0, 1, -1, 0)),
        ("389a1", (0, 1, 1, -2, 0)),
    ]
    df_b = bad_euler_factor(curves)
    assert list(df_b.columns) == ["label", "log_B", "n_bad"]
    by_tag = df_b.set_index("label")
    assert float(by_tag.loc["11a1", "log_B"]) == pytest.approx(
        np.log(11.0 / 10.0), rel=1e-7
    )
    assert float(by_tag.loc["37a1", "log_B"]) == pytest.approx(
        np.log(37.0 / 38.0), rel=1e-7
    )
    assert float(by_tag.loc["389a1", "log_B"]) == pytest.approx(
        np.log(389.0 / 388.0), rel=1e-7
    )
    assert int(by_tag.loc["11a1", "n_bad"]) == 1
    assert int(by_tag.loc["37a1", "n_bad"]) == 1
    assert int(by_tag.loc["389a1", "n_bad"]) == 1


def test_reduction_types_reference():
    curves = [
        ("11a1", (0, -1, 1, -10, -20)),
        ("37a1", (0, 0, 1, -1, 0)),
        ("389a1", (0, 1, 1, -2, 0)),
        ("27a1", (0, 0, 1, 0, -7)),
    ]
    df_tr = reduction_types(curves)
    assert list(df_tr.columns) == [
        "label",
        "n_split",
        "n_no_split",
        "n_additive",
    ]
    by_tag = df_tr.set_index("label")
    expected = {
        "11a1": (1, 0, 0),
        "37a1": (0, 1, 0),
        "389a1": (1, 0, 0),
        "27a1": (0, 0, 1),
    }
    for tag, (np_exp, nnp_exp, na_exp) in expected.items():
        assert int(by_tag.loc[tag, "n_split"]) == np_exp
        assert int(by_tag.loc[tag, "n_no_split"]) == nnp_exp
        assert int(by_tag.loc[tag, "n_additive"]) == na_exp



