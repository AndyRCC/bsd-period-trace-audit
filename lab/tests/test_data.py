from pathlib import Path
import numpy as np
import pytest

from lab.data import load_bsd, load_generators, one_per_class

ROOT = Path(__file__).resolve().parents[2]
CREMONA_DIR = ROOT / "data" / "cremona"

BSD_COLUMNS = [
    "label",
    "conductor",
    "isoclass",
    "number",
    "ainvs",
    "rank",
    "torsion",
    "tamagawa",
    "omega",
    "l_value",
    "regulator",
    "sha_an",
]


@pytest.fixture(scope="module")
def df_bsd():
    return load_bsd(n_max=99_999)


@pytest.fixture(scope="module")
def df_gens():
    return load_generators(n_max=99_999)


def test_11a1(df_bsd, df_gens):
    assert list(df_bsd.columns) == BSD_COLUMNS
    row = df_bsd.loc[df_bsd["label"] == "11a1"].iloc[0]
    assert row["conductor"] == 11
    assert row["isoclass"] == "a"
    assert row["number"] == 1
    assert row["ainvs"] == (0, -1, 1, -10, -20)
    assert row["rank"] == 0
    assert row["torsion"] == 5
    assert row["tamagawa"] == 5
    assert row["omega"] == pytest.approx(1.26920930427955, rel=1e-9)
    assert row["l_value"] == pytest.approx(0.253841860855911, rel=1e-9)
    assert row["regulator"] == pytest.approx(1.0, rel=1e-9)
    assert row["sha_an"] == pytest.approx(1.0, rel=1e-9)

    g = df_gens.loc[df_gens["label"] == "11a1"].iloc[0]
    assert g["gens"] == []


def test_37a1(df_bsd, df_gens):
    row = df_bsd.loc[df_bsd["label"] == "37a1"].iloc[0]
    assert row["conductor"] == 37
    assert row["isoclass"] == "a"
    assert row["number"] == 1
    assert row["ainvs"] == (0, 0, 1, -1, 0)
    assert row["rank"] == 1
    assert row["torsion"] == 1
    assert row["tamagawa"] == 1
    assert row["omega"] == pytest.approx(5.98691729246392, rel=1e-9)
    assert row["l_value"] == pytest.approx(0.305999773834052, rel=1e-9)
    assert row["regulator"] == pytest.approx(0.0511114082399688, rel=1e-9)
    assert row["sha_an"] == pytest.approx(1.0, rel=1e-9)

    g = df_gens.loc[df_gens["label"] == "37a1"].iloc[0]
    assert len(g["gens"]) == 1


def test_389a1(df_bsd, df_gens):
    row = df_bsd.loc[df_bsd["label"] == "389a1"].iloc[0]
    assert row["conductor"] == 389
    assert row["isoclass"] == "a"
    assert row["number"] == 1
    assert row["ainvs"] == (0, 1, 1, -2, 0)
    assert row["rank"] == 2
    assert row["torsion"] == 1
    assert row["tamagawa"] == 1
    assert row["omega"] == pytest.approx(4.98042512171011, rel=1e-9)
    assert row["l_value"] == pytest.approx(0.759316500288427, rel=1e-9)
    assert row["regulator"] == pytest.approx(0.152460177943144, rel=1e-9)
    assert row["sha_an"] == pytest.approx(1.0, rel=1e-9)

    g = df_gens.loc[df_gens["label"] == "389a1"].iloc[0]
    assert len(g["gens"]) == 2


def test_minimum_conductor(df_bsd):
    assert int(df_bsd["conductor"].min()) == 11
    assert int((df_bsd["conductor"] == 11).sum()) == 3


def test_counts_1_9999():
    df_9999 = load_bsd(n_max=9_999)
    assert len(df_9999) == 64_687
    classes = one_per_class(df_9999)
    assert len(classes) == 38_042
    by_rank = classes["rank"].value_counts().to_dict()
    assert by_rank.get(0, 0) == 16_450
    assert by_rank.get(1, 0) == 19_622
    assert by_rank.get(2, 0) == 1_969
    assert by_rank.get(3, 0) == 1


def test_counts_1_99999(df_bsd):
    assert len(df_bsd) == 657_396
    classes = one_per_class(df_bsd)
    assert len(classes) == 437_226


def test_column_identity(df_bsd):
    rhs = (
        df_bsd["omega"]
        * df_bsd["regulator"]
        * df_bsd["tamagawa"]
        * df_bsd["sha_an"]
        / (df_bsd["torsion"] ** 2)
    )
    rel_err = np.abs(df_bsd["l_value"] - rhs) / df_bsd["l_value"]
    assert bool((rel_err < 1e-5).all()), f"Maximum relative error: {rel_err.max()}"


def test_one_per_class(df_bsd):
    classes = one_per_class(df_bsd)
    assert len(classes) == 437_226
    assert bool((classes["number"] == 1).all())
    assert not bool(classes.duplicated(subset=["conductor", "isoclass"]).any())


def test_no_lost_lines(df_bsd, df_gens):
    files_bsd = sorted(CREMONA_DIR.glob("allbsd.*"))
    assert len(files_bsd) == 10
    lines_bsd = 0
    for path in files_bsd:
        with path.open("r", encoding="utf-8") as f:
            lines_bsd += sum(1 for line in f if line.strip())
    assert len(df_bsd) == lines_bsd

    files_gens = sorted(CREMONA_DIR.glob("allgens.*"))
    assert len(files_gens) == 10
    lines_gens = 0
    for path in files_gens:
        with path.open("r", encoding="utf-8") as f:
            lines_gens += sum(1 for line in f if line.strip())
    assert len(df_gens) == lines_gens


def test_generators_with_torsion(df_bsd, df_gens):
    row_bsd = df_bsd.loc[df_bsd["label"] == "65a2"].iloc[0]
    assert row_bsd["rank"] == 1
    assert row_bsd["torsion"] == 2
    row_gens = df_gens.loc[df_gens["label"] == "65a2"].iloc[0]
    assert row_gens["gens"] == ["[1:2:1]"]

