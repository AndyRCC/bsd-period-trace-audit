from pathlib import Path
import numpy as np
import pytest

from lab.datos import cargar_bsd, cargar_generadores, una_por_clase

RAIZ = Path(__file__).resolve().parents[2]
DIR_CREMONA = RAIZ / "datos" / "cremona"

COLUMNAS_BSD = [
    "etiqueta",
    "conductor",
    "clase",
    "numero",
    "ainvs",
    "rango",
    "torsion",
    "tamagawa",
    "omega",
    "l_valor",
    "regulador",
    "sha_an",
]


@pytest.fixture(scope="module")
def df_bsd():
    return cargar_bsd(n_max=99_999)


@pytest.fixture(scope="module")
def df_gens():
    return cargar_generadores(n_max=99_999)


def test_11a1(df_bsd, df_gens):
    assert list(df_bsd.columns) == COLUMNAS_BSD
    fila = df_bsd.loc[df_bsd["etiqueta"] == "11a1"].iloc[0]
    assert fila["conductor"] == 11
    assert fila["clase"] == "a"
    assert fila["numero"] == 1
    assert fila["ainvs"] == (0, -1, 1, -10, -20)
    assert fila["rango"] == 0
    assert fila["torsion"] == 5
    assert fila["tamagawa"] == 5
    assert fila["omega"] == pytest.approx(1.26920930427955, rel=1e-9)
    assert fila["l_valor"] == pytest.approx(0.253841860855911, rel=1e-9)
    assert fila["regulador"] == pytest.approx(1.0, rel=1e-9)
    assert fila["sha_an"] == pytest.approx(1.0, rel=1e-9)

    g = df_gens.loc[df_gens["etiqueta"] == "11a1"].iloc[0]
    assert g["gens"] == []


def test_37a1(df_bsd, df_gens):
    fila = df_bsd.loc[df_bsd["etiqueta"] == "37a1"].iloc[0]
    assert fila["conductor"] == 37
    assert fila["clase"] == "a"
    assert fila["numero"] == 1
    assert fila["ainvs"] == (0, 0, 1, -1, 0)
    assert fila["rango"] == 1
    assert fila["torsion"] == 1
    assert fila["tamagawa"] == 1
    assert fila["omega"] == pytest.approx(5.98691729246392, rel=1e-9)
    assert fila["l_valor"] == pytest.approx(0.305999773834052, rel=1e-9)
    assert fila["regulador"] == pytest.approx(0.0511114082399688, rel=1e-9)
    assert fila["sha_an"] == pytest.approx(1.0, rel=1e-9)

    g = df_gens.loc[df_gens["etiqueta"] == "37a1"].iloc[0]
    assert len(g["gens"]) == 1


def test_389a1(df_bsd, df_gens):
    fila = df_bsd.loc[df_bsd["etiqueta"] == "389a1"].iloc[0]
    assert fila["conductor"] == 389
    assert fila["clase"] == "a"
    assert fila["numero"] == 1
    assert fila["ainvs"] == (0, 1, 1, -2, 0)
    assert fila["rango"] == 2
    assert fila["torsion"] == 1
    assert fila["tamagawa"] == 1
    assert fila["omega"] == pytest.approx(4.98042512171011, rel=1e-9)
    assert fila["l_valor"] == pytest.approx(0.759316500288427, rel=1e-9)
    assert fila["regulador"] == pytest.approx(0.152460177943144, rel=1e-9)
    assert fila["sha_an"] == pytest.approx(1.0, rel=1e-9)

    g = df_gens.loc[df_gens["etiqueta"] == "389a1"].iloc[0]
    assert len(g["gens"]) == 2


def test_conductor_minimo(df_bsd):
    assert int(df_bsd["conductor"].min()) == 11
    assert int((df_bsd["conductor"] == 11).sum()) == 3


def test_conteos_1_9999():
    df_9999 = cargar_bsd(n_max=9_999)
    assert len(df_9999) == 64_687
    clases = una_por_clase(df_9999)
    assert len(clases) == 38_042
    por_rango = clases["rango"].value_counts().to_dict()
    assert por_rango.get(0, 0) == 16_450
    assert por_rango.get(1, 0) == 19_622
    assert por_rango.get(2, 0) == 1_969
    assert por_rango.get(3, 0) == 1


def test_conteos_1_99999(df_bsd):
    assert len(df_bsd) == 657_396
    clases = una_por_clase(df_bsd)
    assert len(clases) == 437_226


def test_identidad_columnas(df_bsd):
    rhs = (
        df_bsd["omega"]
        * df_bsd["regulador"]
        * df_bsd["tamagawa"]
        * df_bsd["sha_an"]
        / (df_bsd["torsion"] ** 2)
    )
    rel_err = np.abs(df_bsd["l_valor"] - rhs) / df_bsd["l_valor"]
    assert bool((rel_err < 1e-5).all()), f"Máximo error relativo: {rel_err.max()}"


def test_una_por_clase(df_bsd):
    clases = una_por_clase(df_bsd)
    assert len(clases) == 437_226
    assert bool((clases["numero"] == 1).all())
    assert not bool(clases.duplicated(subset=["conductor", "clase"]).any())


def test_sin_lineas_perdidas(df_bsd, df_gens):
    archivos_bsd = sorted(DIR_CREMONA.glob("allbsd.*"))
    assert len(archivos_bsd) == 10
    lineas_bsd = 0
    for ruta in archivos_bsd:
        with ruta.open("r", encoding="utf-8") as f:
            lineas_bsd += sum(1 for linea in f if linea.strip())
    assert len(df_bsd) == lineas_bsd

    archivos_gens = sorted(DIR_CREMONA.glob("allgens.*"))
    assert len(archivos_gens) == 10
    lineas_gens = 0
    for ruta in archivos_gens:
        with ruta.open("r", encoding="utf-8") as f:
            lineas_gens += sum(1 for linea in f if linea.strip())
    assert len(df_gens) == lineas_gens


def test_generadores_con_torsion(df_bsd, df_gens):
    fila_bsd = df_bsd.loc[df_bsd["etiqueta"] == "65a2"].iloc[0]
    assert fila_bsd["rango"] == 1
    assert fila_bsd["torsion"] == 2
    fila_gens = df_gens.loc[df_gens["etiqueta"] == "65a2"].iloc[0]
    assert fila_gens["gens"] == ["[1:2:1]"]

