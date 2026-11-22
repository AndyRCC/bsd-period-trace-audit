import numpy as np
import pytest

from lab.datos import cargar_bsd, cargar_generadores
from lab.gp import (
    _punto_proyectivo_a_gp,
    coeficientes_ap,
    correr_gp,
    factor_euler_malos,
    invariantes_pari,
    promedios_ap,
    tipos_reduccion,
)

COLUMNAS_INVARIANTES = [
    "etiqueta",
    "rango_an",
    "l_valor",
    "omega",
    "tamagawa",
    "torsion",
    "regulador",
    "raiz",
]

CURVAS_REFERENCIA = [
    ("11a1", (0, -1, 1, -10, -20), []),
    ("37a1", (0, 0, 1, -1, 0), ["[0:0:1]"]),
    ("389a1", (0, 1, 1, -2, 0), ["[0:0:1]", "[1:0:1]"]),
]

ESPERADOS_REFERENCIA = {
    "11a1": {
        "rango_an": 0,
        "torsion": 5,
        "tamagawa": 5,
        "omega": 1.26920930427955,
        "l_valor": 0.253841860855911,
        "regulador": 1.0,
        "raiz": 1,
    },
    "37a1": {
        "rango_an": 1,
        "torsion": 1,
        "tamagawa": 1,
        "omega": 5.98691729246392,
        "l_valor": 0.305999773834052,
        "regulador": 0.0511114082399688,
        "raiz": -1,
    },
    "389a1": {
        "rango_an": 2,
        "torsion": 1,
        "tamagawa": 1,
        "omega": 4.98042512171011,
        "l_valor": 0.759316500288427,
        "regulador": 0.152460177943144,
        "raiz": 1,
    },
}


@pytest.fixture(scope="module")
def df_invariantes_ref():
    return invariantes_pari(CURVAS_REFERENCIA)


@pytest.mark.parametrize("etiqueta", ["11a1", "37a1", "389a1"])
def test_invariantes_referencia(df_invariantes_ref, etiqueta):
    assert list(df_invariantes_ref.columns) == COLUMNAS_INVARIANTES
    assert len(df_invariantes_ref) == 3
    fila = df_invariantes_ref.loc[df_invariantes_ref["etiqueta"] == etiqueta].iloc[0]
    esp = ESPERADOS_REFERENCIA[etiqueta]
    assert int(fila["rango_an"]) == esp["rango_an"]
    assert int(fila["torsion"]) == esp["torsion"]
    assert int(fila["tamagawa"]) == esp["tamagawa"]
    assert int(fila["raiz"]) == esp["raiz"]
    assert float(fila["omega"]) == pytest.approx(esp["omega"], rel=1e-8)
    assert float(fila["l_valor"]) == pytest.approx(esp["l_valor"], rel=1e-8)
    assert float(fila["regulador"]) == pytest.approx(esp["regulador"], rel=1e-8)


def test_5077a1_contra_tabla():
    df_bsd = cargar_bsd(n_max=9_999)
    df_gens = cargar_generadores(n_max=9_999)
    fila_bsd = df_bsd.loc[df_bsd["etiqueta"] == "5077a1"].iloc[0]
    fila_gens = df_gens.loc[df_gens["etiqueta"] == "5077a1"].iloc[0]

    assert int(fila_bsd["rango"]) == 3
    assert int(fila_bsd["torsion"]) == 1
    assert int(fila_bsd["tamagawa"]) == 1

    df_pari = invariantes_pari([("5077a1", fila_bsd["ainvs"], fila_gens["gens"])])
    fila_pari = df_pari.iloc[0]

    assert int(fila_pari["rango_an"]) == 3
    assert int(fila_pari["torsion"]) == 1
    assert int(fila_pari["tamagawa"]) == 1
    assert int(fila_pari["raiz"]) == -1
    assert float(fila_pari["omega"]) == pytest.approx(float(fila_bsd["omega"]), rel=1e-8)
    assert float(fila_pari["l_valor"]) == pytest.approx(float(fila_bsd["l_valor"]), rel=1e-8)
    assert float(fila_pari["regulador"]) == pytest.approx(float(fila_bsd["regulador"]), rel=1e-8)


def test_generadores_en_curva():
    etiquetas = ["37a1", "389a1", "5077a1", "65a2"]
    df_bsd = cargar_bsd(n_max=9_999)
    df_gens = cargar_generadores(n_max=9_999)

    bsd_sub = df_bsd.loc[df_bsd["etiqueta"].isin(etiquetas)].set_index("etiqueta")
    gens_sub = df_gens.loc[df_gens["etiqueta"].isin(etiquetas)].set_index("etiqueta")

    lineas_gp = []
    total_puntos = 0
    for et in etiquetas:
        ai = bsd_sub.loc[et, "ainvs"]
        gens = gens_sub.loc[et, "gens"]
        assert len(gens) == int(bsd_sub.loc[et, "rango"])
        ai_gp = f"[{ai[0]},{ai[1]},{ai[2]},{ai[3]},{ai[4]}]"
        for p in gens:
            p_gp = _punto_proyectivo_a_gp(p)
            lineas_gp.append(f"E = ellinit({ai_gp}); print(ellisoncurve(E, {p_gp}));")
            total_puntos += 1

    salida = correr_gp("\n".join(lineas_gp))
    resultados = [int(x.strip()) for x in salida.splitlines() if x.strip()]
    assert len(resultados) == total_puntos
    assert all(r == 1 for r in resultados)


def test_ap_11a1():
    df_ap = coeficientes_ap([("11a1", (0, -1, 1, -10, -20))], n_primos=1000)
    assert df_ap.index.name == "etiqueta"
    assert df_ap.shape == (1, 1000)
    assert int(df_ap.loc["11a1", 2]) == -2
    assert int(df_ap.loc["11a1", 3]) == -1
    assert int(df_ap.loc["11a1", 5]) == 1
    assert int(df_ap.loc["11a1", 7]) == -2
    assert int(df_ap.loc["11a1", 11]) == 1
    assert int(df_ap.loc["11a1", 13]) == 4


def test_gp_timeout():
    with pytest.raises(TimeoutError):
        correr_gp("while(1, )", timeout_s=2)


def test_gp_error():
    with pytest.raises(RuntimeError):
        correr_gp("1/0;")


def test_gp_ruta_con_espacio(tmp_path, monkeypatch):
    dir_con_espacio = tmp_path / "directorio con espacio"
    dir_con_espacio.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(dir_con_espacio)
    salida = correr_gp("print(2 + 2);")
    assert salida.strip() == "4"


def test_promedios_ap_contra_coeficientes():
    df_bsd = cargar_bsd(n_max=200)
    muestra = df_bsd.iloc[:5]
    grupos = [0, 0, 1, 1, 0]
    curvas_prom = [
        (fila["etiqueta"], fila["ainvs"], g)
        for (_, fila), g in zip(muestra.iterrows(), grupos)
    ]
    curvas_coef = [
        (fila["etiqueta"], fila["ainvs"]) for _, fila in muestra.iterrows()
    ]

    df_prom = promedios_ap(curvas_prom, p_max=100, excluir_malos=False)
    assert list(df_prom.columns) == ["grupo", "p", "n", "suma", "media"]

    df_coef = coeficientes_ap(curvas_coef, n_primos=25)
    assert list(df_coef.columns)[-1] == 97

    for g in (0, 1):
        idxs = [i for i, val in enumerate(grupos) if val == g]
        sub_coef = df_coef.iloc[idxs]
        sub_prom = df_prom.loc[df_prom["grupo"] == g].set_index("p")
        assert len(sub_prom) == 25
        for p in df_coef.columns:
            assert int(sub_prom.loc[p, "n"]) == len(idxs)
            assert int(sub_prom.loc[p, "suma"]) == int(sub_coef[p].sum())
            assert float(sub_prom.loc[p, "media"]) == pytest.approx(
                float(sub_coef[p].mean()), rel=1e-12, abs=1e-12
            )


def test_promedios_ap_excluir_malos():
    curvas = [("11a1", (0, -1, 1, -10, -20), 0)]
    df_prom = promedios_ap(curvas, p_max=20, excluir_malos=True)
    por_p = df_prom.set_index("p")
    assert int(por_p.loc[11, "n"]) == 0
    assert int(por_p.loc[11, "suma"]) == 0
    assert np.isnan(float(por_p.loc[11, "media"]))
    for p in (2, 3, 5, 7, 13, 17, 19):
        assert int(por_p.loc[p, "n"]) == 1


def test_factor_euler_malos_referencia():
    curvas = [
        ("11a1", (0, -1, 1, -10, -20)),
        ("37a1", (0, 0, 1, -1, 0)),
        ("389a1", (0, 1, 1, -2, 0)),
    ]
    df_b = factor_euler_malos(curvas)
    assert list(df_b.columns) == ["etiqueta", "log_B", "n_malos"]
    por_et = df_b.set_index("etiqueta")
    assert float(por_et.loc["11a1", "log_B"]) == pytest.approx(
        np.log(11.0 / 10.0), rel=1e-7
    )
    assert float(por_et.loc["37a1", "log_B"]) == pytest.approx(
        np.log(37.0 / 38.0), rel=1e-7
    )
    assert float(por_et.loc["389a1", "log_B"]) == pytest.approx(
        np.log(389.0 / 388.0), rel=1e-7
    )
    assert int(por_et.loc["11a1", "n_malos"]) == 1
    assert int(por_et.loc["37a1", "n_malos"]) == 1
    assert int(por_et.loc["389a1", "n_malos"]) == 1


def test_tipos_reduccion_referencia():
    curvas = [
        ("11a1", (0, -1, 1, -10, -20)),
        ("37a1", (0, 0, 1, -1, 0)),
        ("389a1", (0, 1, 1, -2, 0)),
        ("27a1", (0, 0, 1, 0, -7)),
    ]
    df_tr = tipos_reduccion(curvas)
    assert list(df_tr.columns) == [
        "etiqueta",
        "n_partida",
        "n_no_partida",
        "n_aditiva",
    ]
    por_et = df_tr.set_index("etiqueta")
    esperados = {
        "11a1": (1, 0, 0),
        "37a1": (0, 1, 0),
        "389a1": (1, 0, 0),
        "27a1": (0, 0, 1),
    }
    for et, (np_esp, nnp_esp, na_esp) in esperados.items():
        assert int(por_et.loc[et, "n_partida"]) == np_esp
        assert int(por_et.loc[et, "n_no_partida"]) == nnp_esp
        assert int(por_et.loc[et, "n_aditiva"]) == na_esp



