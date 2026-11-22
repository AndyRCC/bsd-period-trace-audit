from __future__ import annotations

import os
from pathlib import Path
import subprocess
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_GP_TXT = Path(__file__).resolve().parent / "gp_ruta.txt"

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


def _obtener_gp_exe() -> str:
    env_gp = os.environ.get("GP_EXE", "").strip()
    if env_gp:
        return env_gp
    if RUTA_GP_TXT.exists():
        primera = RUTA_GP_TXT.read_text(encoding="utf-8").splitlines()[0].strip()
        if primera:
            os.environ["GP_EXE"] = primera
            return primera
    raise FileNotFoundError(
        "No se encontró la ruta a gp.exe ni en GP_EXE ni en lab/gp_ruta.txt"
    )


def _punto_proyectivo_a_gp(punto_str: str) -> str:
    limpio = punto_str.strip()
    if not (limpio.startswith("[") and limpio.endswith("]")):
        raise ValueError(f"Formato de punto inválido: {punto_str!r}")
    partes = limpio[1:-1].split(":")
    if len(partes) != 3:
        raise ValueError(f"Se esperaban 3 coordenadas proyectivas en {punto_str!r}")
    x_int = int(partes[0])
    y_int = int(partes[1])
    z_int = int(partes[2])
    if z_int == 0:
        return "[0]"
    return f"[{x_int}/{z_int},{y_int}/{z_int}]"


def _tiene_error_gp(texto: str) -> bool:
    for linea in texto.splitlines():
        if "***" in linea:
            limpia = linea.strip()
            if limpia.startswith("***   Warning: new maximum stack size") or limpia.startswith(
                "***   Warning: increasing stack size"
            ):
                continue
            return True
    return False


def correr_gp(script: str, timeout_s: int = 900) -> str:
    gp_exe = _obtener_gp_exe()
    script_completo = f"default(parisizemax, 2^30);\n{script}\nquit;\n"
    try:
        proc = subprocess.run(
            [gp_exe, "-f", "-q"],
            input=script_completo,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"PARI/GP excedió el tiempo límite de {timeout_s} segundos"
        ) from exc

    salida = proc.stdout or ""
    errores = proc.stderr or ""
    if _tiene_error_gp(salida) or _tiene_error_gp(errores) or proc.returncode != 0:
        detalle = (errores + "\n" + salida).strip()
        raise RuntimeError(f"Error en ejecución de PARI/GP (rc={proc.returncode}):\n{detalle}")

    return salida


def invariantes_pari(
    curvas: list[tuple[str, tuple[int, ...], list[str]]],
) -> pd.DataFrame:
    if not curvas:
        return pd.DataFrame(columns=COLUMNAS_INVARIANTES)

    lineas_gp: list[str] = [
        "default(realprecision, 19);",
        "calc_inv(et, ai, pts) = {",
        "  my(E = ellinit(ai), ar, r, lval, om, cp, tor, reg, eps);",
        "  foreach(pts, P, if(!ellisoncurve(E, P), error(\"Punto fuera de la curva: \", et, \" \", P)));",
        "  ar = ellanalyticrank(E);",
        "  r = ar[1];",
        "  lval = ar[2] / (r!);",
        "  om = E.omega[1] * if(E.disc > 0, 2, 1);",
        "  cp = ellglobalred(E)[3];",
        "  tor = elltors(E)[1];",
        "  reg = if(#pts == 0, 1.0, matdet(ellheightmatrix(E, pts)));",
        "  eps = ellrootno(E);",
        "  printf(\"RES\\t%s\\t%d\\t%.17g\\t%.17g\\t%d\\t%d\\t%.17g\\t%d\\n\", et, r, lval, om, cp, tor, reg, eps);",
        "};",
    ]

    for etiqueta, ainvs, gens in curvas:
        if len(ainvs) != 5:
            raise ValueError(f"Se esperaban 5 coeficientes Weierstrass para {etiqueta}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        pts_gp = "[" + ",".join(_punto_proyectivo_a_gp(p) for p in gens) + "]"
        et_segura = etiqueta.replace("\\", "\\\\").replace('"', '\\"')
        lineas_gp.append(f'calc_inv("{et_segura}", {ai_gp}, {pts_gp});')

    salida = correr_gp("\n".join(lineas_gp))

    registros: list[dict[str, str | int | float]] = []
    for linea in salida.splitlines():
        if not linea.startswith("RES\t"):
            continue
        _, et, r_s, l_s, om_s, cp_s, tor_s, reg_s, eps_s = linea.split("\t")
        registros.append(
            {
                "etiqueta": et,
                "rango_an": int(r_s),
                "l_valor": float(l_s.replace(" ", "")),
                "omega": float(om_s.replace(" ", "")),
                "tamagawa": int(cp_s),
                "torsion": int(tor_s),
                "regulador": float(reg_s.replace(" ", "")),
                "raiz": int(eps_s),
            }
        )

    if len(registros) != len(curvas):
        raise RuntimeError(
            f"Se esperaban {len(curvas)} filas de PARI/GP, pero se recibieron {len(registros)}"
        )

    return pd.DataFrame(registros, columns=COLUMNAS_INVARIANTES)


def coeficientes_ap(
    curvas: list[tuple[str, tuple[int, ...]]],
    n_primos: int = 1000,
) -> pd.DataFrame:
    if n_primos <= 0:
        raise ValueError("n_primos debe ser positivo")

    lineas_gp: list[str] = [
        f"my_primes = primes({int(n_primos)});",
        'print("PRIMOS\\t", strjoin([Str(p) | p <- my_primes], "\\t"));',
    ]

    for etiqueta, ainvs in curvas:
        if len(ainvs) != 5:
            raise ValueError(f"Se esperaban 5 coeficientes Weierstrass para {etiqueta}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        et_segura = etiqueta.replace("\\", "\\\\").replace('"', '\\"')
        lineas_gp.append(
            f'my(E = ellinit({ai_gp}, 1)); print("AP\\t{et_segura}\\t", strjoin([Str(ellap(E, p)) | p <- my_primes], "\\t"));'
        )

    salida = correr_gp("\n".join(lineas_gp))

    primos: list[int] = []
    etiquetas: list[str] = []
    filas_ap: list[list[int]] = []

    for linea in salida.splitlines():
        if linea.startswith("PRIMOS\t"):
            partes = linea.split("\t")[1:]
            primos = [int(p) for p in partes]
        elif linea.startswith("AP\t"):
            partes = linea.split("\t")
            etiquetas.append(partes[1])
            filas_ap.append([int(x) for x in partes[2:]])

    if not CurvasOPrimosValidos(curvas, primos, filas_ap):
        raise RuntimeError("Salida incompleta de PARI/GP en coeficientes_ap")

    df = pd.DataFrame(filas_ap, index=pd.Index(etiquetas, name="etiqueta"), columns=primos)
    return df


def CurvasOPrimosValidos(
    curvas: list[tuple[str, tuple[int, ...]]],
    primos: list[int],
    filas_ap: list[list[int]],
) -> bool:
    return len(primos) > 0 and len(filas_ap) == len(curvas)


def promedios_ap(
    curvas: list[tuple[str, tuple[int, ...], int]],
    p_max: int,
    excluir_malos: bool = False,
) -> pd.DataFrame:
    columnas = ["grupo", "p", "n", "suma", "media"]
    if not curvas or p_max < 2:
        return pd.DataFrame(columns=columnas)

    grupos_unicos = sorted({int(g) for _, _, g in curvas})
    mapa_g = {g: idx + 1 for idx, g in enumerate(grupos_unicos)}
    n_grupos = len(grupos_unicos)
    grupos_gp = "[" + ",".join(str(g) for g in grupos_unicos) + "]"

    lineas_gp: list[str] = [
        f"P = primes([2, {int(p_max)}]);",
        "nP = #P;",
        f"grupos = {grupos_gp};",
        f"nG = {n_grupos};",
        "S = matrix(nG, nP);",
        "C = matrix(nG, nP);",
    ]

    if excluir_malos:
        lineas_gp.extend(
            [
                "acum(ig, ai) = {",
                "  my(E = ellinit(ai, 1), D = E.disc, p);",
                "  for(j = 1, nP,",
                "    p = P[j];",
                "    if(D % p != 0,",
                "      C[ig, j]++;",
                "      S[ig, j] += ellap(E, p);",
                "    );",
                "  );",
                "};",
            ]
        )
    else:
        lineas_gp.extend(
            [
                "cntG = vector(nG);",
                "acum(ig, ai) = {",
                "  my(E = ellinit(ai, 1));",
                "  cntG[ig]++;",
                "  for(j = 1, nP, S[ig, j] += ellap(E, P[j]));",
                "};",
            ]
        )

    for etiqueta, ainvs, grupo in curvas:
        if len(ainvs) != 5:
            raise ValueError(f"Se esperaban 5 coeficientes Weierstrass para {etiqueta}")
        ig = mapa_g[int(grupo)]
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        lineas_gp.append(f"acum({ig}, {ai_gp});")

    if not excluir_malos:
        lineas_gp.append("for(ig = 1, nG, for(j = 1, nP, C[ig, j] = cntG[ig]));")

    lineas_gp.append(
        'for(ig = 1, nG, for(j = 1, nP, printf("PROM\\t%d\\t%d\\t%d\\t%d\\n", grupos[ig], P[j], C[ig, j], S[ig, j])));'
    )

    salida = correr_gp("\n".join(lineas_gp))

    registros: list[dict[str, int | float]] = []
    for linea in salida.splitlines():
        if not linea.startswith("PROM\t"):
            continue
        _, g_s, p_s, n_s, s_s = linea.split("\t")
        n_val = int(n_s)
        s_val = int(s_s)
        media_val = (s_val / n_val) if n_val > 0 else float("nan")
        registros.append(
            {
                "grupo": int(g_s),
                "p": int(p_s),
                "n": n_val,
                "suma": s_val,
                "media": media_val,
            }
        )

    if not registros:
        raise RuntimeError("Salida vacía de PARI/GP en promedios_ap")

    return pd.DataFrame(registros, columns=columnas)


def _extraer_conductor_etiqueta(etiqueta: str) -> int:
    n_chars = 0
    for ch in etiqueta:
        if ch.isdigit():
            n_chars += 1
        else:
            break
    return int(etiqueta[:n_chars]) if n_chars > 0 else 0


def factor_euler_malos(
    curvas: list[tuple[str, tuple[int, ...]]],
) -> pd.DataFrame:
    columnas = ["etiqueta", "log_B", "n_malos"]
    if not curvas:
        return pd.DataFrame(columns=columnas)

    lineas_gp: list[str] = [
        "default(realprecision, 19);",
        "calc_eb(et, ai, nh) = {",
        "  my(E = ellinit(ai, 1), M = if(nh > 1 && E.disc % nh == 0, nh, abs(E.disc)), P = factor(M)[, 1], B = 1, ap);",
        "  foreach(P, p, ap = ellap(E, p); B *= p / (p - ap));",
        '  printf("EB\\t%s\\t%.17g\\t%d\\n", et, if(B == 1, 0.0, log(B)), #P);',
        "};",
    ]

    for etiqueta, ainvs in curvas:
        if len(ainvs) != 5:
            raise ValueError(f"Se esperaban 5 coeficientes Weierstrass para {etiqueta}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        et_segura = etiqueta.replace("\\", "\\\\").replace('"', '\\"')
        nh = _extraer_conductor_etiqueta(etiqueta)
        lineas_gp.append(f'calc_eb("{et_segura}", {ai_gp}, {nh});')

    salida = correr_gp("\n".join(lineas_gp))

    registros: list[dict[str, str | int | float]] = []
    for linea in salida.splitlines():
        if not linea.startswith("EB\t"):
            continue
        _, et, logb_s, nm_s = linea.split("\t")
        registros.append(
            {
                "etiqueta": et,
                "log_B": float(logb_s.replace(" ", "")),
                "n_malos": int(nm_s),
            }
        )

    if len(registros) != len(curvas):
        raise RuntimeError(
            f"Se esperaban {len(curvas)} filas en factor_euler_malos, se recibieron {len(registros)}"
        )

    return pd.DataFrame(registros, columns=columnas)


def tipos_reduccion(
    curvas: list[tuple[str, tuple[int, ...]]],
) -> pd.DataFrame:
    columnas = ["etiqueta", "n_partida", "n_no_partida", "n_aditiva"]
    if not curvas:
        return pd.DataFrame(columns=columnas)

    lineas_gp: list[str] = [
        "calc_tr(et, ai, nh) = {",
        "  my(E = ellinit(ai, 1), M = if(nh > 1 && E.disc % nh == 0, nh, abs(E.disc)), P = factor(M)[, 1], np = 0, nnp = 0, na = 0, ap);",
        "  foreach(P, q, ap = ellap(E, q); if(ap == 1, np++, if(ap == -1, nnp++, if(ap == 0, na++, error(\"ap inesperado en primo malo: \", et, \" q=\", q, \" ap=\", ap)))));",
        '  printf("TR\\t%s\\t%d\\t%d\\t%d\\n", et, np, nnp, na);',
        "};",
    ]

    for etiqueta, ainvs in curvas:
        if len(ainvs) != 5:
            raise ValueError(f"Se esperaban 5 coeficientes Weierstrass para {etiqueta}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        et_segura = etiqueta.replace("\\", "\\\\").replace('"', '\\"')
        nh = _extraer_conductor_etiqueta(etiqueta)
        lineas_gp.append(f'calc_tr("{et_segura}", {ai_gp}, {nh});')

    salida = correr_gp("\n".join(lineas_gp))

    registros: list[dict[str, str | int]] = []
    for linea in salida.splitlines():
        if not linea.startswith("TR\t"):
            continue
        _, et, np_s, nnp_s, na_s = linea.split("\t")
        registros.append(
            {
                "etiqueta": et,
                "n_partida": int(np_s),
                "n_no_partida": int(nnp_s),
                "n_aditiva": int(na_s),
            }
        )

    if len(registros) != len(curvas):
        raise RuntimeError(
            f"Se esperaban {len(curvas)} filas en tipos_reduccion, se recibieron {len(registros)}"
        )

    return pd.DataFrame(registros, columns=columnas)



