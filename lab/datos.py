from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
from pathlib import Path
import urllib.request
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DIR_DATOS = RAIZ / "datos"
DIR_CREMONA = DIR_DATOS / "cremona"
RUTA_MANIFEST = DIR_DATOS / "MANIFEST.md"

URL_BASE = "https://raw.githubusercontent.com/JohnCremona/ecdata/master"


def _nombre_bloque(b: int) -> str:
    inicio = b * 10_000
    fin = inicio + 9_999
    return f"{inicio:05d}-{fin:05d}"


def _sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _descargar_uno(tipo: str, bloque: int) -> dict[str, str | int]:
    rango_str = _nombre_bloque(bloque)
    nombre = f"{tipo}.{rango_str}"
    url = f"{URL_BASE}/{tipo}/{nombre}"
    destino = DIR_CREMONA / nombre

    if not (destino.exists() and destino.stat().st_size > 0):
        temporal = destino.with_suffix(destino.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "bsd-lab/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, temporal.open("wb") as out:
            while True:
                bloque_bytes = resp.read(1024 * 256)
                if not bloque_bytes:
                    break
                out.write(bloque_bytes)
        temporal.replace(destino)

    tamano = destino.stat().st_size
    sha256 = _sha256_archivo(destino)
    return {
        "archivo": nombre,
        "url": url,
        "tamano": tamano,
        "sha256": sha256,
    }


def descargar(
    tipos: tuple[str, ...] = ("allbsd", "allcurves", "allgens"),
    bloques: range = range(10),
) -> None:
    DIR_CREMONA.mkdir(parents=True, exist_ok=True)
    tareas = [(tipo, b) for tipo in tipos for b in bloques]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futuros = [pool.submit(_descargar_uno, tipo, b) for tipo, b in tareas]
        registros = [fut.result() for fut in futuros]

    lineas = [
        "# MANIFEST — Datos de Cremona (`JohnCremona/ecdata`)",
        "",
        "| Archivo | URL | Tamaño (bytes) | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for reg in registros:
        lineas.append(
            f"| `{reg['archivo']}` | {reg['url']} | {reg['tamano']} | `{reg['sha256']}` |"
        )
    lineas.append("")
    RUTA_MANIFEST.write_text("\n".join(lineas), encoding="utf-8")


@functools.lru_cache(maxsize=10)
def _cargar_bloque_bsd(bloque: int) -> pd.DataFrame:
    nombre = f"allbsd.{_nombre_bloque(bloque)}"
    ruta = DIR_CREMONA / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo {ruta}")

    etiquetas: list[str] = []
    conductores: list[int] = []
    clases: list[str] = []
    numeros: list[int] = []
    ainvs_list: list[tuple[int, int, int, int, int]] = []
    rangos: list[int] = []
    torsiones: list[int] = []
    tamagawas: list[int] = []
    omegas: list[float] = []
    l_valores: list[float] = []
    reguladores: list[float] = []
    shas: list[float] = []

    with ruta.open("r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.split()
            if not partes:
                continue
            n_str, clase, num_str, ai_str, r_str, t_str, cp_str, om_str, l1_str, reg_str, sha_str = partes
            a1, a2, a3, a4, a6 = ai_str[1:-1].split(",")
            etiquetas.append(f"{n_str}{clase}{num_str}")
            conductores.append(int(n_str))
            clases.append(clase)
            numeros.append(int(num_str))
            ainvs_list.append((int(a1), int(a2), int(a3), int(a4), int(a6)))
            rangos.append(int(r_str))
            torsiones.append(int(t_str))
            tamagawas.append(int(cp_str))
            omegas.append(float(om_str))
            l_valores.append(float(l1_str))
            reguladores.append(float(reg_str))
            shas.append(float(sha_str))

    return pd.DataFrame(
        {
            "etiqueta": etiquetas,
            "conductor": conductores,
            "clase": clases,
            "numero": numeros,
            "ainvs": ainvs_list,
            "rango": rangos,
            "torsion": torsiones,
            "tamagawa": tamagawas,
            "omega": omegas,
            "l_valor": l_valores,
            "regulador": reguladores,
            "sha_an": shas,
        }
    )


def cargar_bsd(n_max: int = 99_999) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for b in range(10):
        inicio = b * 10_000
        if inicio > n_max:
            break
        df_b = _cargar_bloque_bsd(b)
        if inicio + 9_999 > n_max:
            df_b = df_b.loc[df_b["conductor"] <= n_max]
        dfs.append(df_b)

    if not dfs:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.concat(dfs, ignore_index=True)


@functools.lru_cache(maxsize=10)
def _cargar_bloque_generadores(bloque: int) -> tuple[pd.DataFrame, list[int]]:
    nombre = f"allgens.{_nombre_bloque(bloque)}"
    ruta = DIR_CREMONA / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo {ruta}")

    etiquetas: list[str] = []
    conductores: list[int] = []
    gens_list: list[list[str]] = []

    with ruta.open("r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.split()
            if not partes:
                continue
            n_str, clase, num_str = partes[0], partes[1], partes[2]
            rango = int(partes[4])
            gens = partes[6 : 6 + rango] if rango > 0 else []
            etiquetas.append(f"{n_str}{clase}{num_str}")
            conductores.append(int(n_str))
            gens_list.append(gens)

    df = pd.DataFrame({"etiqueta": etiquetas, "gens": gens_list})
    return df, conductores


def cargar_generadores(n_max: int = 99_999) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for b in range(10):
        inicio = b * 10_000
        if inicio > n_max:
            break
        df_b, conductores = _cargar_bloque_generadores(b)
        if inicio + 9_999 > n_max:
            mask = [c <= n_max for c in conductores]
            df_b = df_b.loc[mask]
        dfs.append(df_b)

    if not dfs:
        return pd.DataFrame(columns=["etiqueta", "gens"])
    return pd.concat(dfs, ignore_index=True)


def una_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["numero"] == 1].copy()
