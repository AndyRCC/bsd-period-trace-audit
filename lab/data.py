from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
from pathlib import Path
import urllib.request
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CREMONA_DIR = DATA_DIR / "cremona"
MANIFEST_PATH = DATA_DIR / "MANIFEST.md"

URL_BASE = "https://raw.githubusercontent.com/JohnCremona/ecdata/master"


def _block_name(b: int) -> str:
    start = b * 10_000
    end = start + 9_999
    return f"{start:05d}-{end:05d}"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(kind: str, block: int) -> dict[str, str | int]:
    rank_str = _block_name(block)
    name = f"{kind}.{rank_str}"
    url = f"{URL_BASE}/{kind}/{name}"
    dest = CREMONA_DIR / name

    if not (dest.exists() and dest.stat().st_size > 0):
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "bsd-lab/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                block_bytes = resp.read(1024 * 256)
                if not block_bytes:
                    break
                out.write(block_bytes)
        tmp.replace(dest)

    size = dest.stat().st_size
    sha256 = _file_sha256(dest)
    return {
        "file": name,
        "url": url,
        "size": size,
        "sha256": sha256,
    }


def download(
    kinds: tuple[str, ...] = ("allbsd", "allcurves", "allgens"),
    blocks: range = range(10),
) -> None:
    CREMONA_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(kind, b) for kind in kinds for b in blocks]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(_download_one, kind, b) for kind, b in tasks]
        records = [fut.result() for fut in futures]

    lines = [
        "# MANIFEST — Cremona data (`JohnCremona/ecdata`)",
        "",
        "| File | URL | Size (bytes) | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for reg in records:
        lines.append(
            f"| `{reg['file']}` | {reg['url']} | {reg['size']} | `{reg['sha256']}` |"
        )
    lines.append("")
    MANIFEST_PATH.write_text("\n".join(lines), encoding="utf-8")


@functools.lru_cache(maxsize=10)
def _load_bsd_block(block: int) -> pd.DataFrame:
    name = f"allbsd.{_block_name(block)}"
    path = CREMONA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    labels: list[str] = []
    conductors: list[int] = []
    classes: list[str] = []
    numbers: list[int] = []
    ainvs_list: list[tuple[int, int, int, int, int]] = []
    ranks: list[int] = []
    torsions: list[int] = []
    tamagawas: list[int] = []
    omegas: list[float] = []
    l_values: list[float] = []
    regulators: list[float] = []
    shas: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            n_str, isoclass, num_str, ai_str, r_str, t_str, cp_str, om_str, l1_str, reg_str, sha_str = parts
            a1, a2, a3, a4, a6 = ai_str[1:-1].split(",")
            labels.append(f"{n_str}{isoclass}{num_str}")
            conductors.append(int(n_str))
            classes.append(isoclass)
            numbers.append(int(num_str))
            ainvs_list.append((int(a1), int(a2), int(a3), int(a4), int(a6)))
            ranks.append(int(r_str))
            torsions.append(int(t_str))
            tamagawas.append(int(cp_str))
            omegas.append(float(om_str))
            l_values.append(float(l1_str))
            regulators.append(float(reg_str))
            shas.append(float(sha_str))

    return pd.DataFrame(
        {
            "label": labels,
            "conductor": conductors,
            "isoclass": classes,
            "number": numbers,
            "ainvs": ainvs_list,
            "rank": ranks,
            "torsion": torsions,
            "tamagawa": tamagawas,
            "omega": omegas,
            "l_value": l_values,
            "regulator": regulators,
            "sha_an": shas,
        }
    )


def load_bsd(n_max: int = 99_999) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for b in range(10):
        start = b * 10_000
        if start > n_max:
            break
        df_b = _load_bsd_block(b)
        if start + 9_999 > n_max:
            df_b = df_b.loc[df_b["conductor"] <= n_max]
        dfs.append(df_b)

    if not dfs:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.concat(dfs, ignore_index=True)


@functools.lru_cache(maxsize=10)
def _load_generators_block(block: int) -> tuple[pd.DataFrame, list[int]]:
    name = f"allgens.{_block_name(block)}"
    path = CREMONA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    labels: list[str] = []
    conductors: list[int] = []
    gens_list: list[list[str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            n_str, isoclass, num_str = parts[0], parts[1], parts[2]
            rank = int(parts[4])
            gens = parts[6 : 6 + rank] if rank > 0 else []
            labels.append(f"{n_str}{isoclass}{num_str}")
            conductors.append(int(n_str))
            gens_list.append(gens)

    df = pd.DataFrame({"label": labels, "gens": gens_list})
    return df, conductors


def load_generators(n_max: int = 99_999) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for b in range(10):
        start = b * 10_000
        if start > n_max:
            break
        df_b, conductors = _load_generators_block(b)
        if start + 9_999 > n_max:
            mask = [c <= n_max for c in conductors]
            df_b = df_b.loc[mask]
        dfs.append(df_b)

    if not dfs:
        return pd.DataFrame(columns=["label", "gens"])
    return pd.concat(dfs, ignore_index=True)


def one_per_class(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["number"] == 1].copy()
