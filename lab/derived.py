from __future__ import annotations

import functools
from pathlib import Path
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.gp import ap_coefficients

ROOT = Path(__file__).resolve().parents[1]
DERIVED_DIR = ROOT / "data" / "derived"
MAX_BATCH_SIZE = 50_000


def discriminant(ainvs: tuple[int, ...]) -> int:
    if len(ainvs) != 5:
        raise ValueError("Expected 5 Weierstrass coefficients (a1, a2, a3, a4, a6)")
    a1, a2, a3, a4, a6 = (int(x) for x in ainvs)
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = (
        a1 * a1 * a6
        + 4 * a2 * a6
        - a1 * a3 * a4
        + a2 * a3 * a3
        - a4 * a4
    )
    return -b2 * b2 * b8 - 8 * (b4**3) - 27 * (b6 * b6) + 9 * b2 * b4 * b6


def build_ap_cache(n_primes: int = 25) -> Path:
    if n_primes <= 0:
        raise ValueError("n_primes must be positive")
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = DERIVED_DIR / f"ap_{int(n_primes)}.npz"

    classes = one_per_class(load_bsd(n_max=99_999)).reset_index(drop=True)
    total = len(classes)

    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (
                "labels" in data
                and "primes" in data
                and "ap" in data
                and data["ap"].shape == (total, int(n_primes))
            ):
                return path

    curves = [(row["label"], row["ainvs"]) for _, row in classes.iterrows()]
    blocks_ap: list[np.ndarray] = []
    primes_arr: np.ndarray | None = None

    for start in range(0, total, MAX_BATCH_SIZE):
        end = min(start + MAX_BATCH_SIZE, total)
        batch = curves[start:end]
        df_batch = ap_coefficients(batch, n_primes=int(n_primes))
        if primes_arr is None:
            primes_arr = np.array([int(p) for p in df_batch.columns], dtype=np.int32)
        blocks_ap.append(df_batch.to_numpy(dtype=np.int16))

    if primes_arr is None:
        raise RuntimeError("Could not compute the ap coefficients")

    matrix_ap = np.vstack(blocks_ap).astype(np.int16, copy=False)
    labels_arr = classes["label"].to_numpy(dtype="U16")

    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        labels=labels_arr,
        primes=primes_arr,
        ap=matrix_ap,
    )
    tmp.replace(path)
    _load_ap_cache_mem.cache_clear()
    return path


@functools.lru_cache(maxsize=4)
def _load_ap_cache_mem(n_primes: int) -> pd.DataFrame:
    path = build_ap_cache(n_primes=n_primes)
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"]
        primes = [int(p) for p in data["primes"]]
        ap = data["ap"].astype(np.int16, copy=True)
    return pd.DataFrame(ap, index=pd.Index(labels, name="label"), columns=primes)


def load_ap_cache(n_primes: int = 25) -> pd.DataFrame:
    return _load_ap_cache_mem(int(n_primes)).copy()


def legendre_symbol(a: int, p: int) -> int:
    p_int = int(p)
    if p_int <= 2 or p_int % 2 == 0:
        raise ValueError("p must be an odd prime")
    r = int(a) % p_int
    if r == 0:
        return 0
    e = pow(r, (p_int - 1) // 2, p_int)
    return 1 if e == 1 else -1

