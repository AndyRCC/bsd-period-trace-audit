from __future__ import annotations

import numpy as np


def permutar_en_bins(
    etiquetas: np.ndarray,
    bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    et = np.asarray(etiquetas)
    b = np.asarray(bins)
    if et.shape != b.shape:
        raise ValueError("etiquetas y bins deben tener la misma forma")
    resultado = np.array(et, copy=True)
    if len(et) <= 1:
        return resultado

    orden = np.argsort(b, kind="mergesort")
    b_ord = b[orden]
    cortes = np.flatnonzero(b_ord[1:] != b_ord[:-1]) + 1
    for idx_grupo in np.split(orden, cortes):
        if len(idx_grupo) > 1:
            resultado[idx_grupo] = rng.permutation(et[idx_grupo])
    return resultado


def diferencia_perfiles(
    ap: np.ndarray,
    buenos: np.ndarray,
    grupo: np.ndarray,
) -> tuple[np.ndarray, float]:
    a = np.asarray(ap, dtype=np.float64)
    b = np.asarray(buenos, dtype=bool)
    g = np.asarray(grupo, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("ap y buenos deben tener la misma forma")
    if a.shape[0] != g.shape[0]:
        raise ValueError("grupo debe tener longitud igual al número de filas de ap")

    mask_t = b & g[:, None]
    mask_f = b & (~g[:, None])

    n_t = mask_t.sum(axis=0, dtype=np.float64)
    n_f = mask_f.sum(axis=0, dtype=np.float64)

    suma_t = np.where(mask_t, a, 0.0).sum(axis=0)
    suma_f = np.where(mask_f, a, 0.0).sum(axis=0)

    media_t = suma_t / n_t
    media_f = suma_f / n_f
    delta = media_t - media_f
    rms = float(np.sqrt(np.mean(delta * delta)))
    return delta, rms


def distribucion_nula_perfiles(
    ap: np.ndarray,
    buenos: np.ndarray,
    grupo: np.ndarray,
    bins: np.ndarray,
    n_perm: int = 1000,
    semilla: int = 20261121,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(ap, dtype=np.float64)
    b = np.asarray(buenos, dtype=bool)
    g = np.asarray(grupo, dtype=bool)
    b_ids = np.asarray(bins)

    ap_buenos = np.where(b, a, 0.0)
    suma_tot = ap_buenos.sum(axis=0)
    n_tot = b.sum(axis=0, dtype=np.float64)

    orden = np.argsort(b_ids, kind="mergesort")
    b_ord = b_ids[orden]
    cortes = np.flatnonzero(b_ord[1:] != b_ord[:-1]) + 1
    grupos_idx = [idx for idx in np.split(orden, cortes) if len(idx) > 1]
    valores_orig = [g[idx].copy() for idx in grupos_idx]

    rng = np.random.default_rng(semilla)
    n_primos = a.shape[1]
    deltas_nulos = np.empty((n_perm, n_primos), dtype=np.float64)
    rms_nulos = np.empty(n_perm, dtype=np.float64)

    perm = g.copy()
    for i in range(n_perm):
        for idx, vals in zip(grupos_idx, valores_orig):
            perm[idx] = rng.permutation(vals)
        suma_t = ap_buenos[perm].sum(axis=0)
        n_t = b[perm].sum(axis=0, dtype=np.float64)
        suma_f = suma_tot - suma_t
        n_f = n_tot - n_t
        d = (suma_t / n_t) - (suma_f / n_f)
        deltas_nulos[i] = d
        rms_nulos[i] = float(np.sqrt(np.mean(d * d)))

    return deltas_nulos, rms_nulos


def correlacion_parcial_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
) -> tuple[float, int]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(bins)
    if xv.shape != yv.shape or xv.shape != bv.shape:
        raise ValueError("x, y y bins deben tener la misma forma")
    n = int(len(xv))
    if n == 0:
        return float("nan"), 0

    _, inv, counts = np.unique(bv, return_inverse=True, return_counts=True)
    n_bins = int(len(counts))
    sum_x = np.bincount(inv, weights=xv, minlength=n_bins)
    sum_y = np.bincount(inv, weights=yv, minlength=n_bins)
    mean_x = sum_x / counts
    mean_y = sum_y / counts

    rx = xv - mean_x[inv]
    ry = yv - mean_y[inv]

    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    rho = float(np.sum(rx * ry) / denom) if denom > 0.0 else 0.0
    n_ef = n - n_bins - 1
    return rho, n_ef


def covarianza_en_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(bins)
    if xv.shape != yv.shape or xv.shape != bv.shape:
        raise ValueError("x, y y bins deben tener la misma forma")
    n = int(len(xv))
    if n == 0:
        return float("nan")

    _, inv, counts = np.unique(bv, return_inverse=True, return_counts=True)
    n_bins = int(len(counts))
    sum_x = np.bincount(inv, weights=xv, minlength=n_bins)
    sum_y = np.bincount(inv, weights=yv, minlength=n_bins)
    rx = xv - (sum_x / counts)[inv]
    ry = yv - (sum_y / counts)[inv]
    return float(np.mean(rx * ry))


def correlacion_parcial_cov(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
    Z: np.ndarray,
) -> tuple[float, int]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(bins)
    zv = np.asarray(Z, dtype=np.float64)
    if xv.shape != yv.shape or xv.shape != bv.shape:
        raise ValueError("x, y y bins deben tener la misma forma")
    n = int(len(xv))
    if zv.ndim == 1:
        if zv.shape[0] != n:
            raise ValueError("Z debe tener la misma longitud que x")
        zv = zv[:, None]
    elif zv.ndim == 2:
        if zv.shape[0] != n:
            raise ValueError("Z debe tener el mismo número de filas que x")
    else:
        raise ValueError("Z debe ser un arreglo 1D o 2D")

    k_z = int(zv.shape[1])
    if n == 0:
        return float("nan"), 0

    _, inv, counts = np.unique(bv, return_inverse=True, return_counts=True)
    n_bins = int(len(counts))
    rx = xv - (np.bincount(inv, weights=xv, minlength=n_bins) / counts)[inv]
    ry = yv - (np.bincount(inv, weights=yv, minlength=n_bins) / counts)[inv]

    rz = np.empty_like(zv, dtype=np.float64)
    for col in range(k_z):
        zc = zv[:, col]
        rz[:, col] = zc - (np.bincount(inv, weights=zc, minlength=n_bins) / counts)[inv]

    beta_x, _, _, _ = np.linalg.lstsq(rz, rx, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(rz, ry, rcond=None)
    ex = rx - rz @ beta_x
    ey = ry - rz @ beta_y

    denom = float(np.sqrt(np.sum(ex * ex) * np.sum(ey * ey)))
    rho = float(np.sum(ex * ey) / denom) if denom > 0.0 else 0.0
    n_ef = n - n_bins - k_z - 1
    return rho, n_ef


def residuos_parciales_cov(
    x: np.ndarray,
    bins: np.ndarray,
    Z: np.ndarray | None = None,
) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    bv = np.asarray(bins)
    if xv.shape != bv.shape:
        raise ValueError("x y bins deben tener la misma forma")
    n = int(len(xv))
    if n == 0:
        return np.empty(0, dtype=np.float64)

    _, inv, counts = np.unique(bv, return_inverse=True, return_counts=True)
    n_bins = int(len(counts))
    rx = xv - (np.bincount(inv, weights=xv, minlength=n_bins) / counts)[inv]
    if Z is None:
        return rx

    zv = np.asarray(Z, dtype=np.float64)
    if zv.ndim == 1:
        zv = zv[:, None]
    k_z = int(zv.shape[1])
    rz = np.empty_like(zv, dtype=np.float64)
    for col in range(k_z):
        zc = zv[:, col]
        rz[:, col] = zc - (np.bincount(inv, weights=zc, minlength=n_bins) / counts)[inv]

    beta_x, _, _, _ = np.linalg.lstsq(rz, rx, rcond=None)
    return rx - rz @ beta_x


def bins_estratificados(
    L: np.ndarray,
    estrato: np.ndarray,
    q_max: int = 100,
    n_min: int = 200,
) -> np.ndarray:
    import pandas as pd

    lv = np.asarray(L, dtype=np.float64)
    ev = np.asarray(estrato)
    if lv.shape != ev.shape:
        raise ValueError("L y estrato deben tener la misma forma")
    n = int(len(lv))
    resultado = np.full(n, -1, dtype=int)
    if n == 0:
        return resultado

    orden = np.argsort(ev, kind="mergesort")
    e_ord = ev[orden]
    cortes = np.flatnonzero(e_ord[1:] != e_ord[:-1]) + 1
    offset = 0

    for idx_est in np.split(orden, cortes):
        n_est = int(len(idx_est))
        if n_est < n_min:
            continue
        q = max(1, min(int(q_max), n_est // 50))
        if q == 1:
            b_local = np.zeros(n_est, dtype=int)
        else:
            b_local = np.asarray(
                pd.qcut(lv[idx_est], q=q, labels=False, duplicates="drop"),
                dtype=int,
            )
        resultado[idx_est] = offset + b_local
        offset += int(b_local.max()) + 1

    return resultado


def clave_estrato(
    cp: np.ndarray,
    T: np.ndarray,
    sha: np.ndarray,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    cp_v = np.asarray(cp)
    t_v = np.asarray(T)
    sha_v = np.asarray(sha, dtype=np.float64)
    if cp_v.shape != t_v.shape or cp_v.shape != sha_v.shape:
        raise ValueError("cp, T y sha deben tener la misma forma")

    n = int(len(cp_v))
    c_code = np.zeros(n, dtype=np.int64)
    c_code[cp_v == 2] = 1
    c_code[cp_v == 3] = 2
    c_code[cp_v == 4] = 3
    c_code[(cp_v >= 5) & (cp_v <= 8)] = 4
    c_code[cp_v >= 9] = 5

    t_code = np.zeros(n, dtype=np.int64)
    t_code[t_v == 2] = 1
    t_code[t_v == 3] = 2
    t_code[t_v >= 4] = 3

    s_code = np.zeros(n, dtype=np.int64)
    s_code[np.rint(sha_v) >= 4] = 1

    base = c_code * 8 + t_code * 2 + s_code
    if extra is None:
        return base

    ex = np.asarray(extra)
    if ex.shape != base.shape:
        raise ValueError("extra debe tener la misma forma que cp, T y sha")
    if np.issubdtype(ex.dtype, np.integer):
        return ex.astype(np.int64) * 48 + base
    _, ex_inv = np.unique(ex, return_inverse=True)
    return ex_inv.astype(np.int64) * 48 + base





