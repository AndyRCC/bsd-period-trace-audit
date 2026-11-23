from __future__ import annotations

import numpy as np


def permute_in_bins(
    labels: np.ndarray,
    bins: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    tag = np.asarray(labels)
    b = np.asarray(bins)
    if tag.shape != b.shape:
        raise ValueError("labels and bins must have the same shape")
    result = np.array(tag, copy=True)
    if len(tag) <= 1:
        return result

    order = np.argsort(b, kind="mergesort")
    b_ord = b[order]
    cuts = np.flatnonzero(b_ord[1:] != b_ord[:-1]) + 1
    for idx_group in np.split(order, cuts):
        if len(idx_group) > 1:
            result[idx_group] = rng.permutation(tag[idx_group])
    return result


def profile_difference(
    ap: np.ndarray,
    good: np.ndarray,
    group: np.ndarray,
) -> tuple[np.ndarray, float]:
    a = np.asarray(ap, dtype=np.float64)
    b = np.asarray(good, dtype=bool)
    g = np.asarray(group, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("ap and good must have the same shape")
    if a.shape[0] != g.shape[0]:
        raise ValueError("group must have length equal to the number of rows of ap")

    mask_t = b & g[:, None]
    mask_f = b & (~g[:, None])

    n_t = mask_t.sum(axis=0, dtype=np.float64)
    n_f = mask_f.sum(axis=0, dtype=np.float64)

    sum_t = np.where(mask_t, a, 0.0).sum(axis=0)
    sum_f = np.where(mask_f, a, 0.0).sum(axis=0)

    mean_t = sum_t / n_t
    mean_f = sum_f / n_f
    delta = mean_t - mean_f
    rms = float(np.sqrt(np.mean(delta * delta)))
    return delta, rms


def null_profile_distribution(
    ap: np.ndarray,
    good: np.ndarray,
    group: np.ndarray,
    bins: np.ndarray,
    n_perm: int = 1000,
    seed: int = 20261121,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(ap, dtype=np.float64)
    b = np.asarray(good, dtype=bool)
    g = np.asarray(group, dtype=bool)
    b_ids = np.asarray(bins)

    ap_good = np.where(b, a, 0.0)
    sum_tot = ap_good.sum(axis=0)
    n_tot = b.sum(axis=0, dtype=np.float64)

    order = np.argsort(b_ids, kind="mergesort")
    b_ord = b_ids[order]
    cuts = np.flatnonzero(b_ord[1:] != b_ord[:-1]) + 1
    groups_idx = [idx for idx in np.split(order, cuts) if len(idx) > 1]
    values_orig = [g[idx].copy() for idx in groups_idx]

    rng = np.random.default_rng(seed)
    n_primes = a.shape[1]
    deltas_nulls = np.empty((n_perm, n_primes), dtype=np.float64)
    rms_nulls = np.empty(n_perm, dtype=np.float64)

    perm = g.copy()
    for i in range(n_perm):
        for idx, vals in zip(groups_idx, values_orig):
            perm[idx] = rng.permutation(vals)
        sum_t = ap_good[perm].sum(axis=0)
        n_t = b[perm].sum(axis=0, dtype=np.float64)
        sum_f = sum_tot - sum_t
        n_f = n_tot - n_t
        d = (sum_t / n_t) - (sum_f / n_f)
        deltas_nulls[i] = d
        rms_nulls[i] = float(np.sqrt(np.mean(d * d)))

    return deltas_nulls, rms_nulls


def partial_correlation_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
) -> tuple[float, int]:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(bins)
    if xv.shape != yv.shape or xv.shape != bv.shape:
        raise ValueError("x, y and bins must have the same shape")
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


def covariance_in_bins(
    x: np.ndarray,
    y: np.ndarray,
    bins: np.ndarray,
) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    bv = np.asarray(bins)
    if xv.shape != yv.shape or xv.shape != bv.shape:
        raise ValueError("x, y and bins must have the same shape")
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


def partial_correlation_cov(
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
        raise ValueError("x, y and bins must have the same shape")
    n = int(len(xv))
    if zv.ndim == 1:
        if zv.shape[0] != n:
            raise ValueError("Z must have the same length as x")
        zv = zv[:, None]
    elif zv.ndim == 2:
        if zv.shape[0] != n:
            raise ValueError("Z must have the same number of rows as x")
    else:
        raise ValueError("Z must be a 1D or 2D array")

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


def partial_residuals_cov(
    x: np.ndarray,
    bins: np.ndarray,
    Z: np.ndarray | None = None,
) -> np.ndarray:
    xv = np.asarray(x, dtype=np.float64)
    bv = np.asarray(bins)
    if xv.shape != bv.shape:
        raise ValueError("x and bins must have the same shape")
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


def stratified_bins(
    L: np.ndarray,
    stratum: np.ndarray,
    q_max: int = 100,
    n_min: int = 200,
) -> np.ndarray:
    import pandas as pd

    lv = np.asarray(L, dtype=np.float64)
    ev = np.asarray(stratum)
    if lv.shape != ev.shape:
        raise ValueError("L and stratum must have the same shape")
    n = int(len(lv))
    result = np.full(n, -1, dtype=int)
    if n == 0:
        return result

    order = np.argsort(ev, kind="mergesort")
    e_ord = ev[order]
    cuts = np.flatnonzero(e_ord[1:] != e_ord[:-1]) + 1
    offset = 0

    for idx_strat in np.split(order, cuts):
        n_strat = int(len(idx_strat))
        if n_strat < n_min:
            continue
        q = max(1, min(int(q_max), n_strat // 50))
        if q == 1:
            b_local = np.zeros(n_strat, dtype=int)
        else:
            b_local = np.asarray(
                pd.qcut(lv[idx_strat], q=q, labels=False, duplicates="drop"),
                dtype=int,
            )
        result[idx_strat] = offset + b_local
        offset += int(b_local.max()) + 1

    return result


def stratum_key(
    cp: np.ndarray,
    T: np.ndarray,
    sha: np.ndarray,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    cp_v = np.asarray(cp)
    t_v = np.asarray(T)
    sha_v = np.asarray(sha, dtype=np.float64)
    if cp_v.shape != t_v.shape or cp_v.shape != sha_v.shape:
        raise ValueError("cp, T and sha must have the same shape")

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
        raise ValueError("extra must have the same shape as cp, T and sha")
    if np.issubdtype(ex.dtype, np.integer):
        return ex.astype(np.int64) * 48 + base
    _, ex_inv = np.unique(ex, return_inverse=True)
    return ex_inv.astype(np.int64) * 48 + base





