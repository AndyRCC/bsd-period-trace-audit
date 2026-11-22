"""Genera versiones con etiquetas y títulos en inglés de las cuatro figuras del
informe en informe/figuras/, usando exactamente los mismos
CSV de resultados. En la Figura 4 (h16.png) añade la serie del control conjunto
{log B, log(|Δ|/N)} tomada de e10_1_diagnostico_riesgo.csv.
"""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
DIR_RES = RAIZ / "resultados"
DIR_CREMONA = RAIZ / "datos" / "cremona"
DIR_OUT = RAIZ / "informe" / "figuras"


def _nbar_r01(a: int, b: int) -> float:
    ns = []
    for bl in range(10):
        ini = bl * 10_000
        fin = ini + 9_999
        if fin < a or ini > b:
            continue
        ruta = DIR_CREMONA / f"allbsd.{ini:05d}-{fin:05d}"
        with ruta.open("r", encoding="utf-8") as fh:
            for ln in fh:
                p = ln.strip().split()
                if not p:
                    continue
                if int(p[2]) != 1:
                    continue
                N = int(p[0])
                r = int(p[4])
                if a <= N <= b and r in (0, 1):
                    ns.append(N)
    return float(np.mean(ns))


def _binar_x(primos: np.ndarray, vals: np.ndarray, nbar: float) -> tuple[np.ndarray, np.ndarray]:
    x = primos.astype(np.float64) / nbar
    edges = np.linspace(0.0, 0.9, 51)
    centros = 0.5 * (edges[:-1] + edges[1:])
    medias = np.full(50, np.nan, dtype=np.float64)
    for i in range(50):
        m = (x >= edges[i]) & (x < edges[i + 1]) if i < 49 else (x >= edges[i]) & (x <= edges[i + 1])
        if np.any(m):
            medias[i] = float(np.mean(vals[m]))
    return centros, medias


def fig1_escala() -> None:
    df = pd.read_csv(DIR_RES / "04-murmuraciones" / "promedios.csv")
    p75_0 = df[(df["rango_N"] == "7500-10000") & (df["variante"] == "todos") & (df["grupo"] == 0)].sort_values("p")
    p75_1 = df[(df["rango_N"] == "7500-10000") & (df["variante"] == "todos") & (df["grupo"] == 1)].sort_values("p")
    p20_0 = df[(df["rango_N"] == "20000-25000") & (df["variante"] == "todos") & (df["grupo"] == 0)].sort_values("p")
    p20_1 = df[(df["rango_N"] == "20000-25000") & (df["variante"] == "todos") & (df["grupo"] == 1)].sort_values("p")

    pr75 = p75_0["p"].to_numpy(dtype=np.float64)
    d75 = p75_0["media"].to_numpy(dtype=np.float64) - p75_1["media"].to_numpy(dtype=np.float64)
    pr20 = p20_0["p"].to_numpy(dtype=np.float64)
    d20 = p20_0["media"].to_numpy(dtype=np.float64) - p20_1["media"].to_numpy(dtype=np.float64)

    nbar75 = _nbar_r01(7500, 10000)
    nbar20 = _nbar_r01(20000, 25000)

    cx, b75 = _binar_x(pr75, d75, nbar75)
    _, b20 = _binar_x(pr20, d20, nbar20)
    _, b75n = _binar_x(pr75, d75 / np.sqrt(pr75), nbar75)
    _, b20n = _binar_x(pr20, d20 / np.sqrt(pr20), nbar20)

    r_raw = float(np.corrcoef(b75, b20)[0, 1])
    r_norm = float(np.corrcoef(b75n, b20n)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), dpi=150)
    ax1.plot(cx, b75, "o-", color="#1f5a99", markersize=4, linewidth=1.6, label=r"$N \in [7500, 10000]$")
    ax1.plot(cx, b20, "s--", color="#c0392b", markersize=4, linewidth=1.6, label=r"$N \in [20000, 25000]$")
    ax1.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9)
    ax1.set_xlabel(r"Scaled prime $x = p / \bar{N}$")
    ax1.set_ylabel(r"Binned $D(p) = f_0(p) - f_1(p)$")
    ax1.set_title(f"(A) Raw amplitude ($r = {r_raw:.4f}$)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(cx, b75n, "o-", color="#1f5a99", markersize=4, linewidth=1.6, label=r"$N \in [7500, 10000]$")
    ax2.plot(cx, b20n, "s--", color="#c0392b", markersize=4, linewidth=1.6, label=r"$N \in [20000, 25000]$")
    ax2.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9)
    ax2.set_xlabel(r"Scaled prime $x = p / \bar{N}$")
    ax2.set_ylabel(r"Binned $D(p) / \sqrt{p}$")
    ax2.set_title(f"(B) Normalised by $\\sqrt{{p}}$ ($r = {r_norm:.4f}$)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Scale invariance of elliptic curve murmurations (one curve per isogeny class)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_OUT / "escala.png", bbox_inches="tight")
    plt.close(fig)


def fig2_mecanismo() -> None:
    df = pd.read_csv(DIR_RES / "07-mecanismo-h8" / "colisionador.csv")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 5.1), dpi=150, sharey=True)
    series_cfg = [
        ("rho_logomega", "o-", "#1f5a99", "No covariate (baseline)"),
        ("rho_logB", "s-", "#c0392b", r"Controlling $Z = \log B$"),
        ("rho_logtam", "^-", "#8e44ad", r"Controlling $Z = \log \mathrm{Tam}$"),
        ("rho_logdisc", "d-", "#27ae60", r"Controlling $Z = \log |\Delta|$"),
        ("rho_placebo", "x:", "#7f8c8d", "Controlling $Z =$ placebo noise"),
    ]
    for ax, rango, titulo in [
        (ax0, 0, r"(A) Rank 0 ($N < 50\,000$): $\rho(a_p, \log \Omega \mid L(1), Z)$"),
        (ax1, 1, r"(B) Rank 1 ($N < 50\,000$): $\rho(a_p, \log \Omega \mid L'(1), Z)$"),
    ]:
        sub = df[(df["rango"] == rango) & (df["ventana"] == "1-49999")].sort_values("p")
        for col_y, fmt, col, lbl in series_cfg:
            ax.plot(sub["p"], sub[col_y], fmt, color=col, markersize=4.3, linewidth=1.45, label=lbl)
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9)
        ax.set_xlabel(r"Good prime $p \leq 97$ ($p$ not dividing $N$)")
        ax.set_ylabel(r"Partial correlation $\rho$")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        r"Effect of candidate covariates on $\rho(a_p, \log \Omega \mid L^{(r)})$ for $N < 50\,000$ (one curve per class)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_OUT / "mecanismo.png", bbox_inches="tight")
    plt.close(fig)


def fig3_estratos() -> None:
    df = pd.read_csv(DIR_RES / "09-estratos-legendre" / "estratos.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.3), dpi=150, sharey=True)
    tipos_plot = [
        ("global", "o-", "#1f5a99", -0.15, r"Pooled ($L$-bins only)"),
        ("agrupada", "s-", "#c0392b", 0.0, r"Pooled within $(\mathrm{Tam}, T, \mathrm{Sha})$ strata"),
        ("puro", "^--", "#27ae60", 0.15, r"Pure stratum ($\mathrm{Tam}=T=\mathrm{Sha}=1$)"),
    ]
    etiquetas_x = [
        "p=3\n(1–50k)",
        "p=5\n(1–50k)",
        "p=7\n(1–50k)",
        "p=3\n(50k–100k)",
        "p=5\n(50k–100k)",
        "p=7\n(50k–100k)",
    ]
    pares_xp = [
        ("1-49999", 3),
        ("1-49999", 5),
        ("1-49999", 7),
        ("50000-99999", 3),
        ("50000-99999", 5),
        ("50000-99999", 7),
    ]
    x_base = np.arange(len(pares_xp), dtype=float)

    for ax, rango, titulo in [
        (axes[0], 1, r"(A) Rank 1: $\rho(a_p, \log \Omega \mid L'(1)) \pm 2\,\mathrm{s.e.}$"),
        (axes[1], 2, r"(B) Rank 2: $\rho(a_p, \log \Omega \mid L''(1)/2) \pm 2\,\mathrm{s.e.}$"),
    ]:
        sub_r = df[df["rango"] == rango]
        for tipo_key, fmt, col, dx, label in tipos_plot:
            ys, errs = [], []
            for vent, pv in pares_xp:
                fila = sub_r[(sub_r["ventana"] == vent) & (sub_r["p"] == pv) & (sub_r["tipo"] == tipo_key)].iloc[0]
                ys.append(float(fila["rho"]))
                errs.append(2.0 * float(fila["se"]))
            ax.errorbar(
                x_base + dx, ys, yerr=errs, fmt=fmt, color=col, ecolor=col,
                elinewidth=1.25, capsize=3.3, markersize=5.8, linewidth=1.55, label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9)
        ax.axvline(2.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(etiquetas_x)
        ax.set_ylabel(r"Partial correlation $\rho(a_p, \log \Omega \mid L^{(r)})$")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.7)

    fig.suptitle(
        r"Composition effect (Simpson reversal) across $(\mathrm{Tam}, T, \mathrm{Sha})$ strata at ranks 1 and 2",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_OUT / "estratos.png", bbox_inches="tight")
    plt.close(fig)


def fig4_h16() -> None:
    df_p12 = pd.read_csv(DIR_RES / "10-cerrar-h16" / "p12_estratos_B.csv")
    df_e1 = pd.read_csv(DIR_RES / "10-cerrar-h16" / "exploratorio" / "e10_1_diagnostico_riesgo.csv")
    df = df_p12.merge(
        df_e1[["rango", "ventana", "p", "rho_L_logB_logexc"]],
        on=["rango", "ventana", "p"],
        how="left",
    )
    # SE para 2 covariables: prácticamente idéntico a se_estrato (diferencia < 1e-6), calculamos exacto por seguridad
    # se_estrato = 1/sqrt(n - n_bins - 1); con 2 covariables: 1/sqrt(1/se_estrato^2 - 2)
    df["se_joint"] = 1.0 / np.sqrt((1.0 / (df["se_estrato"] ** 2)) - 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.7), dpi=150, sharey=True)
    series_plot = [
        ("rho_estrato", "se_estrato", "o-", "#c0392b", -0.24, "Within strata (baseline)"),
        ("rho_qB", "se_qB", "s-", "#2980b9", -0.12, r"Strata $\times$ quintiles of $B$"),
        ("rho_logB", "se_logB", "^--", "#16a085", 0.00, r"Strata + linear $\log B$"),
        ("rho_logdisc", "se_logdisc", "D-", "#8e44ad", 0.12, r"Strata + linear $\log |\Delta|$"),
        ("rho_L_logB_logexc", "se_joint", "P-", "#e67e22", 0.24, r"Strata + joint $\{\log B, \log(|\Delta|/N)\}$"),
    ]
    etiquetas_x = [
        "p=3\n(1–50k)",
        "p=5\n(1–50k)",
        "p=7\n(1–50k)",
        "p=3\n(50k–100k)",
        "p=5\n(50k–100k)",
        "p=7\n(50k–100k)",
    ]
    pares_xp = [
        ("1-49999", 3),
        ("1-49999", 5),
        ("1-49999", 7),
        ("50000-99999", 3),
        ("50000-99999", 5),
        ("50000-99999", 7),
    ]
    x_base = np.arange(len(pares_xp), dtype=float)

    for ax, rango, titulo in [
        (axes[0], 1, r"(A) Rank 1: $\rho(a_p, \log \Omega \mid L'(1), \text{strata}) \pm 2\,\mathrm{s.e.}$"),
        (axes[1], 2, r"(B) Rank 2: $\rho(a_p, \log \Omega \mid L''(1)/2, \text{strata}) \pm 2\,\mathrm{s.e.}$"),
    ]:
        sub_r = df[df["rango"] == rango]
        for col_rho, col_se, fmt, color, dx, label in series_plot:
            ys, errs = [], []
            for vent, pv in pares_xp:
                fila = sub_r[(sub_r["ventana"] == vent) & (sub_r["p"] == pv)].iloc[0]
                ys.append(float(fila[col_rho]))
                errs.append(2.0 * float(fila[col_se]))
            ax.errorbar(
                x_base + dx, ys, yerr=errs, fmt=fmt, color=color, ecolor=color,
                elinewidth=1.2, capsize=2.8, markersize=5.4, linewidth=1.45, label=label,
            )
        ax.axhline(0.0, color="#222222", linestyle="--", linewidth=0.9)
        ax.axvline(2.5, color="#888888", linestyle=":", linewidth=1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(etiquetas_x)
        ax.set_ylabel(r"Partial correlation $\rho(a_p, \log \Omega \mid L^{(r)}, \text{strata})$")
        ax.set_title(titulo)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.3)

    fig.suptitle(
        r"Within-stratum negative correlation: single controls ($B$, $\log |\Delta|$) and joint control $\{\log B, \log(|\Delta|/N)\}$",
        fontsize=11.8,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(DIR_OUT / "h16.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    fig1_escala()
    fig2_mecanismo()
    fig3_estratos()
    fig4_h16()
    print("4 figuras en inglés generadas en:", DIR_OUT)


if __name__ == "__main__":
    main()
