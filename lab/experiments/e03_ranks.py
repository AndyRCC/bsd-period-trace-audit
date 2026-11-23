from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "03-rank-distribution"


def run_ranks() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_bsd = load_bsd(n_max=99_999)
    classes = one_per_class(df_bsd)

    rows: list[dict[str, str | int | float]] = []
    for b in range(10):
        n_min = 1 if b == 0 else b * 10_000
        n_max = b * 10_000 + 9_999
        block_label = f"{n_min}-{n_max}"

        sub = classes.loc[(classes["conductor"] >= n_min) & (classes["conductor"] <= n_max)]
        n_classes = len(sub)
        counts = sub["rank"].value_counts().to_dict()
        r0 = int(counts.get(0, 0))
        r1 = int(counts.get(1, 0))
        r2 = int(counts.get(2, 0))
        r3 = int(counts.get(3, 0))

        if b == 0:
            expected_b0 = (16_450, 19_622, 1_969, 1)
            if (r0, r1, r2, r3) != expected_b0 or n_classes != 38_042:
                raise RuntimeError(
                    f"Counts for block 1-9999 do not match the published ones: "
                    f"obtained={(r0, r1, r2, r3)} (n={n_classes}), expected={expected_b0}"
                )

        rank_vals = sub["rank"].to_numpy(dtype=float)
        mean_rank = float(np.mean(rank_vals))
        standard_error = float(np.std(rank_vals, ddof=1) / np.sqrt(n_classes))

        odd_vals = (sub["rank"].to_numpy(dtype=int) % 2 == 1).astype(float)
        prop_odd = float(np.mean(odd_vals))
        err_odd = float(np.sqrt(prop_odd * (1.0 - prop_odd) / n_classes))

        rows.append(
            {
                "block": block_label,
                "n_min": n_min,
                "n_max": n_max,
                "n_classes": n_classes,
                "r0": r0,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "prop_r0": r0 / n_classes,
                "prop_r1": r1 / n_classes,
                "prop_r2": r2 / n_classes,
                "prop_r3": r3 / n_classes,
                "mean_rank": mean_rank,
                "standard_error": standard_error,
                "prop_parity_odd": prop_odd,
                "parity_standard_error": err_odd,
            }
        )

    table = pd.DataFrame(rows)
    path_csv = OUTPUT_DIR / "table.csv"
    table.to_csv(path_csv, index=False)

    # Two-panel figure
    x = np.arange(len(table))
    labels_x = [f"{b * 10}k–{b * 10 + 10}k" if b > 0 else "1–10k" for b in range(10)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)

    ax1.errorbar(
        x,
        table["mean_rank"],
        yerr=table["standard_error"],
        fmt="o-",
        color="#1f5a99",
        ecolor="#1f5a99",
        elinewidth=1.3,
        capsize=4,
        markersize=6,
        linewidth=1.8,
        label="Mean rank ± SE",
    )
    ax1.axhline(0.5, color="#888888", linestyle="--", linewidth=1.1, label="Asymptotic prediction (0.5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_x, rotation=35, ha="right")
    ax1.set_xlabel("Conductor block N")
    ax1.set_ylabel("Mean rank per isogeny class")
    ax1.set_title("A. Mean rank per block of 10 000 conductors")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.errorbar(
        x,
        table["prop_parity_odd"],
        yerr=table["parity_standard_error"],
        fmt="s-",
        color="#b8332a",
        ecolor="#b8332a",
        elinewidth=1.3,
        capsize=4,
        markersize=6,
        linewidth=1.8,
        label="Odd-rank proportion (ε = −1) ± SE",
    )
    ax2.axhline(0.5, color="#222222", linestyle="--", linewidth=1.2, label="Equidistribution (0.5)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_x, rotation=35, ha="right")
    ax2.set_xlabel("Conductor block N")
    ax2.set_ylabel("Proportion of classes with odd rank")
    ax2.set_title("B. Proportion of odd parity (ε = −1) per block")
    ax2.set_ylim(0.48, 0.56)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.suptitle(
        "Rank and parity distribution in Cremona's tables (N ≤ 99 999, one curve per class)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path_fig = OUTPUT_DIR / "figure.png"
    fig.savefig(path_fig, bbox_inches="tight")
    plt.close(fig)

    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    run_ranks()
