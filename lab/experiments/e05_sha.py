from __future__ import annotations

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab.data import load_bsd, one_per_class
from lab.derived import load_ap_cache
from lab.stats import profile_difference, null_profile_distribution

SEED = 20261121
N_PERM = 1000
N_PRIMES = 25

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "05-sha-murmurations"

WINDOWS = [
    ("10000-50000", 10_000, 50_000),
    ("50000-99999", 50_000, 99_999),
]


def _tamagawa_cat(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(len(arr), dtype=int)
    res[arr == 2] = 1
    res[(arr >= 3) & (arr <= 4)] = 2
    res[arr >= 5] = 3
    return res


def _torsion_cat(arr: np.ndarray) -> np.ndarray:
    res = np.zeros(len(arr), dtype=int)
    res[arr == 2] = 1
    res[arr >= 3] = 2
    return res


def _build_nulls(sub: pd.DataFrame, rank: int) -> dict[str, np.ndarray]:
    n = len(sub)
    b_l100 = pd.qcut(sub["l_value"], q=100, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_l10 = pd.qcut(sub["l_value"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_cond10 = pd.qcut(sub["conductor"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_om10 = pd.qcut(sub["omega"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
    b_tam = _tamagawa_cat(sub["tamagawa"].to_numpy(dtype=int))
    b_tor = _torsion_cat(sub["torsion"].to_numpy(dtype=int))

    nulls: dict[str, np.ndarray] = {
        "unconditioned": np.zeros(n, dtype=int),
        "cond_L_100": b_l100,
        "cond_L10_conductor10": b_l10 * 100 + b_cond10,
        "cond_L10_omega10": b_l10 * 100 + b_om10,
    }
    if rank == 1:
        b_reg10 = pd.qcut(sub["regulator"], q=10, labels=False, duplicates="drop").to_numpy(dtype=int)
        nulls["cond_L10_regulator10"] = b_l10 * 100 + b_reg10

    nulls["cond_L10_tamagawa"] = b_l10 * 10 + b_tam
    nulls["cond_L10_torsion"] = b_l10 * 10 + b_tor
    return nulls


def run_sha_murmurations() -> dict:
    t_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_bsd = load_bsd(n_max=99_999)
    sha_max_df = (
        df_bsd.groupby(["conductor", "isoclass"], as_index=False)["sha_an"]
        .max()
        .rename(columns={"sha_an": "sha_max_class"})
    )

    classes = one_per_class(df_bsd).merge(
        sha_max_df, on=["conductor", "isoclass"], how="left"
    )
    df_ap = load_ap_cache(n_primes=N_PRIMES)
    primes = np.array([int(p) for p in df_ap.columns], dtype=np.int32)

    rows_summary: list[dict[str, str | int | float]] = []
    rows_delta: list[dict[str, str | int | float]] = []
    rows_isogeny: list[dict[str, str | int | float]] = []

    plot_data: dict[tuple[int, str], dict[str, np.ndarray]] = {}

    for rank in (0, 1):
        for win_name, n_min, n_max in WINDOWS:
            sub = classes.loc[
                (classes["rank"] == rank)
                & (classes["conductor"] >= n_min)
                & (classes["conductor"] <= n_max)
            ].reset_index(drop=True)

            sha_c1 = np.rint(sub["sha_an"].to_numpy(dtype=float)).astype(int)
            mask_valid = (sha_c1 == 1) | (sha_c1 >= 4)
            sub = sub.loc[mask_valid].reset_index(drop=True)
            sha_c1 = sha_c1[mask_valid]

            group = sha_c1 >= 4
            n_sha1 = int(np.sum(~group))
            n_sha4 = int(np.sum(group))

            ap_sub = df_ap.loc[list(sub["label"])].to_numpy(dtype=np.int16)
            cond_sub = sub["conductor"].to_numpy(dtype=np.int32)
            good_sub = (cond_sub[:, None] % primes[None, :]) != 0

            delta_obs, rms_obs = profile_difference(ap_sub, good_sub, group)

            nulls_bins = _build_nulls(sub, rank=rank)

            deltas_uncond: np.ndarray | None = None
            deltas_L: np.ndarray | None = None

            for null_name, bins in nulls_bins.items():
                deltas_nulls, rms_nulls = null_profile_distribution(
                    ap_sub,
                    good_sub,
                    group,
                    bins,
                    n_perm=N_PERM,
                    seed=SEED,
                )
                mean_null = float(np.mean(rms_nulls))
                sd_null = float(np.std(rms_nulls, ddof=1))
                p_val = float((1 + int(np.sum(rms_nulls >= rms_obs))) / (N_PERM + 1))

                rows_summary.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "null": null_name,
                        "n_sha1": n_sha1,
                        "n_sha4": n_sha4,
                        "rms_obs": rms_obs,
                        "mean_null": mean_null,
                        "sd_null": sd_null,
                        "p": p_val,
                    }
                )

                if null_name == "unconditioned":
                    deltas_uncond = deltas_nulls
                elif null_name == "cond_L_100":
                    deltas_L = deltas_nulls

            assert deltas_uncond is not None and deltas_L is not None
            mean_d_uncond = deltas_uncond.mean(axis=0)
            sd_d_uncond = deltas_uncond.std(axis=0, ddof=1)
            z_uncond = (delta_obs - mean_d_uncond) / sd_d_uncond

            mean_d_L = deltas_L.mean(axis=0)
            sd_d_L = deltas_L.std(axis=0, ddof=1)
            z_L = (delta_obs - mean_d_L) / sd_d_L

            for j, p_prime in enumerate(primes):
                rows_delta.append(
                    {
                        "rank": rank,
                        "window": win_name,
                        "p": int(p_prime),
                        "delta": float(delta_obs[j]),
                        "z_unconditioned": float(z_uncond[j]),
                        "z_cond_L": float(z_L[j]),
                    }
                )

            plot_data[(rank, win_name)] = {
                "primes": primes,
                "delta": delta_obs,
                "mean_uncond": mean_d_uncond,
                "sd_uncond": sd_d_uncond,
                "mean_L": mean_d_L,
                "sd_L": sd_d_L,
            }

            # Isogeny sensitivity: maximum Sha in the class
            sha_max = np.rint(sub["sha_max_class"].to_numpy(dtype=float)).astype(int)
            group_max = sha_max >= 4
            n_changed = int(np.sum(group_max != group))
            n_sha1_max = int(np.sum(~group_max))
            n_sha4_max = int(np.sum(group_max))

            _, rms_obs_max = profile_difference(ap_sub, good_sub, group_max)
            _, rms_nulls_max = null_profile_distribution(
                ap_sub,
                good_sub,
                group_max,
                nulls_bins["cond_L_100"],
                n_perm=N_PERM,
                seed=SEED,
            )
            mean_null_max = float(np.mean(rms_nulls_max))
            sd_null_max = float(np.std(rms_nulls_max, ddof=1))
            p_val_max = float(
                (1 + int(np.sum(rms_nulls_max >= rms_obs_max))) / (N_PERM + 1)
            )

            rows_isogeny.append(
                {
                    "rank": rank,
                    "window": win_name,
                    "n_changed": n_changed,
                    "n_sha1_max": n_sha1_max,
                    "n_sha4_max": n_sha4_max,
                    "rms_obs_c1": rms_obs,
                    "rms_obs_max": rms_obs_max,
                    "mean_null_max": mean_null_max,
                    "sd_null_max": sd_null_max,
                    "p_cond_L_max": p_val_max,
                }
            )

    df_summary = pd.DataFrame(rows_summary)
    df_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    df_delta = pd.DataFrame(rows_delta)
    df_delta.to_csv(OUTPUT_DIR / "delta_p.csv", index=False)

    # Figure delta_p.png: one panel per rank (main window and replication, with ±2σ bands)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=150)

    for ax, rank, title in [
        (ax0, 0, "A. Rank 0: Δ(p) = ⟨aₚ⟩(Sha≥4) − ⟨aₚ⟩(Sha=1)"),
        (ax1, 1, "B. Rank 1: Δ(p) = ⟨aₚ⟩(Sha≥4) − ⟨aₚ⟩(Sha=1)"),
    ]:
        d_main = plot_data[(rank, "10000-50000")]
        d_rep = plot_data[(rank, "50000-99999")]
        p_arr = d_main["primes"]

        ax.fill_between(
            p_arr,
            d_main["mean_uncond"] - 2.0 * d_main["sd_uncond"],
            d_main["mean_uncond"] + 2.0 * d_main["sd_uncond"],
            color="#95a5a6",
            alpha=0.28,
            label="Unconditioned null ±2σ (10k–50k)",
        )
        ax.fill_between(
            p_arr,
            d_main["mean_L"] - 2.0 * d_main["sd_L"],
            d_main["mean_L"] + 2.0 * d_main["sd_L"],
            color="#f39c12",
            alpha=0.28,
            label="Null conditioned on L ±2σ (10k–50k)",
        )
        ax.plot(
            p_arr,
            d_main["mean_L"],
            "--",
            color="#d68910",
            linewidth=1.2,
            label="Null mean conditioned on L (10k–50k)",
        )
        ax.plot(
            p_arr,
            d_main["delta"],
            "o-",
            color="#1f5a99",
            markersize=4.5,
            linewidth=1.8,
            label="Observed Δ(p), N ∈ [10 000, 50 000]",
        )
        ax.plot(
            p_arr,
            d_rep["delta"],
            "s-",
            color="#c0392b",
            markersize=4.0,
            linewidth=1.5,
            alpha=0.85,
            label="Observed Δ(p), N ∈ [50 000, 99 999]",
        )
        ax.axhline(0.0, color="#222222", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Prime p (first 25 primes, p ≤ 97)")
        ax.set_ylabel("Δ(p) = ⟨aₚ⟩(Sha ≥ 4) − ⟨aₚ⟩(Sha = 1)")
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        "Effect of analytic Sha on the aₚ profile (one curve per class, p ∤ N)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delta_p.png", bbox_inches="tight")
    plt.close(fig)

    t_end = time.perf_counter()
    print(f"Total time e05_sha: {t_end - t_start:.2f} s")
    print("\n--- SUMMARY ---")
    print(df_summary.to_string(index=False))
    print("\n--- ISOGENY SENSITIVITY (maximum Sha in the class) ---")
    df_iso = pd.DataFrame(rows_isogeny)
    print(df_iso.to_string(index=False))

    return {
        "time_s": t_end - t_start,
        "summary": df_summary,
        "isogeny": df_iso,
    }


if __name__ == "__main__":
    run_sha_murmurations()
