# Period–trace correlations among elliptic curves over Q — an independent audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24450625.svg)](https://doi.org/10.5281/zenodo.24450625)

**Author:** Andy Casafranca · **Version:** 1.0.1 (November 2026) · **DOI:** [10.5281/zenodo.24450625](https://doi.org/10.5281/zenodo.24450625) (all versions; v1.0 is [10.5281/zenodo.24450626](https://doi.org/10.5281/zenodo.24450626))

This repository contains the code, results and preprint of an independent
audit of recent claims (D. Wachs, arXiv:2603.04604 and arXiv:2603.22807, October 2026 versions).
Those claims concern correlations between the Frobenius traces $a_p$ of elliptic curves over
$\mathbb{Q}$ and the invariants in the Birch–Swinnerton-Dyer (BSD) formula.

- **Preprint:** [`paper/preprint.md`](paper/preprint.md)
- **Spanish summary:** [`paper/summary_es.md`](paper/summary_es.md)

## Main findings (see the preprint for numbers and caveats)

1. The positive partial correlation between $a_p$ and the real period $\Omega$ at fixed
   $L(E,1)$ replicates with one curve per isogeny class.
2. At rank 0, the BSD identity routes this correlation through the Tamagawa product. The
   bad-prime Euler factor $B$ then acts as a collider: fixing $L(1)$ reverses the sign of the
   $a_p$–Tamagawa association.
3. At ranks 1 and 2, the correlation reverses sign inside strata of
   (Tamagawa, torsion, Sha): a Simpson / composition effect.
4. The negative within-stratum correlation combines that collider with a coupling of $a_p$ to
   the discriminant excess $\log(|\Delta|/N)$.
5. A documentation error in Cremona's `ecdata` file format for `allgens`, reported upstream as
   [JohnCremona/ecdata#7](https://github.com/JohnCremona/ecdata/issues/7) and corrected by
   [JohnCremona/ecdata#8](https://github.com/JohnCremona/ecdata/pull/8), merged on 23 November 2026.

This is empirical work on published data. **It does not bear on a proof of the BSD conjecture.**

## Repository layout

| Path | Contents |
|---|---|
| `lab/` | Python package: data loading, PARI/GP interface, statistics, experiments (`lab/experiments/e02…e10`), figure script, tests (`lab/tests/`) |
| `results/NN-*/` | Output of each step: CSV tables and figures |
| `paper/` | Preprint, Spanish summary and figures |
| `data/MANIFEST.md` | URLs and SHA-256 hashes of the input data files |

The raw Cremona data (`data/cremona/`, about 120 MB) and the derived caches
(`data/derived/`, about 67 MB) are **not** included; the steps below regenerate them.

## Reproducing the results

Requirements: Python 3.12, and [PARI/GP](https://pari.math.u-bordeaux.fr/) 2.19.1 (or compatible).

1. **Point the code to `gp`.** Set the environment variable `GP_EXE`, or write the full path to
   the `gp` executable on the first line of `lab/gp_path.txt`.
2. **Create the environment.**
   ```
   python -m venv .venv
   .venv/Scripts/python -m pip install -r lab/requirements.txt
   ```
   (On Linux or macOS use `.venv/bin/python`.)
3. **Download Cremona's tables** (30 files, conductors below 100 000), then check their hashes
   against `data/MANIFEST.md`.
   ```
   .venv/Scripts/python -c "from lab.data import download; download()"
   ```
4. **Run the test suite** (40 tests). It also builds the cache of $a_p$.
   ```
   .venv/Scripts/python -m pytest lab/tests -q
   ```
5. **Run the experiments in order.** `e07` builds `data/derived/bad_euler.csv` and `e10`
   builds `data/derived/reduction_types.csv`.
   ```
   .venv/Scripts/python -m lab.experiments.e02_audit
   ...
   .venv/Scripts/python -m lab.experiments.e10_h16
   ```
   The exploratory modules `e08_exploratory`, `e09_exploratory` and `e10_exploratory` are
   optional. All random choices use the seed `20261121`.
6. **Regenerate the figures of the preprint.**
   ```
   .venv/Scripts/python -m lab.experiments.paper_figures
   ```

Each experiment writes to `results/NN-*/` and takes a few minutes or less on a laptop.

## Use of AI tools

AI assistants (Claude, by Anthropic, and Gemini, by Google) were used to help write code, design
and run the analyses, and draft the text. The author directed the project, supervised every step,
checked the results and is responsible for all claims.

## How to cite

See [`CITATION.cff`](CITATION.cff).

## License

- Code (`lab/`): MIT License, see [`LICENSE`](LICENSE).
- Text, results and figures: CC BY 4.0, see [`LICENSE-docs.md`](LICENSE-docs.md).
- Cremona's data are not redistributed here; see [ecdata](https://github.com/JohnCremona/ecdata)
  for their terms.
