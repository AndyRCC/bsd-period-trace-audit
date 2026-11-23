from __future__ import annotations

import os
from pathlib import Path
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GP_PATH_TXT = Path(__file__).resolve().parent / "gp_path.txt"

INVARIANT_COLUMNS = [
    "label",
    "rank_an",
    "l_value",
    "omega",
    "tamagawa",
    "torsion",
    "regulator",
    "root_number",
]


def _get_gp_exe() -> str:
    env_gp = os.environ.get("GP_EXE", "").strip()
    if env_gp:
        return env_gp
    if GP_PATH_TXT.exists():
        first = GP_PATH_TXT.read_text(encoding="utf-8").splitlines()[0].strip()
        if first:
            os.environ["GP_EXE"] = first
            return first
    raise FileNotFoundError(
        "Path to gp.exe not found in GP_EXE or in lab/gp_path.txt"
    )


def _projective_point_to_gp(point_str: str) -> str:
    clean = point_str.strip()
    if not (clean.startswith("[") and clean.endswith("]")):
        raise ValueError(f"Invalid point format: {point_str!r}")
    parts = clean[1:-1].split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 projective coordinates in {point_str!r}")
    x_int = int(parts[0])
    y_int = int(parts[1])
    z_int = int(parts[2])
    if z_int == 0:
        return "[0]"
    return f"[{x_int}/{z_int},{y_int}/{z_int}]"


def _has_gp_error(text: str) -> bool:
    for line in text.splitlines():
        if "***" in line:
            clean = line.strip()
            if clean.startswith("***   Warning: new maximum stack size") or clean.startswith(
                "***   Warning: increasing stack size"
            ):
                continue
            return True
    return False


def run_gp(script: str, timeout_s: int = 900) -> str:
    gp_exe = _get_gp_exe()
    script_full = f"default(parisizemax, 2^30);\n{script}\nquit;\n"
    try:
        proc = subprocess.run(
            [gp_exe, "-f", "-q"],
            input=script_full,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"PARI/GP exceeded the time limit of {timeout_s} seconds"
        ) from exc

    output = proc.stdout or ""
    errors = proc.stderr or ""
    if _has_gp_error(output) or _has_gp_error(errors) or proc.returncode != 0:
        detail = (errors + "\n" + output).strip()
        raise RuntimeError(f"PARI/GP execution error (rc={proc.returncode}):\n{detail}")

    return output


def pari_invariants(
    curves: list[tuple[str, tuple[int, ...], list[str]]],
) -> pd.DataFrame:
    if not curves:
        return pd.DataFrame(columns=INVARIANT_COLUMNS)

    lines_gp: list[str] = [
        "default(realprecision, 19);",
        "calc_inv(et, ai, pts) = {",
        "  my(E = ellinit(ai), ar, r, lval, om, cp, tor, reg, eps);",
        "  foreach(pts, P, if(!ellisoncurve(E, P), error(\"Point not on the curve: \", et, \" \", P)));",
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

    for label, ainvs, gens in curves:
        if len(ainvs) != 5:
            raise ValueError(f"Expected 5 Weierstrass coefficients for {label}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        pts_gp = "[" + ",".join(_projective_point_to_gp(p) for p in gens) + "]"
        tag_safe = label.replace("\\", "\\\\").replace('"', '\\"')
        lines_gp.append(f'calc_inv("{tag_safe}", {ai_gp}, {pts_gp});')

    output = run_gp("\n".join(lines_gp))

    records: list[dict[str, str | int | float]] = []
    for line in output.splitlines():
        if not line.startswith("RES\t"):
            continue
        _, tag, r_s, l_s, om_s, cp_s, tor_s, reg_s, eps_s = line.split("\t")
        records.append(
            {
                "label": tag,
                "rank_an": int(r_s),
                "l_value": float(l_s.replace(" ", "")),
                "omega": float(om_s.replace(" ", "")),
                "tamagawa": int(cp_s),
                "torsion": int(tor_s),
                "regulator": float(reg_s.replace(" ", "")),
                "root_number": int(eps_s),
            }
        )

    if len(records) != len(curves):
        raise RuntimeError(
            f"Expected {len(curves)} rows from PARI/GP, but received {len(records)}"
        )

    return pd.DataFrame(records, columns=INVARIANT_COLUMNS)


def ap_coefficients(
    curves: list[tuple[str, tuple[int, ...]]],
    n_primes: int = 1000,
) -> pd.DataFrame:
    if n_primes <= 0:
        raise ValueError("n_primes must be positive")

    lines_gp: list[str] = [
        f"my_primes = primes({int(n_primes)});",
        'print("PRIMES\\t", strjoin([Str(p) | p <- my_primes], "\\t"));',
    ]

    for label, ainvs in curves:
        if len(ainvs) != 5:
            raise ValueError(f"Expected 5 Weierstrass coefficients for {label}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        tag_safe = label.replace("\\", "\\\\").replace('"', '\\"')
        lines_gp.append(
            f'my(E = ellinit({ai_gp}, 1)); print("AP\\t{tag_safe}\\t", strjoin([Str(ellap(E, p)) | p <- my_primes], "\\t"));'
        )

    output = run_gp("\n".join(lines_gp))

    primes: list[int] = []
    labels: list[str] = []
    rows_ap: list[list[int]] = []

    for line in output.splitlines():
        if line.startswith("PRIMES\t"):
            parts = line.split("\t")[1:]
            primes = [int(p) for p in parts]
        elif line.startswith("AP\t"):
            parts = line.split("\t")
            labels.append(parts[1])
            rows_ap.append([int(x) for x in parts[2:]])

    if not ValidCurvesOrPrimes(curves, primes, rows_ap):
        raise RuntimeError("Incomplete PARI/GP output in ap_coefficients")

    df = pd.DataFrame(rows_ap, index=pd.Index(labels, name="label"), columns=primes)
    return df


def ValidCurvesOrPrimes(
    curves: list[tuple[str, tuple[int, ...]]],
    primes: list[int],
    rows_ap: list[list[int]],
) -> bool:
    return len(primes) > 0 and len(rows_ap) == len(curves)


def ap_averages(
    curves: list[tuple[str, tuple[int, ...], int]],
    p_max: int,
    exclude_bad: bool = False,
) -> pd.DataFrame:
    colnames = ["group", "p", "n", "acc", "mean"]
    if not curves or p_max < 2:
        return pd.DataFrame(columns=colnames)

    groups_unique = sorted({int(g) for _, _, g in curves})
    g_map = {g: idx + 1 for idx, g in enumerate(groups_unique)}
    n_groups = len(groups_unique)
    groups_gp = "[" + ",".join(str(g) for g in groups_unique) + "]"

    lines_gp: list[str] = [
        f"P = primes([2, {int(p_max)}]);",
        "nP = #P;",
        f"groups = {groups_gp};",
        f"nG = {n_groups};",
        "S = matrix(nG, nP);",
        "C = matrix(nG, nP);",
    ]

    if exclude_bad:
        lines_gp.extend(
            [
                "accum(ig, ai) = {",
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
        lines_gp.extend(
            [
                "cntG = vector(nG);",
                "accum(ig, ai) = {",
                "  my(E = ellinit(ai, 1));",
                "  cntG[ig]++;",
                "  for(j = 1, nP, S[ig, j] += ellap(E, P[j]));",
                "};",
            ]
        )

    for label, ainvs, group in curves:
        if len(ainvs) != 5:
            raise ValueError(f"Expected 5 Weierstrass coefficients for {label}")
        ig = g_map[int(group)]
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        lines_gp.append(f"accum({ig}, {ai_gp});")

    if not exclude_bad:
        lines_gp.append("for(ig = 1, nG, for(j = 1, nP, C[ig, j] = cntG[ig]));")

    lines_gp.append(
        'for(ig = 1, nG, for(j = 1, nP, printf("AVG\\t%d\\t%d\\t%d\\t%d\\n", groups[ig], P[j], C[ig, j], S[ig, j])));'
    )

    output = run_gp("\n".join(lines_gp))

    records: list[dict[str, int | float]] = []
    for line in output.splitlines():
        if not line.startswith("AVG\t"):
            continue
        _, g_s, p_s, n_s, s_s = line.split("\t")
        n_val = int(n_s)
        s_val = int(s_s)
        mean_val = (s_val / n_val) if n_val > 0 else float("nan")
        records.append(
            {
                "group": int(g_s),
                "p": int(p_s),
                "n": n_val,
                "acc": s_val,
                "mean": mean_val,
            }
        )

    if not records:
        raise RuntimeError("Empty PARI/GP output in ap_averages")

    return pd.DataFrame(records, columns=colnames)


def _conductor_from_label(label: str) -> int:
    n_chars = 0
    for ch in label:
        if ch.isdigit():
            n_chars += 1
        else:
            break
    return int(label[:n_chars]) if n_chars > 0 else 0


def bad_euler_factor(
    curves: list[tuple[str, tuple[int, ...]]],
) -> pd.DataFrame:
    colnames = ["label", "log_B", "n_bad"]
    if not curves:
        return pd.DataFrame(columns=colnames)

    lines_gp: list[str] = [
        "default(realprecision, 19);",
        "calc_eb(et, ai, nh) = {",
        "  my(E = ellinit(ai, 1), M = if(nh > 1 && E.disc % nh == 0, nh, abs(E.disc)), P = factor(M)[, 1], B = 1, ap);",
        "  foreach(P, p, ap = ellap(E, p); B *= p / (p - ap));",
        '  printf("EB\\t%s\\t%.17g\\t%d\\n", et, if(B == 1, 0.0, log(B)), #P);',
        "};",
    ]

    for label, ainvs in curves:
        if len(ainvs) != 5:
            raise ValueError(f"Expected 5 Weierstrass coefficients for {label}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        tag_safe = label.replace("\\", "\\\\").replace('"', '\\"')
        nh = _conductor_from_label(label)
        lines_gp.append(f'calc_eb("{tag_safe}", {ai_gp}, {nh});')

    output = run_gp("\n".join(lines_gp))

    records: list[dict[str, str | int | float]] = []
    for line in output.splitlines():
        if not line.startswith("EB\t"):
            continue
        _, tag, logb_s, nm_s = line.split("\t")
        records.append(
            {
                "label": tag,
                "log_B": float(logb_s.replace(" ", "")),
                "n_bad": int(nm_s),
            }
        )

    if len(records) != len(curves):
        raise RuntimeError(
            f"Expected {len(curves)} rows in bad_euler_factor, received {len(records)}"
        )

    return pd.DataFrame(records, columns=colnames)


def reduction_types(
    curves: list[tuple[str, tuple[int, ...]]],
) -> pd.DataFrame:
    colnames = ["label", "n_split", "n_no_split", "n_additive"]
    if not curves:
        return pd.DataFrame(columns=colnames)

    lines_gp: list[str] = [
        "calc_tr(et, ai, nh) = {",
        "  my(E = ellinit(ai, 1), M = if(nh > 1 && E.disc % nh == 0, nh, abs(E.disc)), P = factor(M)[, 1], np = 0, nnp = 0, na = 0, ap);",
        "  foreach(P, q, ap = ellap(E, q); if(ap == 1, np++, if(ap == -1, nnp++, if(ap == 0, na++, error(\"unexpected ap at a bad prime: \", et, \" q=\", q, \" ap=\", ap)))));",
        '  printf("TR\\t%s\\t%d\\t%d\\t%d\\n", et, np, nnp, na);',
        "};",
    ]

    for label, ainvs in curves:
        if len(ainvs) != 5:
            raise ValueError(f"Expected 5 Weierstrass coefficients for {label}")
        ai_gp = f"[{ainvs[0]},{ainvs[1]},{ainvs[2]},{ainvs[3]},{ainvs[4]}]"
        tag_safe = label.replace("\\", "\\\\").replace('"', '\\"')
        nh = _conductor_from_label(label)
        lines_gp.append(f'calc_tr("{tag_safe}", {ai_gp}, {nh});')

    output = run_gp("\n".join(lines_gp))

    records: list[dict[str, str | int]] = []
    for line in output.splitlines():
        if not line.startswith("TR\t"):
            continue
        _, tag, np_s, nnp_s, na_s = line.split("\t")
        records.append(
            {
                "label": tag,
                "n_split": int(np_s),
                "n_no_split": int(nnp_s),
                "n_additive": int(na_s),
            }
        )

    if len(records) != len(curves):
        raise RuntimeError(
            f"Expected {len(curves)} rows in reduction_types, received {len(records)}"
        )

    return pd.DataFrame(records, columns=colnames)



