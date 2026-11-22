"""Validate extracted phase fractions against values reported in the paper.

- normalise_reported(): meta.yaml values (percent or fraction) -> fractions.
- tolerance_pp(): which tolerance applies (config validation.tol_pp).
- evaluate(): abs_error_pp, sensitivity deltas, qc_flags on a manifest record.
- write_report(): reports/validation/summary.csv, summary.md, bland_altman.png.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.annotations import IGNORE

PHASE_KEYS = ("ferrite", "austenite", "sigma")
# Flags owned (recomputed) by evaluate(); other stages' flags are preserved.
VALIDATION_FLAGS = ("jpeg_source", "boundary_sensitive", "threshold_sensitive", "out_of_tolerance",
                    "missing_scale", "missing_reported", "missing_extraction")


def normalise_reported(rep: dict[str, Any] | None, sum_tolerance_pp: float = 2.0
                       ) -> tuple[dict[str, Any], list[str]]:
    """Return (PhaseFractions-shaped dict in fractions 0..1, flags).

    Unit: `reported.unit` (percent|fraction) when given; otherwise percent if any
    value > 1 or the values sum to more than 1.5, fraction if all <= 1 and they
    sum to ~1, else percent with flag reported_unit_ambiguous."""
    rep = dict(rep or {})
    vals = {k: rep.get(k) for k in PHASE_KEYS if rep.get(k) is not None}
    flags: list[str] = []
    unit = rep.get("unit")
    if unit not in ("percent", "fraction"):
        s = sum(float(v) for v in vals.values())
        if any(float(v) > 1 for v in vals.values()) or s > 1.5:
            unit = "percent"
        elif vals and abs(s - 1) <= sum_tolerance_pp / 100:
            unit = "fraction"
        else:
            unit = "percent"
            if vals:
                flags.append("reported_unit_ambiguous")
    div = 100.0 if unit == "percent" else 1.0
    out = {k: (float(rep[k]) / div if rep.get(k) is not None else None) for k in PHASE_KEYS}
    out["sd"] = float(rep["sd"]) / div if rep.get("sd") is not None else None
    out.update({k: rep.get(k) for k in ("method", "scope", "location_in_paper")})
    given = [v for k, v in out.items() if k in PHASE_KEYS and v is not None]
    if len(given) >= 2 and abs(sum(given) - 1) * 100 > sum_tolerance_pp and out["sigma"] is None \
            and len(given) == 2:
        flags.append("reported_sum_mismatch")
    return out, flags


def tolerance_pp(reported: dict[str, Any], image_type: str | None, condition: str | None,
                 vcfg: dict[str, Any]) -> tuple[float, str]:
    tol = vcfg.get("tol_pp", {})
    method, scope = reported.get("method"), reported.get("scope")
    if method == "EBSD" and (image_type or "").startswith("ebsd") and scope == "this_panel":
        t, rule = tol.get("ebsd_same_map", 5), "ebsd_same_map"
    elif method == "XRD":
        t, rule = tol.get("xrd", 10), "xrd"
    else:
        t, rule = tol.get("ia", 8), "ia"
    if condition and condition in vcfg.get("asbuilt_conditions", ["as_built"]):
        if tol.get("asbuilt_abs", 2) < t:
            t, rule = tol.get("asbuilt_abs", 2), f"{rule}+asbuilt_abs"
    return float(t), rule


def abs_errors_pp(extracted: dict[str, float], reported: dict[str, Any]) -> dict[str, float]:
    return {k: abs(float(extracted[k]) - float(reported[k])) * 100.0
            for k in PHASE_KEYS
            if reported.get(k) is not None and extracted.get(k) is not None
            and not np.isnan(extracted[k])}


def boundary_band_exclude(mask: np.ndarray, px: int) -> np.ndarray:
    """Set pixels within px of a phase boundary to 255 (sensitivity run)."""
    from scipy.ndimage import binary_dilation
    v = mask != IGNORE
    edge = np.zeros(mask.shape, bool)
    dh = (mask[:, 1:] != mask[:, :-1]) & v[:, 1:] & v[:, :-1]
    dv = (mask[1:, :] != mask[:-1, :]) & v[1:, :] & v[:-1, :]
    edge[:, 1:] |= dh
    edge[:, :-1] |= dh
    edge[1:, :] |= dv
    edge[:-1, :] |= dv
    if px > 1:
        edge = binary_dilation(edge, iterations=px - 1)
    out = mask.copy()
    out[edge] = IGNORE
    return out


def evaluate(rec: dict[str, Any], variants: dict[str, dict[str, Any]],
             vcfg: dict[str, Any]) -> dict[str, Any]:
    """Fill abs_error_pp, extracted.sensitivity and qc_flags on a record dict.

    variants: {name: {"fractions": {...}, "kind": "boundary"|"threshold"|"method"}}.
    """
    flags = [f for f in rec.get("qc_flags", []) if f not in VALIDATION_FLAGS]
    rep = rec.get("reported") or {}
    ext = rec.get("extracted")
    if rec.get("image", {}).get("is_jpeg"):
        flags.append("jpeg_source")
    if not rec.get("image", {}).get("pixel_size_um"):
        flags.append("missing_scale")
    if all(rep.get(k) is None for k in PHASE_KEYS):
        flags.append("missing_reported")
    rec["abs_error_pp"] = None
    if not ext or not ext.get("fractions"):
        flags.append("missing_extraction")
        rec["qc_flags"] = sorted(set(flags))
        return rec

    base = ext["fractions"]
    sens: dict[str, Any] = {}
    for name, v in variants.items():
        fr = v["fractions"]
        sens[name] = {"kind": v["kind"], "ferrite": fr.get("ferrite"),
                      "delta_pp": (fr["ferrite"] - base["ferrite"]) * 100.0
                      if fr.get("ferrite") is not None and not np.isnan(fr["ferrite"]) else None}
    ext["sensitivity"] = sens
    lim = float(vcfg.get("boundary_sensitive_pp", 2.0))

    def worst(kind):
        d = [abs(s["delta_pp"]) for s in sens.values() if s["kind"] == kind and s["delta_pp"] is not None]
        return max(d) if d else 0.0
    if worst("boundary") > lim:
        flags.append("boundary_sensitive")
    if worst("threshold") > lim:
        flags.append("threshold_sensitive")

    errs = abs_errors_pp(base, rep)
    if errs:
        rec["abs_error_pp"] = errs
        tol, rule = tolerance_pp(rep, rec.get("image", {}).get("image_type"), rec.get("condition"), vcfg)
        ext.setdefault("params", {})["tolerance_pp"] = tol
        ext["params"]["tolerance_rule"] = rule
        if max(errs.values()) > tol:
            flags.append("out_of_tolerance")
    rec["qc_flags"] = sorted(set(flags))
    return rec


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _row(rec: dict[str, Any], group_by: str) -> dict[str, Any]:
    rep, ext = rec.get("reported") or {}, (rec.get("extracted") or {})
    fr = ext.get("fractions") or {}
    phase = next((k for k in PHASE_KEYS if rep.get(k) is not None), None)
    r_pp = rep[phase] * 100 if phase else np.nan
    e_pp = fr[phase] * 100 if phase and fr.get(phase) is not None else np.nan
    group = rec.get("source", {}).get(group_by) or rec.get(group_by) \
        or rec["sample_id"].split("_f")[0]
    params = ext.get("params") or {}
    return {
        "sample_id": rec["sample_id"], "group": group,
        "image_type": rec.get("image", {}).get("image_type"),
        "source_format": "JPEG" if rec.get("image", {}).get("is_jpeg") else "PNG/other",
        "condition": rec.get("condition"), "reported_method": rep.get("method"),
        "reported_scope": rep.get("scope"), "phase": phase,
        "reported_pp": r_pp, "extracted_pp": e_pp, "error_pp": e_pp - r_pp,
        "abs_error_pp": abs(e_pp - r_pp), "tolerance_pp": params.get("tolerance_pp"),
        "tolerance_rule": params.get("tolerance_rule"),
        "ignored_fraction": ext.get("ignored_fraction"),
        "extraction_method": ext.get("method"),
        "pixel_size_um": rec.get("image", {}).get("pixel_size_um"),
        "qc_flags": ";".join(rec.get("qc_flags", [])),
    }


def _stats(df) -> dict[str, Any]:
    d = df.dropna(subset=["error_pp"])
    if d.empty:
        return {"n": 0, "MAE_pp": np.nan, "bias_pp": np.nan, "max_abs_pp": np.nan, "within_tol": 0}
    within = (d["abs_error_pp"] <= d["tolerance_pp"].astype(float)).sum()
    return {"n": len(d), "MAE_pp": d["abs_error_pp"].mean(), "bias_pp": d["error_pp"].mean(),
            "max_abs_pp": d["abs_error_pp"].max(), "within_tol": int(within)}


def _md_table(df, by: str | None) -> str:
    import pandas as pd
    rows = [{"group": "all", **_stats(df)}] if by is None else \
        [{"group": str(k), **_stats(g)} for k, g in df.groupby(by, dropna=False)]
    t = pd.DataFrame(rows)
    head = "| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n"
    body = "".join("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |\n"
                   for r in t.itertuples(index=False))
    return head + body


def bland_altman(df, out_path: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.dropna(subset=["error_pp"])
    if d.empty:
        return None
    # Fixed categorical order (identity), plus marker shape so colour is never alone.
    style = {"ebsd_phase": ("#2a78d6", "o"), "ebsd_phase_iq": ("#eb6834", "s"),
             "om": ("#1baf7a", "^"), "sem": ("#eda100", "D")}
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    mean = (d["reported_pp"] + d["extracted_pp"]) / 2
    bias, sd = d["error_pp"].mean(), d["error_pp"].std(ddof=1) if len(d) > 1 else 0.0
    for lvl, ls in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
        ax.axhline(lvl, color="#6b6b66", lw=1, ls=ls, zorder=1)
    ax.text(1.0, bias, f" bias {bias:+.2f}", transform=ax.get_yaxis_transform(), va="center",
            fontsize=8, color="#3d3d3a")
    if len(d) > 1:
        ax.text(1.0, bias + 1.96 * sd, f" +1.96 SD {bias + 1.96 * sd:+.2f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8, color="#3d3d3a")
        ax.text(1.0, bias - 1.96 * sd, f" -1.96 SD {bias - 1.96 * sd:+.2f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8, color="#3d3d3a")
    for it, g in d.groupby("image_type"):
        colour, marker = style.get(it, ("#6b6b66", "o"))
        for fmt, gg in g.groupby("source_format"):
            jpeg = fmt == "JPEG"
            ax.scatter(mean[gg.index], gg["error_pp"], s=56, marker=marker, zorder=3,
                       facecolors="white" if jpeg else colour, edgecolors=colour, linewidths=1.6,
                       label=f"{it} ({'JPEG' if jpeg else 'PNG'})")
    ax.set_xlabel("Mean of reported and extracted (pp)")
    ax.set_ylabel("Extracted − reported (pp)")
    ax.set_title("Phase fraction agreement (Bland–Altman)", fontsize=10, loc="left")
    ax.grid(True, color="#e4e3dd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png")
    plt.close(fig)
    return out_path


def write_report(records: list[dict[str, Any]], out_dir: str | Path, group_by: str = "paper_doi",
                 config_hash: str | None = None) -> dict[str, Path]:
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_row(r, group_by) for r in records])
    csv = out_dir / "summary.csv"
    df.to_csv(csv, index=False)
    flags = pd.Series([f for r in records for f in r.get("qc_flags", [])]).value_counts()
    md = [
        "# Phase-fraction validation summary\n",
        f"Config hash: `{config_hash}`  \nRecords: {len(df)}  \n"
        "Error = extracted − reported, in percentage points (pp), on the first reported phase "
        "(ferrite, else austenite, else sigma).\n",
        "## Overall\n", _md_table(df, None),
        "\n## By image type\n", _md_table(df, "image_type"),
        "\n## By source format\n", _md_table(df, "source_format"),
        f"\n## By {group_by}\n", _md_table(df, "group"),
        "\n## QC flags\n",
        "".join(f"- `{k}`: {v}\n" for k, v in flags.items()) or "- none\n",
        "\n## Per sample\n",
        "| sample_id | phase | reported | extracted | error | tol | flags |\n|---|---|---|---|---|---|---|\n",
        "".join(f"| {r.sample_id} | {r.phase} | {r.reported_pp:.2f} | {r.extracted_pp:.2f} | "
                f"{r.error_pp:+.2f} | {r.tolerance_pp} | {r.qc_flags} |\n"
                for r in df.itertuples()),
    ]
    mdp = out_dir / "summary.md"
    mdp.write_text("".join(md), encoding="utf-8")
    out = {"csv": csv, "md": mdp}
    ba = bland_altman(df, out_dir / "bland_altman.png")
    if ba:
        out["bland_altman"] = ba
    return out
