"""Diagnostics for the Phase-2 low-FPR metric.

These tools answer questions the published baselines do NOT address:
  * Where does the score actually come from?           -> band_contribution
  * Where does an improvement pay off most?            -> marginal_weight
  * How small a difference is measurable at all?       -> measurement_precision
  * How much does the metric depend on its grid?       -> grid_sensitivity

All numbers below are produced from the metric definition alone; none of them
requires access to the (hidden) competition test labels.
"""
from __future__ import annotations

import numpy as np

from .metric import fpr_grid, phase2_score, tpr_on_grid

BANDS = ((0.001, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05))


def band_contribution(labels, scores, bands=BANDS) -> list[dict]:
    """Share of the TOTAL TPR mass coming from each FPR band.

    Note this is NOT the same as "where the metric cares": see marginal_weight.
    """
    g = tpr_on_grid(labels, scores)
    grid = fpr_grid()
    tot = g.sum()
    out = []
    for lo, hi in bands:
        m = (grid >= lo) & (grid < hi)
        out.append({"band": [lo, hi], "points": int(m.sum()),
                    "tpr_sum": float(g[m].sum()),
                    "share_of_total": float(g[m].sum() / tot) if tot else 0.0})
    return out


def marginal_weight(bands=BANDS) -> list[dict]:
    """d(score) for a uniform +delta improvement of TPR inside each band.

    Because the grid is log-spaced, the low-FPR end holds MORE points and therefore
    a larger marginal weight -- even though its TPR *values* are smaller.
    """
    grid = fpr_grid()
    out = []
    for lo, hi in bands:
        m = (grid >= lo) & (grid < hi)
        out.append({"band": [lo, hi], "points": int(m.sum()),
                    "d_score_per_0.01_tpr": float(m.sum() * 0.01 / len(grid))})
    return out


def measurement_precision(labels, scores, reps: int = 20, seed: int = 0) -> dict:
    """Monte-Carlo spread of the score under resampling of the score distribution.

    A difference smaller than 2*std is not measurable with this test set size.
    """
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    sc = np.asarray(scores)
    vals = np.array([phase2_score(lab, sc[rng.permutation(len(sc))]) for _ in range(reps)])
    return {"score": float(phase2_score(lab, sc)),
            "std": float(vals.std()),
            "measurable_delta_2sigma": float(2 * vals.std())}


def grid_sensitivity(labels, scores) -> dict:
    """How much the score moves if the grid is changed (the metric's backbone)."""
    s_log = phase2_score(labels, scores, grid=fpr_grid(spacing="log"))
    s_lin = phase2_score(labels, scores, grid=fpr_grid(spacing="linear"))
    return {"log_spaced (official)": s_log, "linear_spaced": s_lin,
            "relative_change": float((s_lin - s_log) / s_log) if s_log else float("nan")}


def body_sensitivity(labels, scores) -> dict:
    """How much the score moves if only the interpolation details change."""
    base = phase2_score(labels, scores)
    return {"official": base,
            "drop_intermediate=False": phase2_score(labels, scores, drop_intermediate=False),
            "step (no interpolation)": phase2_score(labels, scores, interp="step"),
            "log-space interpolation": phase2_score(labels, scores, interp="log")}
