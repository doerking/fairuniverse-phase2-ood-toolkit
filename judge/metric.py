"""Phase-2 OoD detection metric — reimplementation from the published specification.

Spec source (public):
  https://github.com/FAIR-Universe/Cosmology_Challenge  ->  pages/Phase2/evaluation.md
  Leaderboard score = (1/N) * sum_i TPR(FPR_i),
  with N = 100 logarithmically spaced FPRs in [0.001, 0.05].

Why reimplement: the organizers' scoring program is not licensed for redistribution,
but the formula IS published.  This module implements the formula exactly and is
numerically verified against the official `scoring_program/score.py`
(see tests/test_all.py, case `metric_matches_official`).

IMPORTANT geometry notes (measured, see docs/FINDINGS.md):
  * The grid is LOG-spaced, not linear.  Switching to a linear grid raises the
    score by 18%-96% on synthetic cases -> the metric is *grid-defined*.
  * Interpolation details (drop_intermediate, step vs linear) change the score
    by < 1e-4 -> the metric is *robust to its body*, *sensitive to its backbone*.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve

FPR_MIN = 0.001
FPR_MAX = 0.05
N_POINTS = 100
RANDOM_BASELINE = 0.0128          # organizers' published random score


def fpr_grid(n: int = N_POINTS, fpr_min: float = FPR_MIN, fpr_max: float = FPR_MAX,
             spacing: str = "log") -> np.ndarray:
    """The 100 evaluation points. `spacing='linear'` is provided for diagnostics only."""
    if spacing == "log":
        return np.logspace(np.log10(fpr_min), np.log10(fpr_max), n)
    return np.linspace(fpr_min, fpr_max, n)


def tpr_on_grid(labels, scores, grid=None, drop_intermediate: bool = True,
                interp: str = "linear") -> np.ndarray:
    """TPR evaluated on the metric grid.

    interp: 'linear'  -> np.interp in FPR (the official behaviour)
            'log'     -> linear in log(FPR)
            'step'    -> right-continuous staircase (no interpolation)
    """
    grid = fpr_grid() if grid is None else np.asarray(grid)
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=drop_intermediate)
    if interp == "step":
        idx = (np.searchsorted(fpr, grid, side="right") - 1).clip(0, len(tpr) - 1)
        return tpr[idx]
    if interp == "log":
        return np.interp(np.log(grid), np.log(fpr), tpr)
    return np.interp(grid, fpr, tpr)


def phase2_score(labels, scores, **kw) -> float:
    """The leaderboard score. Larger = better OoD detection at low FPR."""
    return float(np.mean(tpr_on_grid(labels, scores, **kw)))


def random_reference(n_test: int = 10_000, n_ood: int | None = None, reps: int = 200,
                     seed: int = 0) -> dict:
    """Mean +/- std of the score under random scores. Useful as a sanity floor."""
    rng = np.random.default_rng(seed)
    n_ood = n_test // 5 if n_ood is None else n_ood
    lab = np.r_[np.ones(n_ood, int), np.zeros(n_test - n_ood, int)]
    vals = np.array([phase2_score(lab, rng.random(n_test)) for _ in range(reps)])
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "theory": float(fpr_grid().mean())}
