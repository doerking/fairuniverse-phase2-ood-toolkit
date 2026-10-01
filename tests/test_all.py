#!/usr/bin/env python3
"""Self-tests for the Phase-2 OoD toolkit.

Run:  python tests/test_all.py
Exits non-zero on failure.  No pytest required.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from judge import (phase2_score, fpr_grid, tpr_on_grid, band_contribution,
                   marginal_weight, grid_sensitivity, body_sensitivity)
from oodgen import spatial_bias, baryon_bias, load_transfer_grid, ell_grid, comoving_hmpc

FAILS = []


def chk(name, cond, detail=""):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    print("=" * 74)
    print("Phase-2 OoD toolkit - self tests")
    print("=" * 74)

    # ---------- 1. metric: analytic anchors ----------
    print("\n[1] metric anchors")
    lab = np.r_[np.ones(50, int), np.zeros(50, int)]
    chk("perfect separation -> 1.0",
        abs(phase2_score(lab, np.r_[np.ones(50), np.zeros(50)]) - 1.0) < 1e-12)
    chk("inverted -> 0.0",
        abs(phase2_score(lab, np.r_[np.zeros(50), np.ones(50)])) < 1e-12)

    g = fpr_grid()
    chk("grid = 100 log-spaced points in [0.001, 0.05]",
        len(g) == 100 and abs(g[0] - 0.001) < 1e-12 and abs(g[-1] - 0.05) < 1e-12)
    chk("random baseline ≈ arithmetic mean of the grid",
        abs(g.mean() - 0.0127) < 0.001, f"grid mean = {g.mean():.5f}")

    rng = np.random.default_rng(3)
    N = 10_000
    lab_big = np.r_[np.ones(N // 5, int), np.zeros(N - N // 5, int)]
    vals = [phase2_score(lab_big, rng.random(N)) for _ in range(60)]
    chk("random scores reproduce the published 0.0128 (N=10k)",
        abs(np.mean(vals) - 0.0128) < 0.004, f"simulated {np.mean(vals):.5f}")

    # the random baseline is SAMPLE-SIZE dependent: small N biases it upward,
    # because the ROC staircase quantises TPR to 1/n_ood.
    small = [phase2_score(np.r_[np.ones(50, int), np.zeros(50, int)], rng.random(100))
             for _ in range(60)]
    chk("small N biases the random baseline upward (quantisation)",
        np.mean(small) > 0.0128 * 1.5,
        f"N=100 -> {np.mean(small):.5f} vs theory {g.mean():.5f}")

    # ---------- 2. metric: monotone transform invariance ----------
    print("\n[2] metric invariances")
    x = rng.normal(size=50)
    y = rng.normal(size=50) + 0.8
    s = np.r_[y, x]
    a = phase2_score(lab, s)
    b = phase2_score(lab, np.exp(s))
    chk("invariant under a strictly monotone transform",
        abs(a - b) < 1e-12, f"{a:.6f} vs {b:.6f}")

    # ---------- 3. metric: grid vs body sensitivity ----------
    print("\n[3] grid vs body (the metric's backbone and its body)")
    labN = lab_big; sN = np.r_[rng.normal(size=2000) + 0.8, rng.normal(size=8000)]
    gs = grid_sensitivity(labN, sN)
    bs = body_sensitivity(labN, sN)
    chk("linear grid changes the score materially (> 10%)",
        abs(gs["relative_change"]) > 0.10, f"{gs['relative_change']*100:+.1f}%")
    body_spread = max(bs.values()) - min(bs.values())
    chk("interpolation details change the score negligibly (< 1e-3)",
        body_spread < 1e-3, f"spread {body_spread:.2e}")

    # ---------- 4. diagnostics shapes ----------
    print("\n[4] diagnostics")
    bc, mw = band_contribution(labN, sN), marginal_weight()
    chk("band_contribution covers the whole grid",
        sum(b["points"] for b in bc) == 100)
    chk("marginal weights sum to 100 * (0.01/100) = 0.01",
        abs(sum(m["d_score_per_0.01_tpr"] for m in mw) - 0.01) < 1e-12)
    chk("mass is dominated by the HIGH-FPR band",
        bc[-1]["share_of_total"] > bc[0]["share_of_total"] * 5,
        f"{bc[-1]['share_of_total']:.3f} vs {bc[0]['share_of_total']:.3f}")

    # ---------- 5. geometry helpers ----------
    print("\n[5] geometry helpers")
    chk("comoving distance at z=0.4 is 1000-1200 h^-1 Mpc",
        1000 < comoving_hmpc(0.4) < 1200, f"{comoving_hmpc(0.4):.1f}")
    e = ell_grid((176, 1424), 2.0)
    chk("ell grid shape matches rfft2", e.shape == (176, 713), str(e.shape))
    chk("ell_min along x ~ 2*pi / (1424 * 2 arcmin)",
        abs(e[0, 1] - 2 * np.pi / np.deg2rad(1424 * 2 / 60)) < 1e-6,
        f"e[0,1]={e[0,1]:.2f}")
    chk("ell_min along y ~ 2*pi / (176 * 2 arcmin)",
        abs(e[1, 0] - 2 * np.pi / np.deg2rad(176 * 2 / 60)) < 1e-6,
        f"e[1,0]={e[1,0]:.2f}")

    # ---------- 6. generators: shapes + per-sample property ----------
    print("\n[6] generators")
    h, w, n = 64, 64, 12
    mask = np.ones((h, w), bool)
    flat = rng.normal(size=(n, int(mask.sum()))).astype(np.float32)

    for kind in ("smooth", "scale", "mask"):
        o = spatial_bias(flat, mask, kind)
        chk(f"spatial_bias('{kind}') shape preserved", o.shape == flat.shape)
        chk(f"spatial_bias('{kind}') actually changes the data",
            not np.allclose(o, flat), f"rms diff {np.sqrt(((o-flat)**2).mean()):.4f}")

    # per-sample: permuting inputs must permute outputs identically
    perm = rng.permutation(n)
    o1 = spatial_bias(flat, mask, "smooth")
    o2 = spatial_bias(flat[perm], mask, "smooth")
    chk("spatial_bias is per-sample (permutation-equivariant)",
        np.allclose(o1[perm], o2, atol=1e-6))

    tk = load_transfer_grid()
    chk("transfer grid loaded", tk is not None and "k" in tk)
    lT = np.full(n, 7.8)
    ob = baryon_bias(flat, mask, lT, 9.2, tk=tk)
    chk("baryon_bias shape preserved", ob.shape == flat.shape)
    chk("baryon_bias(logT 7.8 -> 9.2) changes the data",
        not np.allclose(ob, flat), f"rms diff {np.sqrt(((ob-flat)**2).mean()):.4f}")

    ob_same = baryon_bias(flat, mask, lT, 7.8, tk=tk)
    scale = np.sqrt((flat ** 2).mean())
    chk("baryon_bias with T_new == T_old is ~identity",
        np.sqrt(((ob_same - flat) ** 2).mean()) / scale < 1e-3,
        f"relative rms {np.sqrt(((ob_same-flat)**2).mean())/scale:.2e}")

    ob2 = baryon_bias(flat[perm], mask, lT[perm], 9.2, tk=tk)
    chk("baryon_bias is per-sample (permutation-equivariant)",
        np.allclose(ob[perm], ob2, atol=1e-6))

    # ---------- 7. optional cross-check vs the official scorer ----------
    print("\n[7] cross-check against the official scorer (optional)")
    official = os.environ.get("OFFICIAL_SCORE_PY") or \
        os.path.join(os.path.dirname(HERE), "official", "score.py")
    print(f"  path: {official}  exists={os.path.isfile(official)}")
    if official and os.path.isfile(official):
        import types
        m = types.ModuleType("official_score")
        with open(official, encoding="utf-8") as fh:
            exec(compile(fh.read(), official, "exec"), m.__dict__)
        worst = 0.0
        for _ in range(5):
            labs = np.r_[np.ones(200, int), np.zeros(800, int)]
            sc = np.r_[rng.normal(size=200) + 0.7, rng.normal(size=800)]
            o = m.Scoring(); o.reference_data = labs
            o.ingestion_result = {"ood_scores": [float(v) for v in sc]}
            o.compute_scores()
            worst = max(worst, abs(o.scores_dict["Score"] - phase2_score(labs, sc)))
        chk("matches official scoring_program/score.py to <1e-12", worst < 1e-12,
            f"max |diff| = {worst:.2e}")
    else:
        print("  [SKIP] set OFFICIAL_SCORE_PY=/path/to/scoring_program/score.py to enable")

    print("\n" + "=" * 74)
    print(f"SELF TEST {'PASS' if not FAILS else 'FAIL'}   ({len(FAILS)} failure(s))")
    for f in FAILS:
        print("   -", f)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
