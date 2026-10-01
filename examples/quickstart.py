#!/usr/bin/env python3
"""Quick start: generate OoD samples and diagnose the Phase-2 metric.

Runs out of the box with synthetic data (no competition data required).
If you have the real data, see --data below.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge import (phase2_score, random_reference, band_contribution, marginal_weight,
                   measurement_precision, grid_sensitivity, body_sensitivity,
                   RANDOM_BASELINE)
from oodgen import spatial_bias, baryon_bias, load_transfer_grid


def synthetic(n=300, h=64, w=64, seed=0):
    """A tiny stand-in for the real data: correlated Gaussian fields."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), bool)
    mask[:, :] = True
    # power-law field so that smoothing / transfer functions actually do something
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.rfftfreq(w)[None, :]
    k = np.sqrt(ky ** 2 + kx ** 2)
    k[0, 0] = 1e-6
    amp = k ** -1.2
    def one():
        ph = rng.normal(size=(h, w // 2 + 1)) + 1j * rng.normal(size=(h, w // 2 + 1))
        return np.fft.irfft2(amp * ph, s=(h, w)).astype(np.float32)
    flat = np.stack([one().ravel() for _ in range(n)])
    return mask, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with the real data (optional)")
    a = ap.parse_args()

    print("=" * 78)
    print("Phase-2 OoD toolkit - quick start")
    print("=" * 78)

    mask, flat = synthetic()
    print(f"\nsynthetic: {flat.shape[0]} maps, mask {mask.sum()} px "
          f"({mask.shape[0]}x{mask.shape[1]})")

    # ---------------- generators ----------------
    print("\n[1] OoD generators (per-sample)")
    ood_smooth = spatial_bias(flat, mask, "smooth")
    print(f"    spatial_bias('smooth')  -> {ood_smooth.shape}")
    tk = load_transfer_grid()
    print(f"    transfer grid: {len([k for k in tk if k.startswith('lT')])} logT_AGN values "
          f"({min(float(k[2:]) for k in tk if k.startswith('lT'))}"
          f" .. {max(float(k[2:]) for k in tk if k.startswith('lT'))})")
    lT_old = np.full(flat.shape[0], 7.8)
    ood_baryon = baryon_bias(flat, mask, lT_old, 9.2, tk=tk)
    print(f"    baryon_bias(logT 7.8 -> 9.2) -> {ood_baryon.shape}")

    # ---------------- metric geometry ----------------
    lab = np.r_[np.ones(flat.shape[0], int), np.zeros(flat.shape[0], int)]
    # deliberately simple demo score: smoothing suppresses the small-scale power,
    # so 1/sigma is a (weak) OoD detector for the "smooth" family.
    sc = np.r_[1.0 / ood_smooth.std(1), 1.0 / flat.std(1)]

    print("\n[2] metric sanity")
    rr = random_reference(n_test=10000, reps=60, seed=1)
    print(f"    random score   simulated {rr['mean']:.5f} +/- {rr['std']:.5f}"
          f"   |  published {RANDOM_BASELINE}  |  theory {rr['theory']:.5f}")
    print(f"    perfect = {phase2_score(np.r_[np.ones(5,int),np.zeros(5,int)], np.r_[np.ones(5),np.zeros(5)]):.5f}"
          f"   inverted = {phase2_score(np.r_[np.ones(5,int),np.zeros(5,int)], np.r_[np.zeros(5),np.ones(5)]):.5f}")

    print("\n[3] metric geometry -- grid vs body")
    print("    grid_sensitivity:", {k: round(v, 5) if isinstance(v, float) else v
                                    for k, v in grid_sensitivity(lab, sc).items()})
    print("    body_sensitivity:", {k: round(v, 5) for k, v in body_sensitivity(lab, sc).items()})

    print("\n[4] where the score comes from, and where it pays")
    print("    %-20s %-8s %-14s %s" % ("FPR band", "points", "share of mass", "marginal dS per +0.01 TPR"))
    for b, m in zip(band_contribution(lab, sc), marginal_weight()):
        print("    [%.3f,%.3f)      %-8d %-14.1f%% %.5f"
              % (b["band"][0], b["band"][1], b["points"], 100 * b["share_of_total"],
                 m["d_score_per_0.01_tpr"]))

    print("\n[5] measurement floor")
    mp = measurement_precision(lab, sc, reps=20)
    print(f"    score {mp['score']:.5f}  std {mp['std']:.5f}  "
          f"=> differences below {mp['measurable_delta_2sigma']:.5f} (2 sigma) are not measurable")

    if a.data:
        print(f"\n[6] real data at {a.data} - see README for the loader")
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
