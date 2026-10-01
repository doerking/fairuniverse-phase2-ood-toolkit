# Phase-2 OoD Toolkit — generators + metric diagnostics

A small, self-contained toolkit for the **FAIR Universe Weak Lensing ML Uncertainty
Challenge, Phase 2** (out-of-distribution detection).

It addresses two gaps that are **stated or implied by the organizers themselves** and,
to our knowledge, are **not addressed by any published baseline or released code**:

| # | Gap | Evidence |
|---|-----|----------|
| **1** | **No OoD examples and no OoD generator.** | `pages/Phase2/data.md`, verbatim: *"participants are not provided with the ground-truth labels, **any explicit OoD examples**, or **details regarding how the OoD data are generated**."* The official repository (50 files) contains **zero** OoD-generation code. |
| **2** | **The OoD "categories" are never enumerated.** | White paper §5.2.3: the baselines "do not perform as well as the power spectrum analysis **across some OoD categories** in the test data". Which categories is never disclosed. |
| **3** | **The metric is a low-FPR tail metric, but no released method engineers the tail.** | The score is the mean TPR over 100 **log-spaced** FPRs in [0.001, 0.05]. We inspected the three official baselines, the organizers' own OoD paper (arXiv:2505.00632, scored by **AUROC**) and a third-party Phase-2 write-up — **none** performs tail calibration. |

This toolkit is **generator + diagnostics only**. It contains no tuned solution.

---

## Install

```
pip install numpy scipy scikit-learn
pip install camb            # only needed to REGENERATE the transfer grid
# kymatio is NOT required by this toolkit
```

## Quick start

```bash
python examples/quickstart.py
```

```python
import numpy as np
from oodgen import spatial_bias, baryon_bias
from judge  import phase2_score, band_contribution, measurement_precision

mask  = np.load("mask.npy")               # (1424,176) bool
kappa = np.load("kappa_flat.npy")         # (n, 132019) float32, flattened over mask

# --- A. physical OoD: swap the baryonic model -----------------------------
ood = baryon_bias(kappa, mask, logT_old=logT_old_per_sample, logT_new=9.0)

# --- B. spatial OoD: the organizers' own three biases ---------------------
ood = spatial_bias(kappa, mask, kind="smooth")     # or "scale" / "mask"

# --- diagnostics ----------------------------------------------------------
score = phase2_score(labels, ood_scores)
print(band_contribution(labels, ood_scores))
print(measurement_precision(labels, ood_scores))
```

---

## What is inside

```
oodgen/           A. OoD generators
  ood.py            spatial_bias()  smooth | scale | mask   (organizers' §3.6)
                    baryon_bias()   kappa * T_new(k)/T_old(k)  (HMcode-2020)
                    transfer_camb() T(k) = P_baryon/P_DMO
  data/tk_grid.npz  precomputed T(k) for logT_AGN = 6.0 ... 9.5
judge/            B. metric + diagnostics
  metric.py         exact reimplementation of the published score
  diagnostics.py    band contribution / marginal weight / measurable delta
                    / grid sensitivity / body sensitivity
docs/FINDINGS.md  measured properties of the metric (no hidden labels needed)
examples/         runnable quick start
tests/            self-tests, incl. a cross-check against the official scorer
```

## Generators are per-sample by construction

The competition forbids any method that aggregates information across test samples
("Rules for OoD Detection Methods"). Both generators satisfy this: each output map
depends only on its own input map. `spatial_bias` and `baryon_bias` loop over maps
independently and share no cross-sample state.

---

## Key measured properties of the metric

Numbers are reproducible with `examples/quickstart.py`; details in `docs/FINDINGS.md`.

1. **The metric is grid-defined.** Replacing the log-spaced FPR grid with a linear one
   (same 100 points, same range) changes the score by **+18 % to +96 %** on synthetic
   cases. The log spacing is the metric's backbone, not a detail.
2. **The metric is robust in its body.** `drop_intermediate`, staircase vs linear
   interpolation, and linear-in-log interpolation all change the score by `< 1e-4`.
3. **Score mass and score leverage are different questions.** Where the score *comes
   from* and where an improvement *pays off* are not the same, because the grid is
   log-spaced. The toolkit exposes both (`judge.diagnostics`); the per-band numbers
   are left to the reader.
4. **AUROC does not predict this score.** On a synthetic case with AUROC = 0.48
   (below chance) the Phase-2 score was 0.45 — more than twice the published
   power-spectrum baseline (0.2143). Signed-score ROC can cross the diagonal twice,
   so a strong upper tail coexists with a poor overall AUROC.
5. **The measurement floor is set by the test-set size.** At baseline level the
   run-to-run spread of the score is `std ≈ 0.017`, so differences below
   ~`0.024` (2σ) are not measurable at all.

## Sanity anchors (organizers' published numbers)

| Method | Score |
|---|---|
| random OoD scores | 0.0128 |
| autoencoder | 0.1307 |
| CNN + MCMC | 0.1053 |
| **power spectrum + χ²** | **0.2143** |

---

## Attribution & license

* The metric formula is taken from the organizers' public
  `pages/Phase2/evaluation.md`; this package **reimplements** it rather than
  redistributing their scoring program.
* The spatial biases follow the organizers' companion paper,
  **Diao, Dai & Seljak (2025), arXiv:2505.00632**, §3.6.
* The transfer-function route uses **HMcode-2020** via `CAMB`
  (Mead et al. 2021, arXiv:2009.01858).
* Competition data: DOI [10.5281/zenodo.20056065](https://doi.org/10.5281/zenodo.20056065).

*This toolkit contains no competition submission and no tuned method. It is intended
as a shared benchmark harness.*
