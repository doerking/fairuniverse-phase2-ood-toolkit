# Measured properties of the Phase-2 metric

Everything below is reproducible from `examples/quickstart.py` and needs **no access to
the hidden test labels** — the metric is a function of `(labels, scores)` and its
geometry can be probed with synthetic score distributions.

Reference implementation: `judge/metric.py` (a reimplementation of the published
formula; verified against the organizers' `scoring_program/score.py`).

```
score = (1/100) * sum_{i=1..100} TPR(FPR_i),   FPR_i = logspace(0.001, 0.05, 100)
```

---

## 1. The metric is grid-defined, not just "AUC below 0.05"

| Grid | Score (4 synthetic cases) |
|---|---|
| **log-spaced (official)** | 0.0407 · 0.1955 · 0.1015 · 0.1453 |
| linear-spaced, same 100 points, same range | 0.0795 · 0.3086 · 0.1197 · 0.1872 |
| **relative change** | **+96 % · +58 % · +18 % · +29 %** |

Consequence: **scores from this metric are not comparable to partial-AUC numbers from
the literature**, which use linearly spaced FPRs.

## 2. The metric is robust in its body

| Variant | Δ relative to official |
|---|---|
| `roc_curve(drop_intermediate=False)` | `≈ 0` |
| right-continuous staircase (no interpolation) | `< 1e-4` |
| left-continuous staircase | `< 1e-4` |
| linear interpolation in `log(FPR)` | `≈ 0` |

Consequence: the *backbone* (grid spacing) carries the metric's meaning; the *body*
(interpolation) is irrelevant. Any local re-evaluation should use the official scorer.

## 3. Score mass and score leverage live in different places

Two easily conflated questions:

* **Where does the score come from?** TPR *values* are larger at high FPR, so the upper
  end of the range dominates the raw sum.
* **Where does an improvement pay off?** Because the grid is **log-spaced**, the lower
  end holds more of the 100 points, so a given TPR improvement is worth comparatively
  more there.

The two answers differ, and both ends of the range matter — for different reasons.

`judge.diagnostics` exposes both quantities (`band_contribution`, `marginal_weight`) so
that anyone can compute the per-band numbers for their own score distribution. The
per-band table itself is deliberately not reproduced here.

## 4. AUROC does not predict this score

Synthetic OoD families, all evaluated with the official metric:

| OoD construction | AUROC | Phase-2 score | TPR@FPR=0.05 |
|---|---|---|---|
| pure location shift μ=0.75 | 0.751 | 0.0866 | 0.240 |
| **pure scale change (variance only)** | **0.481** | **0.4531** | 0.462 |
| 8 % far tail + 92 % near-body | 0.757 | 0.1560 | 0.287 |

* Case 2 has AUROC **below chance** yet scores **0.4531**, more than twice the published
  power-spectrum baseline (0.2143). The signed-score ROC rises above the diagonal in the
  low-FPR region, falls below it in the middle, and rises again — it crosses the diagonal
  **twice**.
* Two constructions with the *same* AUROC (0.75) differ in score by **1.8×**.

Consequence: **optimising AUROC is not a proxy for this metric.** Methods should be
selected on the metric itself — contrast the organizers' own OoD study
(arXiv:2505.00632), which reports AUROC.

## 5. The measurement floor is set by the test-set size

Monte-Carlo spread of the score under resampling of the score distribution
(10 000 test maps, 20 % OoD):

| separation μ | mean score | std | 2σ (not measurable below) |
|---|---|---|---|
| 0.0 (pure noise) | 0.0130 | 0.0020 | 0.0041 |
| 1.0 | 0.0951 | 0.0080 | 0.0160 |
| **1.5 (≈ baseline level)** | **0.1953** | **0.0121** | **0.0241** |
| 2.5 | 0.5262 | 0.0165 | 0.0329 |

Consequence: at baseline level **differences below ≈0.024 are not measurable**. Ranking
gaps smaller than that should not be interpreted as real.

## 6. Low OoD fractions are variance-limited

| OoD fraction | mean ± std (7 seeds) |
|---|---|
| 0.2 % (n=20) | 0.1403 ± **0.0688** |
| 0.5 % (n=50) | 0.1281 ± **0.0300** |
| 1 % (n=100) | 0.1183 ± 0.0097 |
| 2 %–50 % | 0.118 – 0.138 ± ≈0.009 |

The *mean* is flat in the OoD fraction (a single-seed "sweet spot" at 0.5 % turned out to
be an outlier and did not survive repetition). The *variance* explodes below 1 %.
Any evaluation with fewer than ~100 OoD samples needs repeated runs.

## 7. The random baseline itself is sample-size dependent

The theoretical floor is the arithmetic mean of the grid, 0.01266 (= the score of a
classifier whose TPR equals its FPR). At the real test-set size it is reproduced
(0.01253 simulated vs 0.0128 published). At small sample sizes the ROC staircase
quantises TPR to `1/n_OoD`, and linear interpolation across the coarse staircase
**biases the floor upward**:

| test size | simulated random score | theory |
|---|---|---|
| n = 100 (50 / 50) | **0.0268** | 0.01266 |
| n = 10 000 (2000 / 8000) | 0.01253 | 0.01266 |

Consequence: any toy experiment with fewer than a few hundred samples will overstate
absolute scores. Compare *relative* numbers only.

---

## Implication for the official baselines

* `random` 0.0128 → `CNN` 0.1053 → `AE` 0.1307 → `power spectrum` 0.2143.
* Against the σ above, the CNN-vs-AE gap (0.025) is at the edge of measurability, and
  the AE-vs-PS gap (0.084) is comfortably real.
* None of these baselines targets property 3 (leverage) or property 4 (the AUROC
  decoupling). That is the space the toolkit is meant to make measurable.

---

## Reproducing

```bash
python examples/quickstart.py     # prints all tables above from synthetic data
python tests/test_all.py          # self-checks
```
