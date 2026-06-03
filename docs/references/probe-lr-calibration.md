# Probe Learning-Rate Calibration

This page records diagnostic calibration runs for frozen-feature linear probes.
These experiments are not benchmark-contract results. They are used to choose
probe optimizer defaults without overfitting to a single model family.

## Benchmark Policy

Both offline probe heads use the same feature-dimension-scaled learning-rate
policy:

```python
effective_lr = base_lr * min(1.0, (1024.0 / feature_dim) ** 3)
effective_lr = max(effective_lr, 1.0e-3)
```

The same effective scale is also applied to `final_learning_rate`, preserving
the schedule shape. With the default `base_lr = 0.01`, this keeps the previous
optimizer rate for feature dimensions up to `1024`, lowers the rate around
`1280`, and sharply reduces it for larger representations.

The policy is intentionally feature-dimension based rather than
model-family-specific. The dense diagnostics show that the old `0.01` default
causes severe optimization failures for large Toto 2.0 representations. The
categorical diagnostics show that applying the same scaling to AUROC/AUPRC has
only small impact around already-near-ceiling metrics.

## Dense Regression Probe

Date: 2026-06-02.

Setup:

- active run: `num_enabled=1`
- validation seed value: `0`
- train/validation size: full configured split, `256` batches x `256` samples
- probe: dense regression head only
- layers: current best dense layer when known, or `0 + final` for Toto 2.0
- metric used for calibration: macro dense `R2`

The issue being diagnosed was severe negative dense `R2` for the large Toto 2.0
checkpoints while their dense Pearson values stayed reasonable. That pattern
indicates a probe calibration/optimization problem rather than missing
information in the frozen representation.

| Model | Feature dim | Tested learning rates | Best tested LR | Best macro dense R2 | Macro dense R2 at 0.01 |
| --- | ---: | --- | ---: | ---: | ---: |
| `Toto-2.0-4M` | 256 | `0.01`, `0.003`, `0.001` | `0.01` | `0.434` | `0.434` |
| `MantisV2` | 512 | `0.01`, `0.005`, `0.003`, `0.001` | `0.01` | `0.912` | `0.912` |
| `Moirai-1.1-R-Base` | 768 | `0.01`, `0.0067`, `0.005`, `0.003` | `0.01` | `0.471` | `0.471` |
| `Timer-Base-84M` | 1024 | `0.01`, `0.005`, `0.003`, `0.001` | `0.01` | `0.866` | `0.866` |
| `Toto-2.0-313M` | 1024 | `0.01`, `0.003`, `0.001` | `0.01` | `0.439` | `0.439` |
| `TimesFM-2.5-200M` | 1280 | `0.01`, `0.0064`, `0.005`, `0.003` | `0.005` | `0.461` | `0.440` |
| `Toto-2.0-1B` | 1536 | `0.01`, `0.0067`, `0.005`, `0.0044`, `0.0033`, `0.003`, `0.001` | `0.003` | `0.449` | `-53.564` |
| `Toto-2.0-2.5B` | 2048 | `0.01`, `0.005`, `0.004`, `0.003`, `0.0025`, `0.001` | `0.0025` | `0.439` | `-13.485` |

Dense calibration conclusion:

```python
effective_lr = base_lr * min(1.0, (1024.0 / feature_dim) ** 3)
effective_lr = max(effective_lr, 1.0e-3)
```

With `base_lr = 0.01`, this keeps the existing default for feature dimensions up
to `1024`, lowers the rate around `1280`, and sharply reduces it for the large
Toto 2.0 checkpoints that fail at `0.01`.

## Categorical Probe

Date: 2026-06-02.

Setup:

- active run: `num_enabled=1`
- validation seed value: `0`
- train/validation size: full configured split, `256` batches x `256` samples
- probe: categorical classification head only
- layers: current best AUROC/AUPRC layer or both when they differ
- metrics used for calibration: macro AUROC and macro AUPRC

| Model | Feature dim | Tested learning rates | Best AUROC LR | Best macro AUROC | AUROC at 0.01 | Best AUPRC LR | Best macro AUPRC | AUPRC at 0.01 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `MantisV2` | 512 | `0.01`, `0.005`, `0.003`, `0.001` | `0.01` | `0.999680` | `0.999680` | `0.01` | `0.995928` | `0.995928` |
| `Moirai-1.1-R-Base` | 768 | `0.01`, `0.0067`, `0.005`, `0.003` | `0.01` | `0.998686` | `0.998686` | `0.01` | `0.987957` | `0.987957` |
| `Timer-Base-84M` | 1024 | `0.01`, `0.005`, `0.003`, `0.001` | `0.01` | `0.993539` | `0.993539` | `0.01` | `0.929633` | `0.929633` |
| `TimesFM-2.5-200M` | 1280 | `0.01`, `0.0064`, `0.005`, `0.003` | `0.01` | `0.999949` | `0.999949` | `0.01` | `0.999299` | `0.999299` |
| `Toto-2.0-4M` | 256 | `0.01`, `0.003`, `0.001` | `0.001` | `0.999515` | `0.998506` | `0.001` | `0.993573` | `0.989965` |
| `Toto-2.0-313M` | 1024 | `0.01`, `0.003`, `0.001` | `0.001` | `0.999992` | `0.999986` | `0.01` | `0.999913` | `0.999913` |
| `Toto-2.0-1B` | 1536 | `0.01`, `0.005`, `0.003`, `0.001` | `0.001` | `1.000000` | `0.999996` | `0.001` | `0.999985` | `0.999912` |
| `Toto-2.0-2.5B` | 2048 | `0.01`, `0.004`, `0.0025`, `0.001` | `0.0025`/`0.001` | `0.999999` | `0.999998` | `0.0025` | `0.999971` | `0.999954` |

Categorical conclusion:

- The same feature-dimension scaling is acceptable for the categorical probe.
- For `feature_dim <= 1024`, the policy leaves `0.01` unchanged.
- For `feature_dim > 1024`, the measured AUROC/AUPRC changes are small. TimesFM
  at `1280` loses about `0.00037` macro AUPRC under the scaled LR, while the
  large Toto 2.0 checkpoints slightly improve or remain effectively unchanged.
- This tradeoff is preferable to maintaining separate categorical/dense LR
  policies because it gives one transparent benchmark standard and avoids
  model-family-specific exceptions.

## Reproducibility Notes

The diagnostics were run with `scripts/diagnose_toto2_probe_lr.py`, writing CSV
and JSON artifacts under `results/toto2_probe_lr_diagnostics/`. The script
collects frozen features once per model/layer set, then reuses them for a sweep
over exact probe learning rates with benchmark LR scaling disabled. It supports
`--head dense` and `--head categorical` for the two probe heads.
