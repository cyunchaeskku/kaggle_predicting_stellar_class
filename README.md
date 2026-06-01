# Kaggle Predicting Stellar Class

Code for Kaggle Playground Series S6E6: Predicting Stellar Class.

Goal: predict stellar `class` (`GALAXY`, `STAR`, `QSO`) from synthetic tabular astronomy features. Metric is balanced accuracy.

## Competition Info

Competition URL:

```text
https://www.kaggle.com/competitions/playground-series-s6e6/overview
```

Overview:

```text
Your Goal: Predict the stellar class.
```

Evaluation:

```text
Submissions are evaluated on balanced accuracy between the predicted class and observed target.
```

Relevant sklearn metric:

```python
sklearn.metrics.balanced_accuracy_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    adjusted=False,
)
```

## First Version

Current first-version baseline:

- Commit: `b11c871`
- Model: LightGBM multiclass classifier
- Validation: 5-fold stratified CV
- Features: raw numeric columns, categorical columns, color indices, magnitude summaries, coordinate sine/cosine, redshift-color interactions
- Class handling: `class_weight="balanced"` plus OOF-tuned class probability multipliers
- OOF balanced accuracy: `0.964731`
- Tuned OOF balanced accuracy: `0.966328`
- User-reported Kaggle public rank after first submission: `6 / 125`

Main submission file:

```text
outputs/lgbm_baseline/submission.csv
```

## Data Columns

`train.csv` columns:

- `id`: unique row identifier.
- `alpha`: right ascension sky coordinate in degrees.
- `delta`: declination sky coordinate in degrees.
- `u`: ultraviolet-band magnitude. Higher value means dimmer object in this band.
- `g`: green/visible-band magnitude.
- `r`: red-band magnitude.
- `i`: near-infrared-band magnitude.
- `z`: longer near-infrared-band magnitude.
- `redshift`: wavelength shift from object motion or cosmological expansion.
- `spectral_type`: categorical spectral group. Values: `A/F`, `G/K`, `M`, `O/B`.
- `galaxy_population`: categorical galaxy population. Values: `Blue_Cloud`, `Red_Sequence`.
- `class`: target label. Values: `GALAXY`, `STAR`, `QSO`.

`test.csv` has same feature columns except `class`.

Useful derived color features:

- `u-g`
- `g-r`
- `r-i`
- `i-z`
- broader colors such as `u-r`, `u-i`, `g-z`, `u-z`

## Run Baseline

Smoke test:

```bash
python3 src/train_baseline.py \
  --sample-rows 8000 \
  --test-rows 1000 \
  --folds 3 \
  --n-estimators 80 \
  --early-stopping-rounds 10 \
  --output-dir outputs/smoke
```

Full local run:

```bash
python3 src/train_baseline.py --output-dir outputs/lgbm_baseline
```

Faster local variant:

```bash
python3 src/train_baseline.py \
  --n-estimators 1000 \
  --early-stopping-rounds 80 \
  --output-dir outputs/lgbm_fast
```

Outputs:

- `submission.csv`
- `oof_predictions.csv`
- `feature_importance.csv`
- `feature_importance_by_fold.csv`

## Ensemble

After rerunning the baseline (needed to generate `test_probabilities.csv`), combine multiple model outputs:

```bash
python3 src/ensemble.py \
  outputs/lgbm_baseline \
  outputs/lgbm_deep \
  --output-dir outputs/ensemble_v1 \
  --weighting oof_weighted
```

Outputs:

- `ensemble_submission.csv` — submit this
- `ensemble_oof_predictions.csv`
- `ensemble_report.txt` — per-model OOF scores, weights, ensemble scores

> **Note:** The original `outputs/lgbm_baseline` run does not have `test_probabilities.csv`.
> Rerun `python3 src/train_baseline.py --output-dir outputs/lgbm_baseline` after this patch before using ensemble.

## Submission

Submit this file to Kaggle:

```text
outputs/lgbm_baseline/submission.csv
```

Expected submission format:

```csv
id,class
577347,STAR
577348,GALAXY
577349,STAR
```

Do not submit OOF predictions or feature importance files.

## Next Experiments

Use CV first, public leaderboard second. Avoid overfitting to daily submissions.

Recommended next steps:

1. Save test probabilities for ensembling.
2. Add an ensemble script for averaging OOF/test probabilities.
3. Try LightGBM variants:
   - no class multiplier
   - deeper trees: `--n-estimators 4000 --num-leaves 127 --min-child-samples 60`
   - conservative trees: `--num-leaves 31 --min-child-samples 140 --reg-lambda 1.0`
4. Add color-magnitude interactions and `spectral_type + galaxy_population` combo feature.
5. Add XGBoost for model diversity, then blend probabilities.

## Rollback

Current baseline is preserved in git and pushed to GitHub.

Useful restore commands:

```bash
git restore .
git restore src/train_baseline.py
```

Hard rollback only when explicitly intended:

```bash
git reset --hard b11c871
```

## Notes

- CSV files under `data/playground-series-s6e6/` are large. Use `head` for schema checks; avoid full-file terminal reads.
- Generated files under `outputs/` are ignored by git.
- Data files were included in the first commit because the first version was explicitly preserved with `git add .`.
- The current baseline uses CPU LightGBM, not MPS/GPU.
