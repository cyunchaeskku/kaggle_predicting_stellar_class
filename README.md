# Kaggle Predicting Stellar Class

Code for Kaggle Playground Series S6E6: Predicting Stellar Class.

Goal: predict stellar `class` (`GALAXY`, `STAR`, `QSO`) from synthetic tabular astronomy features. Metric is balanced accuracy.

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

Outputs:

- `submission.csv`
- `oof_predictions.csv`
- `feature_importance.csv`
- `feature_importance_by_fold.csv`

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

## Notes

- CSV files under `data/playground-series-s6e6/` are large. Use `head` for schema checks; avoid full-file terminal reads.
- Generated files under `outputs/` are ignored by git.
- Data files were included in the first commit because the first version was explicitly preserved with `git add .`.
