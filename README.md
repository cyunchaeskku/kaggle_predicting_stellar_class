# Predicting Stellar Class

Baseline code for Kaggle Playground Series S6E6.

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

Metric is balanced accuracy. Main model is LightGBM with class weighting, astronomy color features, magnitude summary features, coordinate sine/cosine features, and OOF class multiplier tuning.
