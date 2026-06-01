# Stellar Class — Claude Notes

## Communication
- Always use Rocky mode (ultra-compressed) unless user says "normal mode" or "stop rocky".

## Data safety
- Never full-read CSV files under `data/playground-series-s6e6/` — 577K+ rows will hit token limit. Columns are in README.md.
- Use `head -3`, `wc -l`, or `pd.read_csv(..., nrows=5)` for quick checks.
- Do not delete or re-download data unless user explicitly asks.

## Git
- Avoid `git diff` unless needed — can print too many lines.
- Do not run destructive git commands (`git reset --hard`, etc.) unless user explicitly asks.
- Do not commit files under `outputs/` — gitignored.

## Project state (as of 2026-06-01)
- Metric: balanced accuracy. Target: `GALAXY`, `STAR`, `QSO`.
- Data: `data/playground-series-s6e6/train.csv` (577K rows), `test.csv` (247K rows).
- Baseline: `src/train_baseline.py` — LightGBM 5-fold CV, OOF 0.966328, public rank 6/125.
- Ensemble: `src/ensemble.py` — simple/oof_weighted averaging, class multiplier tuning.

## Best submission so far
- File: `outputs/lgbm_feature_upgrade/submission.csv`
- Kaggle public score: **0.96808**
- Do not overwrite this output dir.

## What is done
- `src/train_baseline.py` saves `test_probabilities.csv` (patched this session).
- `src/ensemble.py` written and smoke-tested.

## What is pending
1. Rerun full baseline to generate `test_probabilities.csv` for real data.
2. Run LightGBM variants: deeper (`--num-leaves 127 --n-estimators 4000`), conservative (`--num-leaves 31 --reg-lambda 1.0`), no-multiplier.
3. Inspect `outputs/lgbm_baseline/feature_importance.csv` before adding new features.
4. Add color-magnitude interactions + `spectral_type`×`galaxy_population` combo feature.
5. Add XGBoost for ensemble diversity.
6. Stack only after 3+ diverse OOF prob files exist.

## Execution environment
- Local Mac (8GB RAM): smoke tests, ensemble script, quick experiments.
- Colab: full LightGBM variants + XGBoost training (more RAM, GPU available).
- When writing a new training script (e.g. `src/train_xgboost.py`), also write a companion `notebooks/run_xgboost_colab.ipynb` that runs it via `!python src/train_xgboost.py ...`. User executes on Colab, downloads OOF + test prob CSVs, ensembles locally.

## Commands
Smoke test:
```bash
python3 src/train_baseline.py --sample-rows 8000 --test-rows 1000 --folds 3 --n-estimators 80 --early-stopping-rounds 10 --output-dir outputs/smoke_seed42
```

Full train:
```bash
python3 src/train_baseline.py --output-dir outputs/lgbm_baseline
```

Ensemble:
```bash
python3 src/ensemble.py outputs/lgbm_baseline outputs/lgbm_deep --output-dir outputs/ensemble_v1 --weighting oof_weighted
```

## Ensemble correctness
- Encode `y_true` using column order from proba CSV, not `LabelEncoder` alphabetical. Use `class_to_idx = {cls: i for i, cls in enumerate(classes)}`.
- Smoke test ensemble with two dirs from *different* seeds/configs — not duplicated same dir.

## Outputs
- `outputs/` is gitignored. Write freely.
- `outputs/lgbm_baseline/` has no `test_probabilities.csv` until baseline is rerun after the patch.
- Submit `ensemble_submission.csv` not `submission.csv` when using ensemble.
