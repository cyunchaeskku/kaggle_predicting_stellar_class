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
- v2 (with pseudo-abs mag) scored **0.96666** — lower. Pseudo-abs mag likely hurts. v2b result pending.

## What is done
- `src/train_baseline.py` saves `test_probabilities.csv` (patched this session).
- `src/ensemble.py` written and smoke-tested.
- `src/train_feature_v2.py` written and smoke-tested (feature diet + pseudo-abs mag + freq encoding).
- `src/train_feature_v2b.py` written and smoke-tested (feature diet + freq encoding, NO pseudo-abs mag).
- `src/train_xgboost.py` written and smoke-tested. Companion: `notebooks/run_xgboost_colab.ipynb`.

## Feature engineering roadmap (ordered by priority)
Steps done in `train_feature_v2.py`:
- Feature diet: dropped alpha_sin/cos, delta_sin/cos (redundant with coord_x/y/z)
- Pseudo-absolute magnitude: `{band}_abs_approx = band - 5*log10(|redshift|+1e-4)` for all 5 bands
- Frequency encoding: `spectral_population_count` from combined train+test

Steps still pending (implement in order):

**Step 4 — Target encoding** → `src/train_feature_v3.py`
- OOF multiclass target rates for `spectral_population` (8 combos)
- Must compute *inside* fold loop to avoid leakage
- Effect likely small (only 8 categories), do after v2 CV result confirms direction

**Step 5 — XGBoost** → `src/train_xgboost.py` + `notebooks/run_xgboost_colab.ipynb`
- Ensemble diversity: XGBoost OOF proba + test proba as separate model
- Run on Colab (memory-heavy)
- Same feature set as train_feature_v2.py

**Step 6 — KNN OOF features** → Colab only
- Use photometric + redshift space (ugriz + redshift), NOT coord_x/y/z
- k=10,50,100; 3 class-ratio features each
- OOF only — fit on train fold, transform val fold each iteration
- Sample test on small subset first to check memory/runtime on 577k rows
- StandardScaler required before KNN

**Step 7 — PCA** → only if plateau after steps 4-6
- PC1/PC2 from ugriz + 10 color indices
- Fit on train only, transform test

## What is pending (infrastructure)
1. Rerun full `train_feature_v2.py` on Colab → `outputs/lgbm_feature_v2/`
2. Compare CV vs baseline (0.966328). Submit only if delta > +0.0005.
3. Run LightGBM variants: deeper (`--num-leaves 127 --n-estimators 4000`), conservative (`--num-leaves 31 --reg-lambda 1.0`).
4. Stack only after 3+ diverse OOF prob files exist.

## Execution environment
- Local Mac (8GB RAM): smoke tests, ensemble script, quick experiments.
- Colab: full LightGBM variants + XGBoost training (more RAM, GPU available).
- When writing a new training script (e.g. `src/train_xgboost.py`), also write a companion `notebooks/run_xgboost_colab.ipynb` that runs it via `!python src/train_xgboost.py ...`. User executes on Colab, downloads OOF + test prob CSVs, ensembles locally.
- XGBoost smoke tests: always pass `--device cpu` (M2 has no CUDA).

## Commands
Smoke test (baseline):
```bash
python3 src/train_baseline.py --sample-rows 8000 --test-rows 1000 --folds 3 --n-estimators 80 --early-stopping-rounds 10 --output-dir outputs/smoke_seed42
```

Smoke test (feature v2):
```bash
python3 src/train_feature_v2.py --sample-rows 8000 --test-rows 1000 --folds 3 --n-estimators 80 --early-stopping-rounds 10 --output-dir outputs/smoke_feature_v2
```

Full train (feature v2 — run on Colab):
```bash
python3 src/train_feature_v2.py --output-dir outputs/lgbm_feature_v2
```

Full train (baseline):
```bash
python3 src/train_baseline.py --output-dir outputs/lgbm_baseline
```

Ensemble:
```bash
python3 src/ensemble.py outputs/lgbm_baseline outputs/lgbm_deep --output-dir outputs/ensemble_v1 --weighting oof_weighted
```

## Gotchas
- Writing `.ipynb` via Write tool: trailing char bug possible. Always validate: `python3 -c "import json; json.load(open('file.ipynb'))"`

## Ensemble correctness
- Encode `y_true` using column order from proba CSV, not `LabelEncoder` alphabetical. Use `class_to_idx = {cls: i for i, cls in enumerate(classes)}`.
- Smoke test ensemble with two dirs from *different* seeds/configs — not duplicated same dir.

## Outputs
- `outputs/` is gitignored. Write freely.
- `outputs/lgbm_baseline/` has no `test_probabilities.csv` until baseline is rerun after the patch.
- Submit `ensemble_submission.csv` not `submission.csv` when using ensemble.
