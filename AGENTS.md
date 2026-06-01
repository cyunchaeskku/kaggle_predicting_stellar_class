# AGENTS.md

This repo is for Kaggle Playground Series S6E6: Predicting Stellar Class.

## Session Context

- User prefers concise Rocky-style communication unless they ask for normal mode.
- Current workspace: `/Users/yunchae/Desktop/Careers/sideProject/kaggle_predciting_stellar_class`
- Current local date from environment: 2026-06-01, timezone `Asia/Seoul`.
- User provided Kaggle overview text because direct Kaggle page access was limited.
- Git remote: `https://github.com/cyunchaeskku/kaggle_predicting_stellar_class.git`
- First preserved commit: `b11c871 Add baseline stellar classification pipeline`.
- User asked to avoid `git diff` unless needed because it can print too many lines.
- User reported first baseline reached Kaggle public rank `6 / 125`.

## Competition Facts

- Competition URL: `https://www.kaggle.com/competitions/playground-series-s6e6/overview`
- Title: Predicting Stellar Class
- Goal: predict stellar `class`.
- Target labels: `GALAXY`, `STAR`, `QSO`.
- Evaluation metric: balanced accuracy between predicted class and observed target.
- Submission format:

```csv
id,class
577347,STAR
577348,GALAXY
577349,STAR
```

- Start Date: June 1, 2026.
- Final Submission Deadline: June 30, 2026 at 11:59 PM UTC.
- Dataset: synthetic tabular Playground dataset.
- Citation: Yao Yan, Walter Reade, Elizabeth Park. Predicting Stellar Class. Kaggle, 2026.

## Local Data

Data was downloaded and extracted under:

```text
data/playground-series-s6e6/
```

Verified files:

- `data/playground-series-s6e6/train.csv`
- `data/playground-series-s6e6/test.csv`
- `data/playground-series-s6e6/sample_submission.csv`
- `data/playground-series-s6e6.zip`

Verified shapes:

- `train.csv`: 577,347 rows, 12 columns.
- `test.csv`: 247,435 rows, 11 columns.
- `sample_submission.csv`: 247,435 rows, 2 columns.

Verified schema:

- Train columns: `id, alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population, class`
- Test columns: `id, alpha, delta, u, g, r, i, z, redshift, spectral_type, galaxy_population`
- Sample submission columns: `id, class`

Verified integrity:

- Missing values: 0 in train and test.
- Duplicate train ids: 0.
- Duplicate test ids: 0.
- Train/test id overlap: 0.
- Sample submission ids match test ids exactly.
- Train ids end at `577346`; test ids start at `577347`.

Target distribution in train:

```text
GALAXY    377480
QSO       117143
STAR       82724
```

Categorical values:

- `spectral_type`: `A/F`, `G/K`, `M`, `O/B`
- `galaxy_population`: `Blue_Cloud`, `Red_Sequence`

## Current First Version

Implemented files:

- `src/train_baseline.py`
- `README.md`
- `.gitignore`

Baseline model:

- LightGBM multiclass classifier.
- 5-fold stratified CV.
- Uses raw numeric features, categorical features, color indices, magnitude summaries, coordinate sine/cosine features, and redshift-color interactions.
- Uses `class_weight="balanced"` and OOF class multiplier tuning.
- Main output directory: `outputs/lgbm_baseline/`.
- Main submission file: `outputs/lgbm_baseline/submission.csv`.

Full baseline command:

```bash
python3 src/train_baseline.py --output-dir outputs/lgbm_baseline
```

First full-run result:

- OOF balanced accuracy: `0.964731`.
- Tuned OOF balanced accuracy: `0.966328`.
- Class multipliers: `GALAXY:0.62123`, `QSO:1.11560`, `STAR:1.26317`.
- Submission rows: `247435`.

## Current Feature Upgrade

After the first version, `src/train_baseline.py` was extended with three feature groups to test next:

1. Redshift bins and flags:
   - `redshift_abs`
   - `redshift_log1p_abs`
   - `redshift_bin`
   - `is_redshift_near_zero`
   - `is_redshift_low`
   - `is_redshift_high`
   - `redshift_x_mag_mean`
   - `redshift_x_mag_range`
2. Cartesian sky coordinates from `alpha` and `delta`:
   - `coord_x`
   - `coord_y`
   - `coord_z`
3. Combined categorical feature:
   - `spectral_population = spectral_type + "__" + galaxy_population`

Treat this as an experimental feature-upgraded LGBM variant until CV and Kaggle scores prove it beats the first baseline.

Current best Kaggle submission:

- File: `outputs/lgbm_feature_upgrade/submission.csv`
- Public score: `0.96808`
- Status: highest known score so far for this project.

## Recommended Next Steps

1. Patch training script to save `test_probabilities.csv` for ensembling.
2. Create `src/ensemble.py` to average OOF/test probabilities and tune class multipliers on ensemble OOF.
3. Run controlled LightGBM variants:
   - raw no multiplier
   - deeper: `--n-estimators 4000 --num-leaves 127 --min-child-samples 60`
   - conservative: `--num-leaves 31 --min-child-samples 140 --reg-lambda 1.0`
4. Evaluate the current feature-upgraded LGBM variant against the first baseline before submitting.
5. Add XGBoost model for ensemble diversity.
6. Submit only high-confidence candidates; avoid public leaderboard overfitting.

## Notes For Future Agents

- Do not delete or re-download data unless user asks.
- Do not use full-file `READ` on CSV files under `data/playground-series-s6e6/`; they have many rows. Use `head` to check columns or small samples when needed.
- Prefer `rg`, `head`, `wc`, and pandas quick checks for exploration.
- Keep changes scoped. This repo has tracked data plus baseline code.
- If adding notebooks, keep outputs reasonable or clear them before commit unless user wants saved outputs.
- Do not run destructive git commands such as `git reset --hard` unless user explicitly asks.
