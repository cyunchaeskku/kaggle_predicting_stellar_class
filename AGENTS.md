# AGENTS.md

This repo is for Kaggle Playground Series S6E6: Predicting Stellar Class.

## Session Context

- User prefers concise Rocky-style communication unless they ask for normal mode.
- Current workspace: `/Users/yunchae/Desktop/Careers/sideProject/kaggle_predciting_stellar_class`
- Current local date from environment: 2026-06-01, timezone `Asia/Seoul`.
- User provided Kaggle overview text because direct Kaggle page access was limited.

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

## Recommended Next Steps

1. Create lightweight EDA notebook or script.
2. Build baseline model with stratified cross-validation and balanced accuracy.
3. Use features:
   - Numeric: `alpha`, `delta`, `u`, `g`, `r`, `i`, `z`, `redshift`
   - Categorical: `spectral_type`, `galaxy_population`
   - Useful engineered colors: `u-g`, `g-r`, `r-i`, `i-z`
4. Try tree models first: LightGBM, CatBoost, XGBoost, or sklearn HistGradientBoosting/RandomForest if dependencies are limited.
5. Save predictions as `submission.csv` with exactly `id,class`.

## Notes For Future Agents

- Do not delete or re-download data unless user asks.
- Do not use full-file `READ` on CSV files under `data/playground-series-s6e6/`; they have many rows. Use `head` to check columns or small samples when needed.
- Prefer `rg`, `head`, `wc`, and pandas quick checks for exploration.
- Keep changes scoped. This repo currently has only data plus this context file.
- If adding notebooks, keep outputs reasonable or clear them before commit unless user wants saved outputs.
