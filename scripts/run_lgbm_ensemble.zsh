#!/usr/bin/env zsh
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENSEMBLE_DIR="outputs/ensemble_v1"

echo "==> 1/3 rerun baseline with test probabilities"
"$PYTHON_BIN" src/train_baseline.py \
  --output-dir outputs/lgbm_baseline

echo "==> 2/3 train deep LightGBM variant"
"$PYTHON_BIN" src/train_baseline.py \
  --output-dir outputs/lgbm_deep \
  --n-estimators 4000 \
  --num-leaves 127 \
  --min-child-samples 60

echo "==> 3/3 build weighted ensemble"
"$PYTHON_BIN" src/ensemble.py \
  outputs/lgbm_baseline \
  outputs/lgbm_deep \
  --output-dir "$ENSEMBLE_DIR" \
  --weighting oof_weighted

echo "==> done"
echo "Submit: $ENSEMBLE_DIR/ensemble_submission.csv"
echo "Report: $ENSEMBLE_DIR/ensemble_report.txt"
