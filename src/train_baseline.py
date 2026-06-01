#!/usr/bin/env python3
"""LightGBM baseline for Kaggle Playground S6E6 stellar classification."""

from __future__ import annotations

import argparse
import gc
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "LightGBM is required. Install with: pip install lightgbm"
    ) from exc


TARGET = "class"
ID_COL = "id"
CAT_COLS = ["spectral_type", "galaxy_population"]
BANDS = ["u", "g", "r", "i", "z"]
BASE_NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CAT_DTYPES = {
    "spectral_type": CategoricalDtype(categories=["A/F", "G/K", "M", "O/B"]),
    "galaxy_population": CategoricalDtype(categories=["Blue_Cloud", "Red_Sequence"]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/playground-series-s6e6"),
        help="Directory containing train.csv, test.csv, sample_submission.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lgbm_baseline"),
        help="Directory for OOF predictions, feature importance, submission.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=0.2)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only first N train rows for smoke tests.",
    )
    parser.add_argument(
        "--test-rows",
        type=int,
        default=None,
        help="Read only first N test/submission rows for smoke tests.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/evaluate only. Do not load test or write submission.",
    )
    parser.add_argument(
        "--tune-class-multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tune simple class probability multipliers on OOF predictions.",
    )
    parser.add_argument(
        "--multiplier-trials",
        type=int,
        default=1500,
        help="Random search trials for class probability multipliers.",
    )
    return parser.parse_args()


def reduce_memory(df: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    numeric_cols = BASE_NUMERIC
    for col in numeric_cols:
        df[col] = df[col].astype("float32")
    df[ID_COL] = df[ID_COL].astype("int32")
    for col in CAT_COLS:
        df[col] = df[col].astype(CAT_DTYPES[col])
    if has_target:
        df[TARGET] = df[TARGET].astype("category")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for left_idx, left in enumerate(BANDS):
        for right in BANDS[left_idx + 1 :]:
            df[f"{left}-{right}"] = (df[left] - df[right]).astype("float32")

    band_values = df[BANDS]
    df["mag_mean"] = band_values.mean(axis=1).astype("float32")
    df["mag_std"] = band_values.std(axis=1).astype("float32")
    df["mag_min"] = band_values.min(axis=1).astype("float32")
    df["mag_max"] = band_values.max(axis=1).astype("float32")
    df["mag_range"] = (df["mag_max"] - df["mag_min"]).astype("float32")

    alpha_rad = np.deg2rad(df["alpha"].astype("float32"))
    delta_rad = np.deg2rad(df["delta"].astype("float32"))
    df["alpha_sin"] = np.sin(alpha_rad).astype("float32")
    df["alpha_cos"] = np.cos(alpha_rad).astype("float32")
    df["delta_sin"] = np.sin(delta_rad).astype("float32")
    df["delta_cos"] = np.cos(delta_rad).astype("float32")

    for color in ["u-g", "g-r", "r-i", "i-z", "u-z"]:
        df[f"redshift_x_{color}"] = (df["redshift"] * df[color]).astype("float32")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    ignored = {ID_COL, TARGET}
    return [col for col in df.columns if col not in ignored]


def read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train = pd.read_csv(args.data_dir / "train.csv", nrows=args.sample_rows)
    train = reduce_memory(train, has_target=True)
    train = add_features(train)

    if args.skip_test:
        return train, None, None

    test = pd.read_csv(args.data_dir / "test.csv", nrows=args.test_rows)
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", nrows=args.test_rows)
    test = reduce_memory(test, has_target=False)
    test = add_features(test)
    return train, test, sample


def make_model(args: argparse.Namespace, seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        class_weight="balanced",
        random_state=seed,
        n_jobs=args.n_jobs,
        force_col_wise=True,
        verbosity=-1,
    )


def tune_class_multipliers(
    y_true: np.ndarray,
    proba: np.ndarray,
    seed: int,
    trials: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = np.ones(proba.shape[1], dtype="float32")
    best_score = balanced_accuracy_score(y_true, np.argmax(proba, axis=1))

    candidates = [np.ones(proba.shape[1], dtype="float32")]
    candidates.extend(rng.lognormal(mean=0.0, sigma=0.25, size=(trials, proba.shape[1])).astype("float32"))

    for weights in candidates:
        pred = np.argmax(proba * weights, axis=1)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best = weights

    best = best / best.mean()
    best_score = balanced_accuracy_score(y_true, np.argmax(proba * best, axis=1))
    return best.astype("float32"), best_score


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample = read_inputs(args)
    labels = LabelEncoder()
    y = labels.fit_transform(train[TARGET].astype(str))
    x = train[feature_columns(train)]
    test_x = None if test is None else test[feature_columns(test)]
    if test_x is not None and list(x.columns) != list(test_x.columns):
        raise ValueError("Train/test feature columns do not match.")

    oof = np.zeros((len(train), len(labels.classes_)), dtype="float32")
    test_pred = None if test is None else np.zeros((len(test), len(labels.classes_)), dtype="float32")
    importances: list[pd.DataFrame] = []

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
        model = make_model(args, seed=args.seed + fold)
        x_train, y_train = x.iloc[train_idx], y[train_idx]
        x_valid, y_valid = x.iloc[valid_idx], y[valid_idx]

        callbacks = [
            lgb.early_stopping(args.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(args.log_every),
        ]
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="multi_logloss",
            categorical_feature=CAT_COLS,
            callbacks=callbacks,
        )

        valid_pred = model.predict_proba(x_valid, num_iteration=model.best_iteration_)
        oof[valid_idx] = valid_pred
        fold_score = balanced_accuracy_score(y_valid, np.argmax(valid_pred, axis=1))
        print(f"fold={fold} best_iter={model.best_iteration_} balanced_accuracy={fold_score:.6f}")

        if test_pred is not None and test_x is not None:
            test_pred += model.predict_proba(test_x, num_iteration=model.best_iteration_) / args.folds

        importances.append(
            pd.DataFrame(
                {
                    "feature": x.columns,
                    "importance": model.feature_importances_,
                    "fold": fold,
                }
            )
        )

        del model, x_train, y_train, x_valid, y_valid, valid_pred
        gc.collect()

    base_pred = np.argmax(oof, axis=1)
    base_score = balanced_accuracy_score(y, base_pred)
    final_weights = np.ones(len(labels.classes_), dtype="float32")
    final_score = base_score

    if args.tune_class_multipliers:
        final_weights, final_score = tune_class_multipliers(
            y_true=y,
            proba=oof,
            seed=args.seed,
            trials=args.multiplier_trials,
        )

    final_pred = np.argmax(oof * final_weights, axis=1)
    print(f"oof_balanced_accuracy={base_score:.6f}")
    print(f"oof_balanced_accuracy_tuned={final_score:.6f}")
    print("class_multipliers=" + ",".join(f"{cls}:{weight:.5f}" for cls, weight in zip(labels.classes_, final_weights)))
    print(classification_report(y, final_pred, target_names=labels.classes_, digits=5))

    oof_df = pd.DataFrame(
        {
            ID_COL: train[ID_COL].to_numpy(),
            "true": train[TARGET].astype(str).to_numpy(),
            "pred": labels.inverse_transform(final_pred),
        }
    )
    for idx, cls in enumerate(labels.classes_):
        oof_df[f"proba_{cls}"] = oof[:, idx]
    oof_df.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    importance_df = pd.concat(importances, ignore_index=True)
    importance_df.to_csv(args.output_dir / "feature_importance_by_fold.csv", index=False)
    (
        importance_df.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .to_csv(args.output_dir / "feature_importance.csv", index=False)
    )

    if test_pred is not None and sample is not None:
        submission = sample[[ID_COL]].copy()
        submission[TARGET] = labels.inverse_transform(np.argmax(test_pred * final_weights, axis=1))
        submission.to_csv(args.output_dir / "submission.csv", index=False)
        print(f"wrote_submission={args.output_dir / 'submission.csv'} rows={len(submission)}")


if __name__ == "__main__":
    main()
