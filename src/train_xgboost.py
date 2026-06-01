#!/usr/bin/env python3
"""XGBoost training script: same feature set as feature-v2b, GPU-ready."""

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
from sklearn.utils.class_weight import compute_sample_weight

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "XGBoost is required. Install with: pip install xgboost"
    ) from exc


TARGET = "class"
ID_COL = "id"
RAW_CAT_COLS = ["spectral_type", "galaxy_population"]
DERIVED_CAT_COLS = ["spectral_population"]
CAT_COLS = RAW_CAT_COLS + DERIVED_CAT_COLS
BANDS = ["u", "g", "r", "i", "z"]
BASE_NUMERIC = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
SPECTRAL_TYPES = ["A/F", "G/K", "M", "O/B"]
GALAXY_POPULATIONS = ["Blue_Cloud", "Red_Sequence"]
CAT_DTYPES = {
    "spectral_type": CategoricalDtype(categories=SPECTRAL_TYPES),
    "galaxy_population": CategoricalDtype(categories=GALAXY_POPULATIONS),
    "spectral_population": CategoricalDtype(
        categories=[
            f"{spectral}__{population}"
            for spectral in SPECTRAL_TYPES
            for population in GALAXY_POPULATIONS
        ]
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/playground-series-s6e6"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/xgboost_feature_v2b"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=0.2)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--test-rows", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--tune-class-multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--multiplier-trials", type=int, default=1500)
    return parser.parse_args()


def reduce_memory(df: pd.DataFrame, has_target: bool) -> pd.DataFrame:
    for col in BASE_NUMERIC:
        df[col] = df[col].astype("float32")
    df[ID_COL] = df[ID_COL].astype("int32")
    for col in RAW_CAT_COLS:
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

    # Cartesian sky coords only — trig intermediates dropped (feature diet)
    alpha_rad = np.deg2rad(df["alpha"].astype("float32"))
    delta_rad = np.deg2rad(df["delta"].astype("float32"))
    cos_delta = np.cos(delta_rad)
    df["coord_x"] = (cos_delta * np.cos(alpha_rad)).astype("float32")
    df["coord_y"] = (cos_delta * np.sin(alpha_rad)).astype("float32")
    df["coord_z"] = np.sin(delta_rad).astype("float32")

    redshift = df["redshift"].astype("float32")
    redshift_abs = np.abs(redshift.astype("float64"))
    df["redshift_abs"] = redshift_abs.astype("float32")
    df["redshift_log1p_abs"] = np.log1p(redshift_abs).astype("float32")
    df["redshift_bin"] = np.digitize(
        redshift,
        bins=np.array([-0.001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0], dtype="float32"),
    ).astype("int8")
    df["is_redshift_near_zero"] = (redshift_abs <= 0.005).astype("int8")
    df["is_redshift_low"] = (redshift < 0.1).astype("int8")
    df["is_redshift_high"] = (redshift >= 1.0).astype("int8")
    df["redshift_x_mag_mean"] = (redshift * df["mag_mean"]).astype("float32")
    df["redshift_x_mag_range"] = (redshift * df["mag_range"]).astype("float32")

    df["spectral_population"] = (
        df["spectral_type"].astype("string")
        + "__"
        + df["galaxy_population"].astype("string")
    ).astype(CAT_DTYPES["spectral_population"])

    for color in ["u-g", "g-r", "r-i", "i-z", "u-z"]:
        df[f"redshift_x_{color}"] = (df["redshift"] * df[color]).astype("float32")

    return df


def add_frequency_encoding(train: pd.DataFrame, test: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    combined = pd.concat(
        [train["spectral_population"]] + ([test["spectral_population"]] if test is not None else []),
        ignore_index=True,
    )
    freq_map = combined.value_counts().to_dict()
    train = train.copy()
    train["spectral_population_count"] = train["spectral_population"].astype(str).map(freq_map).fillna(0).astype("int32")
    if test is not None:
        test = test.copy()
        test["spectral_population_count"] = test["spectral_population"].astype(str).map(freq_map).fillna(0).astype("int32")
    return train, test


def feature_columns(df: pd.DataFrame) -> list[str]:
    ignored = {ID_COL, TARGET}
    return [col for col in df.columns if col not in ignored]


def read_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    train = pd.read_csv(args.data_dir / "train.csv", nrows=args.sample_rows)
    train = reduce_memory(train, has_target=True)
    train = add_features(train)

    if args.skip_test:
        train, _ = add_frequency_encoding(train, None)
        return train, None, None

    test = pd.read_csv(args.data_dir / "test.csv", nrows=args.test_rows)
    sample = pd.read_csv(args.data_dir / "sample_submission.csv", nrows=args.test_rows)
    test = reduce_memory(test, has_target=False)
    test = add_features(test)
    train, test = add_frequency_encoding(train, test)
    return train, test, sample


def make_model(args: argparse.Namespace, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        device=args.device,
        enable_categorical=True,
        random_state=seed,
        n_jobs=args.n_jobs,
        verbosity=0,
        eval_metric="mlogloss",
        early_stopping_rounds=args.early_stopping_rounds,
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

        sample_weight = compute_sample_weight("balanced", y_train)

        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            verbose=args.log_every,
        )

        best_iter = model.best_iteration
        valid_pred = model.predict_proba(x_valid, iteration_range=(0, best_iter + 1))
        oof[valid_idx] = valid_pred
        fold_score = balanced_accuracy_score(y_valid, np.argmax(valid_pred, axis=1))
        print(f"fold={fold} best_iter={best_iter} balanced_accuracy={fold_score:.6f}")

        if test_pred is not None and test_x is not None:
            test_pred += model.predict_proba(test_x, iteration_range=(0, best_iter + 1)) / args.folds

        importances.append(
            pd.DataFrame(
                {
                    "feature": x.columns,
                    "importance": model.feature_importances_,
                    "fold": fold,
                }
            )
        )

        del model, x_train, y_train, x_valid, y_valid, valid_pred, sample_weight
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

        test_proba_df = sample[[ID_COL]].copy()
        for idx, cls in enumerate(labels.classes_):
            test_proba_df[f"proba_{cls}"] = test_pred[:, idx]
        test_proba_df.to_csv(args.output_dir / "test_probabilities.csv", index=False)
        print(f"wrote_test_probabilities={args.output_dir / 'test_probabilities.csv'} rows={len(test_proba_df)}")


if __name__ == "__main__":
    main()
