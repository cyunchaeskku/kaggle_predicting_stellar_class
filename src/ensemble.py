#!/usr/bin/env python3
"""Ensemble OOF/test probabilities from multiple model output dirs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="Model output dirs, each containing oof_predictions.csv and test_probabilities.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--weighting",
        choices=["simple", "oof_weighted"],
        default="simple",
        help="simple: equal weights. oof_weighted: weight by per-model OOF balanced accuracy.",
    )
    parser.add_argument(
        "--tune-class-multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--multiplier-trials", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def load_and_validate(
    dirs: list[Path],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    oof_dfs = []
    test_dfs = []
    for d in dirs:
        oof_path = d / "oof_predictions.csv"
        test_path = d / "test_probabilities.csv"
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing: {oof_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing: {test_path}")
        oof_dfs.append(pd.read_csv(oof_path))
        test_dfs.append(pd.read_csv(test_path))

    proba_cols = [c for c in oof_dfs[0].columns if c.startswith("proba_")]
    ref_oof_ids = oof_dfs[0]["id"].to_numpy()
    ref_oof_true = oof_dfs[0]["true"].to_numpy()
    ref_test_ids = test_dfs[0]["id"].to_numpy()

    test0_proba_cols = [c for c in test_dfs[0].columns if c.startswith("proba_")]
    if test0_proba_cols != proba_cols:
        raise ValueError(f"dir 0 ({dirs[0]}): test probability columns mismatch with OOF")

    for i, (oof, test) in enumerate(zip(oof_dfs[1:], test_dfs[1:]), start=1):
        if not np.array_equal(oof["id"].to_numpy(), ref_oof_ids):
            raise ValueError(f"dir {i} ({dirs[i]}): OOF ids mismatch")
        if not np.array_equal(oof["true"].to_numpy(), ref_oof_true):
            raise ValueError(f"dir {i} ({dirs[i]}): OOF true labels mismatch")
        if not np.array_equal(test["id"].to_numpy(), ref_test_ids):
            raise ValueError(f"dir {i} ({dirs[i]}): test ids mismatch")
        oof_proba_cols = [c for c in oof.columns if c.startswith("proba_")]
        test_proba_cols = [c for c in test.columns if c.startswith("proba_")]
        if oof_proba_cols != proba_cols:
            raise ValueError(f"dir {i} ({dirs[i]}): OOF probability columns mismatch")
        if test_proba_cols != proba_cols:
            raise ValueError(f"dir {i} ({dirs[i]}): test probability columns mismatch")

    oof_probas = [df[proba_cols].to_numpy(dtype="float32") for df in oof_dfs]
    test_probas = [df[proba_cols].to_numpy(dtype="float32") for df in test_dfs]
    classes = [c.replace("proba_", "") for c in proba_cols]
    return oof_probas, test_probas, ref_oof_ids, ref_oof_true, ref_test_ids, classes


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof_probas, test_probas, oof_ids, oof_true, test_ids, classes = load_and_validate(args.dirs)

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_true = np.array([class_to_idx[t] for t in oof_true])

    model_scores = [
        balanced_accuracy_score(y_true, np.argmax(p, axis=1)) for p in oof_probas
    ]

    for d, score in zip(args.dirs, model_scores):
        print(f"model={d}  oof_balanced_accuracy={score:.6f}")

    if args.weighting == "simple":
        weights = np.ones(len(args.dirs), dtype="float32") / len(args.dirs)
    else:
        raw = np.array(model_scores, dtype="float32")
        weights = raw / raw.sum()

    for d, w in zip(args.dirs, weights):
        print(f"model={d}  weight={w:.4f}")

    ensemble_oof = np.zeros_like(oof_probas[0])
    ensemble_test = np.zeros_like(test_probas[0])
    for w, oof_p, test_p in zip(weights, oof_probas, test_probas):
        ensemble_oof += w * oof_p
        ensemble_test += w * test_p

    raw_score = balanced_accuracy_score(y_true, np.argmax(ensemble_oof, axis=1))
    print(f"ensemble_oof_balanced_accuracy={raw_score:.6f}")

    final_weights = np.ones(len(classes), dtype="float32")
    final_score = raw_score
    if args.tune_class_multipliers:
        final_weights, final_score = tune_class_multipliers(
            y_true=y_true,
            proba=ensemble_oof,
            seed=args.seed,
            trials=args.multiplier_trials,
        )
    print(f"ensemble_oof_balanced_accuracy_tuned={final_score:.6f}")
    print("class_multipliers=" + ",".join(f"{cls}:{w:.5f}" for cls, w in zip(classes, final_weights)))

    classes_arr = np.array(classes)
    oof_out = pd.DataFrame({
        "id": oof_ids,
        "true": oof_true,
        "pred": classes_arr[np.argmax(ensemble_oof * final_weights, axis=1)],
    })
    for idx, cls in enumerate(classes):
        oof_out[f"proba_{cls}"] = ensemble_oof[:, idx]
    oof_out.to_csv(args.output_dir / "ensemble_oof_predictions.csv", index=False)

    submission = pd.DataFrame({
        "id": test_ids,
        "class": classes_arr[np.argmax(ensemble_test * final_weights, axis=1)],
    })
    submission.to_csv(args.output_dir / "ensemble_submission.csv", index=False)
    print(f"wrote_submission={args.output_dir / 'ensemble_submission.csv'} rows={len(submission)}")

    report_lines = [
        f"weighting={args.weighting}",
        "",
        "models:",
    ]
    for d, score, w in zip(args.dirs, model_scores, weights):
        report_lines.append(f"  {d}  oof_score={score:.6f}  weight={w:.4f}")
    report_lines += [
        "",
        f"ensemble_oof_balanced_accuracy={raw_score:.6f}",
        f"ensemble_oof_balanced_accuracy_tuned={final_score:.6f}",
        "class_multipliers=" + ",".join(f"{cls}:{w:.5f}" for cls, w in zip(classes, final_weights)),
    ]
    (args.output_dir / "ensemble_report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"wrote_report={args.output_dir / 'ensemble_report.txt'}")


if __name__ == "__main__":
    main()
