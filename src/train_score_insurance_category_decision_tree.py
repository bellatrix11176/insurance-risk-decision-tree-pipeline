from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# =============================================================================
# Insurance Category — Decision Tree (Train + Score + Evidence Locker)
# =============================================================================

LABEL_COL = "InsuranceCategory"
ID_COL = "CustomerID"

# -----------------------------
# Repo-friendly paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Decision Tree
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _find_csv_by_keywords(data_dir: Path, keywords: list[str]) -> Optional[Path]:
    """Return the first CSV path whose filename contains ALL keywords (case-insensitive)."""
    if not data_dir.exists():
        return None
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        return None

    for p in csvs:
        name = p.name.lower()
        if all(k.lower() in name for k in keywords):
            return p
    return None


def _infer_train_score_paths(data_dir: Path) -> Tuple[Path, Path]:
    """
    Try to locate training + scoring CSVs without requiring renames.

    Order of attempts:
      1) Preferred "nice" filenames.
      2) Keyword-based filenames containing Decision Tree + Training/Scoring.
      3) Fallback: infer training as CSV containing LABEL_COL.
    """
    # 1) Preferred names
    preferred_train = data_dir / "insurance_risk_category_train.csv"
    preferred_score = data_dir / "insurance_risk_category_score.csv"
    if preferred_train.exists() and preferred_score.exists():
        return preferred_train, preferred_score

    # 2) Keyword-ish names
    train_guess = _find_csv_by_keywords(data_dir, ["decision", "tree", "training"])
    score_guess = _find_csv_by_keywords(data_dir, ["decision", "tree", "scoring"])
    if train_guess and score_guess:
        return train_guess, score_guess

    # 3) Fallback: label-based inference
    csvs = sorted(data_dir.glob("*.csv"))
    if len(csvs) < 2:
        raise FileNotFoundError(
            f"Expected at least two CSV files in: {data_dir}\n"
            f"Found: {[p.name for p in csvs]}"
        )

    train_path = None
    score_path = None

    for p in csvs:
        try:
            cols = pd.read_csv(p, nrows=5).columns
        except Exception:
            continue
        if LABEL_COL in cols:
            train_path = p
        else:
            if score_path is None:
                score_path = p

    if train_path is None:
        raise FileNotFoundError(
            f"Could not find a training CSV containing label column '{LABEL_COL}' in {data_dir}.\n"
            f"CSV files found: {[p.name for p in csvs]}"
        )

    if score_path is None or score_path == train_path:
        others = [p for p in csvs if p != train_path]
        score_path = others[0]

    return train_path, score_path


def main() -> None:
    # -----------------------------
    # Auto-locate files (no rename required)
    # -----------------------------
    train_path, score_path = _infer_train_score_paths(DATA_DIR)

    # -----------------------------
    # Load data
    # -----------------------------
    train = pd.read_csv(train_path)
    score = pd.read_csv(score_path)

    if LABEL_COL not in train.columns:
        raise ValueError(f"Training file is missing label column: {LABEL_COL} (File: {train_path.name})")
    if ID_COL not in train.columns or ID_COL not in score.columns:
        raise ValueError(f"Expected ID column '{ID_COL}' in both train and score.")

    feature_cols = [c for c in train.columns if c not in [LABEL_COL, ID_COL]]

    missing_in_score = sorted(set(feature_cols) - set(score.columns))
    if missing_in_score:
        raise ValueError(f"Scoring data is missing these feature columns: {missing_in_score}")

    # -----------------------------
    # Range filter scoring data (numeric columns only)
    # -----------------------------
    numeric_cols = [c for c in feature_cols if is_numeric_series(train[c])]

    for c in numeric_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        score[c] = pd.to_numeric(score[c], errors="coerce")

    ranges = train[numeric_cols].agg(["min", "max"]).T
    ranges.index.name = "feature"
    ranges = ranges.rename(columns={"min": "train_min", "max": "train_max"})
    ranges.to_csv(OUT_DIR / "training_feature_ranges_numeric.csv")

    in_range_mask = pd.Series(True, index=score.index)
    out_of_range_summary = []

    for c in numeric_cols:
        cmin = float(ranges.loc[c, "train_min"])
        cmax = float(ranges.loc[c, "train_max"])
        below = score[c] < cmin
        above = score[c] > cmax

        out_of_range_summary.append(
            {
                "feature": c,
                "train_min": cmin,
                "train_max": cmax,
                "score_below_min_count": int(below.sum()),
                "score_above_max_count": int(above.sum()),
                "score_out_of_range_count": int((below | above).sum()),
            }
        )

        in_range_mask &= score[c].between(cmin, cmax, inclusive="both") & score[c].notna()

    out_of_range_df = pd.DataFrame(out_of_range_summary).sort_values(
        by="score_out_of_range_count", ascending=False
    )
    out_of_range_df.to_csv(OUT_DIR / "scoring_out_of_range_summary.csv", index=False)

    score_in_range = score[in_range_mask].copy()
    score_removed = score[~in_range_mask].copy()
    if not score_removed.empty:
        score_removed.to_csv(OUT_DIR / "scoring_rows_removed_out_of_range.csv", index=False)

    # -----------------------------
    # Build preprocessing pipeline (shared across experiments)
    # -----------------------------
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    # -----------------------------
    # Prepare training data
    # -----------------------------
    X_train = train[feature_cols].copy()
    y_train = train[LABEL_COL].astype(str)

    mask_label = y_train.notna()
    X_train = X_train.loc[mask_label]
    y_train = y_train.loc[mask_label]

    # =============================================================================
    # Model selection experiment (Python-only): compare split criteria
    # =============================================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidate_pipes = {
        "gini": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("tree", DecisionTreeClassifier(criterion="gini", random_state=42)),
            ]
        ),
        "entropy": Pipeline(
            steps=[
                ("prep", preprocessor),
                ("tree", DecisionTreeClassifier(criterion="entropy", random_state=42)),
            ]
        ),
    }

    criterion_rows = []
    for name, model in candidate_pipes.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        criterion_rows.append(
            {
                "criterion": name,
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
                "fold_accuracies": ",".join([f"{s:.5f}" for s in scores]),
            }
        )

    criterion_results = pd.DataFrame(criterion_rows).sort_values("mean_accuracy", ascending=False)
    criterion_results.to_csv(OUT_DIR / "criterion_comparison_cv.csv", index=False)

    best_criterion = str(criterion_results.iloc[0]["criterion"])
    best_cv_mean = float(criterion_results.iloc[0]["mean_accuracy"])
    best_cv_std = float(criterion_results.iloc[0]["std_accuracy"])

    pipe = candidate_pipes[best_criterion]
    pipe.fit(X_train, y_train)

    # -----------------------------
    # Training performance
    # -----------------------------
    y_pred_train = pipe.predict(X_train)
    acc = float(accuracy_score(y_train, y_pred_train))

    clf: DecisionTreeClassifier = pipe.named_steps["tree"]
    classes = list(clf.classes_)

    cm = confusion_matrix(y_train, y_pred_train, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in classes],
        columns=[f"Pred_{c}" for c in classes],
    )
    cm_df.to_csv(OUT_DIR / "confusion_matrix_train.csv")

    with open(OUT_DIR / "classification_report_train.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_train, y_pred_train))

    pd.DataFrame(
        {
            "metric": ["accuracy_train", "cv_mean_accuracy_best", "cv_std_accuracy_best", "selected_criterion"],
            "value": [acc, best_cv_mean, best_cv_std, best_criterion],
        }
    ).to_csv(OUT_DIR / "performance_metrics_train.csv", index=False)

    # Confusion matrix plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix (Training) — Accuracy={acc:.3f}")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    save_fig(OUT_DIR / "confusion_matrix_train.png")

    # -----------------------------
    # Feature names after preprocessing (for tree visualization)
    # -----------------------------
    ohe_feature_names: list[str] = []
    if categorical_cols:
        ohe: OneHotEncoder = pipe.named_steps["prep"].named_transformers_["cat"]
        ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols))

    feature_names = numeric_cols + ohe_feature_names

    # Root feature (first split)
    root_feature_idx = int(clf.tree_.feature[0])
    root_feature = feature_names[root_feature_idx] if root_feature_idx >= 0 else "None"

    # Tree plots
    plt.figure(figsize=(18, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=[str(c) for c in classes],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8,
    )
    plt.title("Decision Tree (Top Levels)")
    save_fig(OUT_DIR / "decision_tree_top_levels.png")

    plt.figure(figsize=(24, 14))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=[str(c) for c in classes],
        filled=True,
        rounded=True,
        fontsize=6,
    )
    plt.title("Decision Tree (Full)")
    save_fig(OUT_DIR / "decision_tree_full.png")

    # Feature importances
    imp_df = pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
    imp_df = imp_df.sort_values("importance", ascending=False)
    imp_df.to_csv(OUT_DIR / "feature_importances.csv", index=False)

    plt.figure(figsize=(10, 6))
    top = imp_df.head(15).iloc[::-1]
    plt.barh(top["feature"], top["importance"])
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    save_fig(OUT_DIR / "feature_importances_top15.png")

    # -----------------------------
    # Score the scoring dataset (after range filtering)
    # -----------------------------
    X_score = score_in_range[feature_cols].copy()
    preds = pipe.predict(X_score)

    scored = pd.DataFrame(
        {
            ID_COL: score_in_range[ID_COL].values,
            "PredictedInsuranceCategory": preds,
        }
    )
    scored.to_csv(OUT_DIR / "insurance_category_predictions.csv", index=False)

    low_risk_count = int((scored["PredictedInsuranceCategory"].astype(str).str.lower() == "low risk").sum())
    pred_counts = scored["PredictedInsuranceCategory"].value_counts(dropna=False).reset_index()
    pred_counts.columns = ["PredictedInsuranceCategory", "count"]
    pred_counts.to_csv(OUT_DIR / "prediction_counts.csv", index=False)

    # -----------------------------
    # Minimal run summary (useful for Colab/GitHub)
    # -----------------------------
    print("\n=== Insurance Category Decision Tree — Run Summary ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Training file: {train_path.name}")
    print(f"Scoring file:  {score_path.name}")
    print(f"Selected criterion (CV): {best_criterion} (mean={best_cv_mean:.4f}, std={best_cv_std:.4f})")
    print(f"Training accuracy: {acc:.4f}")
    print(f"Root split feature (first predictor in tree): {root_feature}")
    print(f"Scoring rows removed (out-of-range numeric): {len(score_removed)}")
    print(f"Low Risk predictions in scoring: {low_risk_count}")
    print(f"Outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
