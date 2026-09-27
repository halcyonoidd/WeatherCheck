# -*- coding: utf-8 -*-
"""
Weather Forecast Dataset - Improvement Experiments

Improvements tested against the baseline (StratifiedKFold=5, random_state=42):
    A1 - class_weight="balanced"
    A2 - SMOTE oversampling
    C1 - LDA (supervised dimensionality reduction)
    D1 - Humidity x Cloud_Cover interaction feature

All preprocessing (scaling, PCA/LDA, SMOTE) is fit inside each CV fold only,
to avoid data leakage.
"""

from pathlib import Path
import argparse
import importlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve,
)

try:
    SMOTE = importlib.import_module("imblearn.over_sampling").SMOTE
    ImbPipeline = importlib.import_module("imblearn.pipeline").Pipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    IMBLEARN_AVAILABLE = False


DATA_FILE = "weather_forecast_data.csv"
OUTPUT_DIR = Path("weather_results")
N_SPLITS = 5
RANDOM_STATE = 42

FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]
TARGET = "Rain"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]


def load_data(data_file):
    """Load the CSV and map the target to {no rain: 0, rain: 1}."""
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    required_columns = FEATURES + [TARGET]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df[required_columns].copy()

    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values.")

    df[TARGET] = df[TARGET].astype(str).str.lower().map({"no rain": 0, "rain": 1})
    if df[TARGET].isnull().any():
        raise ValueError("Unrecognized Rain label; only 'rain' / 'no rain' allowed.")

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    return df, X, y


def run_eda(df, X, y):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("1) DATA OVERVIEW")
    print("=" * 70)
    print(f"Number of samples : {df.shape[0]}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Missing values    : {df.isnull().sum().sum()}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts())
    print(f"\nRain proportion: {y.mean() * 100:.2f}%")
    print("\nDescriptive statistics:")
    print(X.describe().round(3).to_string())

    correlation = X.corr()
    print("\nFeature correlation matrix:")
    print(correlation.round(3).to_string())

    X.describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")
    correlation.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    class_counts = y.value_counts().sort_index()
    class_table = pd.DataFrame({
        "Class": ["no rain", "rain"],
        "Count": [class_counts.get(0, 0), class_counts.get(1, 0)],
    })
    class_table["Percentage"] = class_table["Count"] / len(y) * 100
    class_table.to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(["no rain", "rain"], [class_counts.get(0, 0), class_counts.get(1, 0)])
    plt.title("Target Class Distribution")
    plt.xlabel("Rain Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_class_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson Correlation")
    plt.xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
    plt.yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            plt.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_correlation_matrix.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURES):
        axes[i].hist(X[feature], bins=30)
        axes[i].set_title(feature)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Frequency")
    axes[-1].axis("off")
    fig.suptitle("Feature Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_feature_distributions.png", dpi=200)
    plt.close(fig)


def run_pca_analysis(X):
    """PCA explained variance, computed on the full data for EDA purposes only
    (CV models fit PCA inside each fold separately)."""
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print("\n" + "=" * 70)
    print("2) PCA ANALYSIS")
    print("=" * 70)
    for i, (ind, cum) in enumerate(zip(explained, cumulative), start=1):
        print(f"PC{i}: {ind * 100:6.2f}% (Cumulative: {cum * 100:6.2f}%)")

    pd.DataFrame({
        "Principal Component": range(1, len(explained) + 1),
        "Explained Variance": explained,
        "Cumulative Variance": cumulative,
    }).to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)

    components = np.arange(1, len(explained) + 1)
    plt.figure(figsize=(7, 5))
    plt.bar(components, explained * 100, alpha=0.7, label="Individual")
    plt.plot(components, cumulative * 100, marker="o", label="Cumulative")
    plt.xticks(components)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA Explained Variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_pca_explained_variance.png", dpi=200)
    plt.close()


class AddHumidityCloudInteraction(BaseEstimator, TransformerMixin):
    """Adds a Humidity x Cloud_Cover interaction feature."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != 5:
            raise ValueError("Expected a 2D array with exactly 5 features.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        interaction = (X[:, 1] * X[:, 3]).reshape(-1, 1)
        return np.hstack([X, interaction])


def build_models():
    models = {
        "Dummy": Pipeline([
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]),
        "Baseline_LogReg_5F": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Baseline_LogReg_PCA2": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Baseline_LogReg_PCA3": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=3)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        # A1: cost-sensitive learning
        "A1_ClassWeight_Balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        # C1: supervised dimensionality reduction (LDA) vs. unsupervised PCA
        "C1_LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(n_components=1)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        # D1: feature engineering
        "D1_Humidity_x_Cloud": Pipeline([
            ("interaction", AddHumidityCloudInteraction()),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
    }

    # A2: resampling (only if imbalanced-learn is installed)
    if IMBLEARN_AVAILABLE:
        models["A2_SMOTE"] = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ])

    return models


def evaluate_model(model, X, y, splits):
    """Evaluate a model on pre-computed StratifiedKFold splits. The pipeline
    is fit fresh on each training fold, so no preprocessing step ever sees
    the held-out fold."""
    fold_results = []
    oof_pred = np.zeros(len(y), dtype=int)
    oof_prob = np.zeros(len(y), dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        current_model = clone(model)
        current_model.fit(X_train, y_train)

        y_pred = current_model.predict(X_test)
        y_prob = current_model.predict_proba(X_test)[:, 1]
        oof_pred[test_idx] = y_pred
        oof_prob[test_idx] = y_prob

        fold_results.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
        })

    return pd.DataFrame(fold_results), oof_pred, oof_prob


def create_summary_table(all_results):
    rows = []
    for model_name, result in all_results.items():
        fold_df = result["folds"]
        row = {"Model": model_name}
        for metric in METRICS:
            row[f"{metric}_mean"] = fold_df[metric].mean()
            row[f"{metric}_std"] = fold_df[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_table(summary_df):
    print("\n" + "=" * 120)
    print("OVERALL MODEL COMPARISON")
    print("=" * 120)
    print(f"{'Model':<30}{'ACC':>15}{'Precision':>15}{'Recall':>15}{'F1':>15}{'ROC-AUC':>15}{'PR-AUC':>15}")
    print("-" * 120)
    for _, row in summary_df.iterrows():
        print(
            f"{row['Model']:<30}"
            f"{row['Accuracy_mean'] * 100:6.2f}±{row['Accuracy_std'] * 100:4.2f}"
            f"{row['Precision_mean'] * 100:6.2f}±{row['Precision_std'] * 100:4.2f}"
            f"{row['Recall_mean'] * 100:6.2f}±{row['Recall_std'] * 100:4.2f}"
            f"{row['F1_mean'] * 100:6.2f}±{row['F1_std'] * 100:4.2f}"
            f"{row['ROC-AUC_mean'] * 100:6.2f}±{row['ROC-AUC_std'] * 100:4.2f}"
            f"{row['PR-AUC_mean'] * 100:6.2f}±{row['PR-AUC_std'] * 100:4.2f}"
        )


def save_fold_f1_comparison(baseline_folds, improvement_folds, improvement_name, file_name):
    folds = np.arange(1, len(baseline_folds) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(folds, baseline_folds["F1"] * 100, marker="o", label="Baseline")
    plt.plot(folds, improvement_folds["F1"] * 100, marker="o", label=improvement_name)
    plt.xlabel("CV Fold")
    plt.ylabel("F1 (%)")
    plt.title(f"F1 Comparison: Baseline vs {improvement_name}")
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=200)
    plt.close()


def save_confusion_matrix(y, predictions, model_name):
    cm = confusion_matrix(y, predictions)
    cm_table = pd.DataFrame(
        cm,
        index=["Actual no rain", "Actual rain"],
        columns=["Predicted no rain", "Predicted rain"],
    )
    cm_table.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    print("\n" + "=" * 70)
    print(f"CONFUSION MATRIX - {model_name}")
    print("=" * 70)
    print(cm_table)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], ["Predicted no rain", "Predicted rain"])
    plt.yticks([0, 1], ["Actual no rain", "Actual rain"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_confusion_matrix.png", dpi=200)
    plt.close()


def save_misclassified_samples(df, y, predictions, probabilities):
    """Save up to 3 examples: the most confident false positive, the most
    confident false negative, and one additional error."""
    y_array = y.to_numpy()
    wrong = np.where(y_array != predictions)[0]

    selected = []
    fp = [i for i in wrong if y_array[i] == 0 and predictions[i] == 1]
    if fp:
        selected.append(max(fp, key=lambda i: probabilities[i]))
    fn = [i for i in wrong if y_array[i] == 1 and predictions[i] == 0]
    if fn:
        selected.append(min(fn, key=lambda i: probabilities[i]))
    for i in wrong:
        if i not in selected:
            selected.append(i)
        if len(selected) >= 3:
            break

    if not selected:
        print("\nNo misclassified samples found.")
        return

    samples = df.iloc[selected].copy()
    samples["True_Label"] = ["rain" if y_array[i] == 1 else "no rain" for i in selected]
    samples["Predicted_Label"] = ["rain" if predictions[i] == 1 else "no rain" for i in selected]
    samples["Predicted_Rain_Probability"] = [probabilities[i] for i in selected]
    samples.to_csv(OUTPUT_DIR / "misclassified_samples.csv", index=False)

    print("\n" + "=" * 70)
    print("TYPICAL MISCLASSIFIED SAMPLES")
    print("=" * 70)
    print(samples.to_string(index=False))


def save_pr_curve(y, probabilities, model_name):
    precision, recall, _ = precision_recall_curve(y, probabilities)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_final_pr_curve.png", dpi=200)
    plt.close()


def select_best_model(summary_df):
    """Highest mean F1, PR-AUC as tie-breaker. Accuracy is not used as the
    primary criterion since the dataset is imbalanced."""
    sorted_df = summary_df.sort_values(by=["F1_mean", "PR-AUC_mean"], ascending=False)
    return sorted_df.iloc[0]["Model"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_FILE, help="Path to weather_forecast_data.csv")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WEATHER FORECAST - MACHINE LEARNING IMPROVEMENT")
    print("=" * 70)
    print(f"Dataset      : {args.data}")
    print(f"CV folds     : {N_SPLITS}")
    print(f"Random state : {RANDOM_STATE}")
    if not IMBLEARN_AVAILABLE:
        print("WARNING: imbalanced-learn not found, skipping A2_SMOTE.")

    df, X, y = load_data(args.data)
    run_eda(df, X, y)
    run_pca_analysis(X)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))

    models = build_models()
    all_results = {}

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Evaluating: {model_name}")
        print("=" * 70)

        fold_df, oof_pred, oof_prob = evaluate_model(model, X, y, splits)
        print(fold_df.to_string(index=False))

        all_results[model_name] = {"folds": fold_df, "predictions": oof_pred, "probabilities": oof_prob}
        safe_name = model_name.replace(" ", "_").replace("/", "_")
        fold_df.to_csv(OUTPUT_DIR / f"fold_metrics_{safe_name}.csv", index=False)

    summary_df = create_summary_table(all_results)
    summary_df.to_csv(OUTPUT_DIR / "overall_comparison.csv", index=False)
    print_summary_table(summary_df)

    baseline_folds = all_results["Baseline_LogReg_5F"]["folds"]
    comparison_models = [
        ("A1_ClassWeight_Balanced", "06_A1_class_weight_vs_baseline.png"),
        ("A2_SMOTE", "07_A2_smote_vs_baseline.png"),
        ("C1_LDA", "08_C1_lda_vs_baseline.png"),
        ("D1_Humidity_x_Cloud", "09_D1_interaction_vs_baseline.png"),
    ]
    for model_name, file_name in comparison_models:
        if model_name in all_results:
            save_fold_f1_comparison(baseline_folds, all_results[model_name]["folds"], model_name, file_name)

    best_model_name = select_best_model(summary_df)
    best_result = all_results[best_model_name]
    best_row = summary_df[summary_df["Model"] == best_model_name].iloc[0]

    print("\n" + "=" * 70)
    print("FINAL MODEL")
    print("=" * 70)
    print(f"Selected model: {best_model_name}")
    print(f"Mean F1    : {best_row['F1_mean'] * 100:.2f}% ± {best_row['F1_std'] * 100:.2f}%")
    print(f"Mean PR-AUC: {best_row['PR-AUC_mean'] * 100:.2f}% ± {best_row['PR-AUC_std'] * 100:.2f}%")

    save_confusion_matrix(y, best_result["predictions"], best_model_name)
    save_misclassified_samples(df, y, best_result["predictions"], best_result["probabilities"])
    save_pr_curve(y, best_result["probabilities"], best_model_name)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"All results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()