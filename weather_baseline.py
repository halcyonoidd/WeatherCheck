# -*- coding: utf-8 -*-
"""
Weather Forecast Dataset: Machine Learning Baseline Example (Homework base code, must be run and understood first)[cite: 1]

Dataset: Kaggle Weather Forecast Dataset (Author: zeeshier, Apache 2.0)[cite: 1]
  https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset
  2500 daily meteorological observations, 5 numerical features + binary target:[cite: 1]
    Temperature / Humidity / Wind_Speed / Cloud_Cover / Pressure  ->  Rain ('rain' / 'no rain')[cite: 1]
  Note two points (also the "pitfalls" in this homework):[cite: 1]
    1. Highly imbalanced classes: no rain ≈ 87.6%, rain ≈ 12.4%;[cite: 1]
    2. The 5 input features are almost uncorrelated with each other (correlation ≈ 0), meaning PCA dimensionality reduction offers limited benefits.[cite: 1]

Script flow (compare with your lab report structure):[cite: 1]
  1. Data Loading + Exploratory Data Analysis (EDA)      -> Figure fig_w1_eda.png[cite: 1]
  2. Standardization + PCA Dimensionality Reduction & Visualization       -> Figure fig_w2_pca.png[cite: 1]
  3. Logistic Regression Baseline (5-fold Cross-Validation)    -> Terminal metric comparison table[cite: 1]
     - Original 5 features  vs  PCA 2D  vs  PCA 3D[cite: 1]
     - Provides a "predict all no rain" dummy baseline as a reference[cite: 1]
  4. Hint: Why shouldn't we only look at accuracy?[cite: 1]

Run: python weather_baseline.py[cite: 1]
Dependencies: numpy pandas matplotlib scikit-learn (see requirements.txt)[cite: 1]
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

# Chinese fonts (macOS; Windows can be changed to SimHei, Linux can be changed to Noto Sans CJK SC)[cite: 1]
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, RED, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"
N_SPLITS = 5
RANDOM_STATE = 42
CLS_COLORS = {"no rain": BLUE, "rain": ORANGE}
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

# ================================================================
# 1. Data Loading + EDA[cite: 1]
# ================================================================
df = pd.read_csv("weather_forecast_data.csv")
X = df.drop(columns=["Rain"])
y = df["Rain"].map({"no rain": 0, "rain": 1}).to_numpy()

print("=" * 62)
print("1) Data Overview") #[cite: 1]
print("-" * 62)
print(f"Number of samples: {df.shape[0]}   Number of features (numerical): {df.shape[1] - 1}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()
print("Target Rain distribution:") #[cite: 1]
print(df["Rain"].value_counts())
print(f"\nrain proportion = {df['Rain'].eq('rain').mean()*100:.1f}%  "
      f"-> Class imbalance, relying solely on accuracy will be severely misleading!") #[cite: 1]
print()
print("Feature descriptive statistics:") #[cite: 1]
print(df.describe().round(3).to_string())
print()
print("Pairwise feature correlation (≈0 indicates features are almost orthogonal, limited benefit from PCA):") #[cite: 1]
print(df.corr(numeric_only=True).round(3).to_string())

# ---------- EDA Plots ----------[cite: 1]
fig1, axes = plt.subplots(2, 2, figsize=(12, 9))

# (1) Target class distribution[cite: 1]
vc = df["Rain"].value_counts()
bars = axes[0, 0].bar(vc.index, vc.values, color=[BLUE, ORANGE])
for b, k in zip(bars, vc.index):
    axes[0, 0].text(b.get_x() + b.get_width() / 2, b.get_height() + 20,
                    f"{vc[k]}  ({vc[k]/len(df)*100:.1f}%)", ha="center", fontsize=12)
axes[0, 0].set_ylim(0, 2600)
axes[0, 0].set_title("Target Class Distribution (Severely Imbalanced)", fontsize=13) #[cite: 1]

# (2) Feature distribution histograms[cite: 1]
for c in X.columns:
    axes[0, 1].hist(df[c], bins=40, alpha=0.45, label=c)
axes[0, 1].set_title("Distribution of 5 Numerical Features", fontsize=13) #[cite: 1]
axes[0, 1].legend(fontsize=9)
axes[0, 1].set_xlabel("Values") #[cite: 1]

# (3) Correlation heatmap[cite: 1]
corr = df.corr(numeric_only=True)
im = axes[1, 0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1, 0].set_xticks(range(5)); axes[1, 0].set_yticks(range(5))
axes[1, 0].set_xticklabels(corr.columns, rotation=45); axes[1, 0].set_yticklabels(corr.columns)
for i in range(5):
    for j in range(5):
        axes[1, 0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
axes[1, 0].set_title("Feature Correlation (≈0 -> Features are orthogonal)", fontsize=13) #[cite: 1]
plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

# (4) Boxplot (scatter) for the two features most correlated with the target by class[cite: 1]
axes[1, 1].scatter(df["Humidity"], df["Cloud_Cover"], c=df["Rain"].map(CLS_COLORS),
                   alpha=0.45, s=16)
axes[1, 1].set_xlabel("Humidity"); axes[1, 1].set_ylabel("Cloud_Cover")
axes[1, 1].set_title("Humidity vs Cloud_Cover (Orange=rain, Blue=no rain)", fontsize=13) #[cite: 1]

fig1.suptitle("EDA: What's in the Box? (Weather Dataset)", fontsize=15, y=0.99) #[cite: 1]
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig("fig_w1_eda.png", dpi=150, bbox_inches="tight")
print("\nSaved fig_w1_eda.png") #[cite: 1]

# ================================================================
# 2. Standardization + PCA[cite: 1]
# ================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA().fit(X_scaled)
ratio = pca_full.explained_variance_ratio_
cum = np.cumsum(ratio)

print("\n" + "=" * 62)
print("2) PCA Results") #[cite: 1]
print("-" * 62)
for i, (r, c) in enumerate(zip(ratio, cum), 1):
    print(f"PC{i}: Explained variance ratio {r*100:6.2f}%   Cumulative {c*100:6.2f}%")
print(f"\n-> The proportions of each principal component are close, indicating data is evenly distributed across 5 dimensions; "
      f"reducing to 2D retains only about {cum[1]*100:.0f}% of the information.") #[cite: 1]
print("  (Comparison: For the Iris dataset, PC1 alone retains ~70%+, because its features are highly correlated.)") #[cite: 1]

# ---------- PCA Visualization Plot ----------[cite: 1]
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)

fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for label, color in CLS_COLORS.items():
    mask = df["Rain"] == label
    axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1], c=color,
                    s=12, alpha=0.55, label=label)
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
axes[0].set_title("PCA 2D Projection (colored by Rain)", fontsize=13) #[cite: 1]
axes[0].legend()

axes[1].bar(range(1, 6), ratio * 100, color=BLUE, alpha=0.8, label="Individual Principal Component") #[cite: 1]
axes[1].plot(range(1, 6), cum * 100, "o-", color=ORANGE, lw=2, label="Cumulative") #[cite: 1]
for i in range(5):
    axes[1].text(i + 1, ratio[i] * 100 + 1.2, f"{ratio[i]*100:.1f}%", ha="center", fontsize=9)
axes[1].set_xlabel("Principal Component"); axes[1].set_ylabel("Explained Variance Ratio (%)") #[cite: 1]
axes[1].set_title("Explained Variance Ratio by Principal Component", fontsize=13) #[cite: 1]
axes[1].legend()
fig2.tight_layout()
fig2.savefig("fig_w2_pca.png", dpi=150, bbox_inches="tight")
print("Saved fig_w2_pca.png") #[cite: 1]

# ================================================================
# 3. Logistic Regression Baseline: Original Features vs PCA (Stratified 5-Fold CV)[cite: 1]
# ================================================================
# Note: Fit scaler/PCA within the training fold for each CV iteration; 
# do not apply dimensionality reduction on the entire dataset beforehand (data leakage)![cite: 1]
# This uses the simplest approach for readability: refitting within the fold.[cite: 1]
def evaluate_cv(X_all, y_all, build_model):
    """Evaluate on StratifiedKFold, return mean±std for each metric.""" #[cite: 1]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = {m: [] for m in METRIC_NAMES}
    for tr, te in cv.split(X_all, y_all):
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]
        model = build_model()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        scores["Accuracy"].append(accuracy_score(y_te, y_pred))
        scores["Precision"].append(precision_score(y_te, y_pred, zero_division=0))
        scores["Recall"].append(recall_score(y_te, y_pred, zero_division=0))
        scores["F1"].append(f1_score(y_te, y_pred, zero_division=0))
        scores["ROC-AUC"].append(roc_auc_score(y_te, y_prob))
        scores["PR-AUC"].append(average_precision_score(y_te, y_prob))
    return {m: (np.mean(v), np.std(v)) for m, v in scores.items()}

def format_row(name, scores):
    parts = [f"{name:<28}"]
    for m in METRIC_NAMES:
        mean, std = scores[m]
        parts.append(f"{mean*100:6.1f}±{std*100:4.1f}")
    return "  ".join(parts)

print("\n" + "=" * 62)
print("3) Baseline Model: LogisticRegression (StratifiedKFold=5), Metrics=Mean±Std(%)") #[cite: 1]
print("-" * 62)
print(f"{'Model':<28}  {'ACC':>11}  {'PRE':>11}  {'REC':>11}  {'F1':>11}  {'AUROC':>11}  {'AUPRC':>11}") #[cite: 1]

# Reference: Learn nothing, predict majority class for all (no rain)[cite: 1]
dummy = evaluate_cv(X_scaled, y, lambda: DummyClassifier(strategy="most_frequent"))
print(format_row("Dummy (predict all no rain)", dummy)) #[cite: 1]

# Original 5 features[cite: 1]
lr_raw = evaluate_cv(X_scaled, y, lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
print(format_row("LogReg @ Original 5 features", lr_raw)) #[cite: 1]

# PCA to 2 / 3 dimensions (done within fold to avoid data leakage)[cite: 1]
from sklearn.pipeline import Pipeline
for k in (2, 3):
    def build_p(kk=k):
        return Pipeline([
            ("pca", PCA(n_components=kk)),
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ])
    res = evaluate_cv(X_scaled, y, build_p)
    print(format_row(f"LogReg @ PCA {k}D", res)) #[cite: 1]

print("-" * 62)
print("Observations:") #[cite: 1]
print("  1. Dummy Accuracy ≈ 87.4%, but F1 / Recall = 0 — please understand why;") #[cite: 1]
print("  2. With a default threshold of 0.5, the baseline's Recall is low (rain class is hard to recall), making PR-AUC more reliable;") #[cite: 1]
print("  3. After reducing to 2D via PCA, F1 drops from 69.6% to 25.1% — because each orthogonal feature accounts for ~20% of the variance,") #[cite: 1]
print("     losing 3 dimensions loses ~60% of the information; this sharply contrasts with highly correlated data like Iris;") #[cite: 1]
print("  4. Since features are already orthogonal, blindly applying PCA is ill-advised -> you can investigate feature engineering/supervised dimensionality reduction (LDA), etc.") #[cite: 1]

# ================================================================
# 4. Hints for students (Baseline ends here, homework is below)[cite: 1]
# ================================================================
print("\n" + "=" * 62)
print("4) Baseline complete. Next is your homework: investigate methods and improve (see weather_homework.md for details)") #[cite: 1]
print("-" * 62)
print("General directions for improvements to investigate (choose at least 3, clearly state your rationale and comparisons):") #[cite: 1]
print("  A. Class Imbalance: class_weight / SMOTE / Resampling / Adjusting decision threshold (PR Curve)") #[cite: 1]
print("  B. Models: Random Forest / Gradient Boosting Trees (XGBoost/LightGBM) / SVM (RBF) / KNN and hyperparameter tuning") #[cite: 1]
print("  C. Feature Engineering: Interaction features, binning, domain knowledge features (e.g., Humidity × Cloud Cover)") #[cite: 1]
print("  D. Dimensionality Reduction / Feature Selection Comparison: PCA vs LDA vs SelectKBest, exploring differences between supervised/unsupervised") #[cite: 1]
print("  E. Evaluation Framework: PR Curve vs ROC Curve (which is more reliable when imbalanced?), Confusion Matrix") #[cite: 1]
print("  F. Hyperparameter Tuning: GridSearchCV / RandomizedSearchCV / Optuna, report optimal hyperparameters") #[cite: 1]
print("  G. Cross-Validation: RepeatedStratifiedKFold stability analysis") #[cite: 1]