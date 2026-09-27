# Weather Forecast Dataset — Code Package

This folder contains everything needed to reproduce the baseline and the improvement experiments.

## Files

| File | Description |
|---|---|
| `weather_forecast_data.csv` | Kaggle Weather Forecast dataset (2,500 rows, 5 features + `Rain` target). |
| `weather_baseline.py` | **Unmodified** baseline script provided with the assignment. Reproduces the EDA, PCA analysis, and the Dummy / LogReg 5-feature / LogReg PCA-2D / LogReg PCA-3D comparison. |
| `weather_improvement.py` | Improvement experiments (A1 `class_weight="balanced"`, A2 SMOTE, C1 LDA, D1 Humidity × Cloud_Cover interaction), evaluated under the exact same StratifiedKFold(5, seed=42) protocol as the baseline. |

## 1. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn
```

(`imbalanced-learn` is only required for the SMOTE experiment, A2. If it is missing, `weather_improvement.py` will still run — it just prints a warning and skips A2.)

## 2. Run the baseline (unchanged, run this first)

```bash
python weather_baseline.py
```

Reads `weather_forecast_data.csv` from the current folder and prints the data overview, PCA results, and the baseline metrics table to the terminal, plus saves `fig_w1_eda.png` and `fig_w2_pca.png`.

## 3. Run the improvement experiments

```bash
python weather_improvement.py
```

or, to point at a dataset file in a different location:

```bash
python weather_improvement.py --data path/to/weather_forecast_data.csv
```

This will:
1. Re-run the EDA and PCA analysis (for self-contained reporting).
2. Evaluate the Dummy and baseline LogReg models (5 features, PCA-2D, PCA-3D) plus the four improvements (A1, A2, C1, D1) — all under identical `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` splits.
3. Print an overall comparison table (mean ± std for Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) to the terminal.
4. Select the best model (highest mean F1, PR-AUC as tie-breaker) and print its confusion matrix and 3 typical misclassified samples.
5. Save every table (CSV) and figure (PNG) into a new `weather_results/` folder.

## Output (`weather_results/`)

- `class_distribution.csv`, `correlation_matrix.csv`, `descriptive_statistics.csv`, `pca_explained_variance.csv`
- `fold_metrics_<model>.csv` — per-fold metrics for every model
- `overall_comparison.csv` — the final summary table
- `confusion_matrix.csv`, `misclassified_samples.csv`
- `01_class_distribution.png` … `10_final_pr_curve.png` — all report figures

## Notes

- `weather_baseline.py` was **not modified** in any way and remains runnable exactly as provided.
- All preprocessing (scaling, PCA/LDA, SMOTE) is fit **inside each CV fold only**, avoiding data leakage — see the pipelines in `build_models()` in `weather_improvement.py`.
- Both scripts use `random_state = 42` and identical 5-fold splits, so results from `weather_improvement.py` are directly comparable to `weather_baseline.py`.