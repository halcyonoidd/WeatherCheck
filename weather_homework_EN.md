# Machine Learning Lab Assignment: Precipitation Prediction on the Kaggle Weather Forecast Dataset

- Course topic: PCA-based dimensionality reduction and machine learning (continues the in-class PCA derivation and Iris clustering demos)
- Dataset: Kaggle *Weather Forecast Dataset* ([link](https://www.kaggle.com/datasets/zeeshier/weather-forecast-dataset), by zeeshier, Apache 2.0)
- Local file: `weather_forecast_data.csv` (shipped with the assignment; no Kaggle login needed)
- Baseline code: `weather_baseline.py` (run as-is)
- Due date / submission method: ____________ (to be filled in by the TA)

---

## 1. Learning Objectives

Carry out an end-to-end machine learning experiment, "from data to report", and specifically learn to:

1. **Follow a complete modeling workflow**: EDA → preprocessing → modeling → evaluation → improvement → report;
2. **Understand the principles and limits of PCA**: you derived PCA by hand in class; in this assignment you will find that PCA offers little when the features are already (nearly) orthogonal — and you should be able to explain why;
3. **Handle class imbalance**: rain accounts for only ~12.6% of the samples, so the default evaluation protocol (accuracy only, default 0.5 threshold) is seriously misleading — you must investigate methods and fix it;
4. **Do literature research**: read papers/documentation independently, turn the methods you find into rigorous comparison experiments, and present them in a well-structured report.

---

## 2. Dataset Description

| Column | Type | Meaning |
|---|---|---|
| Temperature | float | Air temperature (°C) |
| Humidity | float | Relative humidity (%) |
| Wind_Speed | float | Wind speed (m/s) |
| Cloud_Cover | float | Cloud cover (%) |
| Pressure | float | Atmospheric pressure (hPa) |
| **Rain** | str | Target: `rain` / `no rain` (binary classification) |

- 2,500 observations, no missing values;
- **Class imbalance**: no rain 2,186 (87.4%) vs. rain 314 (12.6%);
- The 5 input features are nearly uncorrelated with each other (pairwise |r| ≈ 0; see the correlation matrix printed by the baseline).

> ⚠️ Think in connection with the course material: "What is the assumption behind PCA? When features are already orthogonal, what can PCA still do for you, and what can it not do?" This is one of the hidden checkpoints of this assignment.

---

## 3. Deliverables

| No. | Deliverable | Description |
|---|---|---|
| D1 | Runnable code | Your own improvement code plus a short README explaining how to run it; the baseline must remain runnable |
| D2 | Lab report (PDF or Markdown) | Written according to the structure in §7, including all figures/tables and conclusions |
| D3 | Literature review notes | Can be presented as §3 of the report (no separate file needed) |

---

## 4. Phase 1: Reproduce and Understand the Baseline (~2 h, mandatory)

```bash
pip install -r requirements.txt    # or install numpy pandas matplotlib scikit-learn yourself
python weather_baseline.py
```

The script prints the data overview, the PCA results, and the baseline metrics below (StratifiedKFold = 5, mean ± std):

| Model | ACC | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy (always "no rain") | ≈87.4% | 0 | 0 | 0 | 0.500 | 0.126 |
| LogisticRegression @ 5 features | ≈93% | ≈76% | ≈64% | ≈70% | ≈97% | ≈82% |
| LogisticRegression @ PCA-2 | drops | … | … | drops sharply | … | … |
| LogisticRegression @ PCA-3 | in between | … | … | … | … | … |

Answer the following 4 questions in writing (they go into report §4):

1. **Why does the Dummy classifier achieve 87.4% accuracy yet have F1 = 0 and Recall = 0?** What does this tell you?
2. **Why does F1 drop sharply (from ~70% to ~25%) when PCA reduces to 2 dimensions?** Explain using the feature correlation matrix; then contrast with the Iris dataset from class (where PC1 alone explains ~73% of the variance) and summarize: *"on what kind of data is PCA useful?"*
3. **What implicit assumption does the default 0.5 probability threshold make?** Why is this assumption questionable under class imbalance? What do PR-AUC and ROC-AUC measure respectively, and which one is more sensitive to the minority class?
4. **Why does the baseline code standardize and apply PCA *inside each cross-validation fold*?** What would go wrong if you standardized on the full data first and then split into train/test? (Hint: data leakage.)

---

## 5. Phase 2: Literature Research (~3–5 h, mandatory)

Research **at least 3** of the directions below (we suggest 2 "method" directions + 1 "evaluation" direction). For each, write a 200–400 word summary of key points, and map it to one experiment in §6. **Do not just copy blog posts — read at least one original paper or the standard library documentation for each direction.**

| Direction | Questions to investigate | Starting references |
|---|---|---|
| A. Handling class imbalance | What are the differences and trade-offs between resampling, cost-sensitive learning, and threshold moving? What motivates SMOTE, and what are its drawbacks? | Chawla et al. 2002 (SMOTE); He & Garcia 2009 (survey); imbalanced-learn docs |
| B. Model family comparison | What are the relative weaknesses of linear models / tree models / kernel methods on an imbalanced problem with a non-linear decision boundary? | sklearn docs; the original paper of each model |
| C. Dimensionality reduction & feature selection | What is the difference in objective between PCA (unsupervised) and LDA (supervised)? When should you use feature selection instead of dimensionality reduction? | Jolliffe, *Principal Component Analysis*; sklearn docs |
| D. Feature engineering | What is the motivation and the risk of binning, interaction features, and domain-knowledge features? | sklearn `PolynomialFeatures` docs; relevant papers |
| E. Evaluation protocol | Under class imbalance, how trustworthy are PR curves vs. ROC curves? How do you choose a threshold? | Saito & Rehmsmeier 2015; confusion matrix & PR curve docs |
| F. Hyperparameter tuning & validation | Trade-offs between GridSearch and RandomSearch? What can validation/learning curves reveal? | sklearn `model_selection` docs |

**Submission requirement**: report §3 must contain a mapping table: "research summary (and relation to existing methods) → the improvement direction you chose and why."

---

## 6. Phase 3: Improvement Experiments (~5–8 h, mandatory)

### Hard requirements

1. **At least 3 independent improvements**, among which:
   - at least 1 must target **class imbalance** (methods from direction A or E);
   - at least 1 must be related to **dimensionality reduction / feature processing** (ties back to the PCA theme of the course; direction C or D);
2. **Same evaluation protocol as the baseline**: StratifiedKFold(5), same random seed, the same 6 metrics (F1 and PR-AUC are mandatory; Accuracy is only for reference);
3. **Control variables**: each improvement changes only one component — state in the report what was changed / not changed relative to the baseline;
4. **All preprocessing (standardization, dimensionality reduction, oversampling) must be fitted inside each CV fold** — never fit on the full data and then split;
5. Produce a **final overall comparison table** and a **confusion matrix**, and pick 2–3 typical misclassified samples to discuss why they fail.

### Reference improvement list (feel free to extend, but each must have an explainable motivation)

- **A1** `class_weight="balanced"` (cost-sensitive) vs. baseline;
- **A2** SMOTE oversampling + classifier (note: SMOTE may only be used inside training folds — never sample the validation set);
- **A3** Threshold tuning: search for the optimal decision threshold along the PR curve on validation folds; report a "threshold vs. F1" curve;
- **B1** Change the model: Random Forest / gradient boosting (sklearn `HistGradientBoostingClassifier`, or XGBoost/LightGBM) / SVM-RBF / KNN;
- **B2** Tune 1 selected model with GridSearchCV or RandomizedSearchCV (report the best hyperparameters);
- **C1** PCA vs. LDA dimensionality reduction comparison (LDA is supervised — think about how its objective differs from PCA) plus "classify after reduction" vs. "classify on original features";
- **C2** Feature selection (`SelectKBest` / `RFE`) vs. PCA;
- **D1** Feature engineering: add interaction features (e.g., `Humidity × Cloud_Cover`) and check whether performance improves;
- **E1** Evaluation analysis: plot both PR and ROC curves for the same model and discuss which curve exposes problems earlier;
- **F1** Learning curve / validation curve: judge whether the current model is underfitting or overfitting.

---

## 7. Report Template (follow this structure; grading follows it too)

> Aim for 8–12 pages including figures; write concisely.

1. **Abstract** (≤250 words: problem, methods, main results, conclusion)
2. **Problem and data**: task definition, column descriptions, statistical analysis (imbalance ratio, correlation matrix, distributions)
3. **Literature research**: the survey from §5 and the mapping table; ≥6 references, consistent citation style
4. **Baseline**: reproduction results table + your answers to the 4 questions in §4
5. **Improvement experiments**: for each improvement = motivation (research basis) → method → results table → 1 figure → brief discussion
6. **Overall comparison and discussion**: overall comparison table; the "best configuration" and why you chose it; limitations and overfitting analysis; if you discover a data pattern (see the hint in §9), you *must* discuss it here
7. **Conclusion and future work**
8. **References**

---

## 8. Grading Rubric (out of 100)

| Points | What is being assessed | Value |
|---|---|---|
| Baseline | Correct reproduction; accurate answers to the 4 questions (especially PCA applicability, imbalanced evaluation, data leakage) | 15 |
| Data exploration | Structured statistical analysis; clean figures; able to read "orthogonality" and "imbalance" from the data | 10 |
| Literature research | Covers ≥3 directions; connects to original literature; research maps 1-to-1 to the chosen experiments | 20 |
| Improvement experiments | ≥3 groups; rigorous control variables; consistent evaluation protocol; no leakage; credible results | 25 |
| Presentation | Well-formed tables/figures (axis labels, titles, units); consistent metric definitions | 15 |
| Report writing | Coherent logic; conclusions supported by evidence; proper citations; limitations and future work discussed | 15 |

---

## 9. Important Notes (read three times)

1. **This dataset appears to be programmatically generated.** If you observe that tree models reach F1 close to 100% with ease, that a simple "humidity + cloud cover" rule exists, or that feature importances are highly concentrated — discuss explicitly in your report: what does this imply, and what can / cannot be concluded from this experiment? (This is a **bonus opportunity**, not a penalty.)
2. **Data leakage** is the most common cause of lost points: any operation that has "seen" the labels must not be part of the pre-training pipeline.
3. Self-check before submission: is the random seed fixed? Is the 5-fold split identical to the baseline? Does every figure have axis labels?
4. Do the research and writing independently; cite properly. Expect similarity-check warnings as per the university's plagiarism policy.

---

## 10. Starting References (not exhaustive)

- Chawla N V, et al. SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 2002.
- He H, Garcia E A. Learning from Imbalanced Data. IEEE TKDE, 2009.
- Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLoS ONE, 2015.
- Jolliffe I T. Principal Component Analysis. Springer, 2002.
- Breiman L. Random Forests. Machine Learning, 2001.
- Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.
- scikit-learn docs: `decomposition.PCA` / `model_selection` / `metrics` / user guide
- imbalanced-learn docs (SMOTE, etc.)