# ELEC70143 – Machine Learning Assignment 2025/2026

This is a two-week assignment (~20 hours expected effort). Focus on **reasoning**, interpretation, and clear figures rather than extensive code development. Many of the questions are open-ended: there will be multiple valid approaches to them. Marks will be allocated based on the individual merits of the approach you adopt, rather than the correct choice of any specific strategy.

**Submission:** a PDF file with clear section headings matching the question numbers. Figures and tables must be clearly explained and interpretable. You may use a Jupyter notebook and convert it to PDF at the end.
**Deadline:** **December 2nd at Midday**.

This is an individual project: you must produce your own work, in line with the library services and plagiarism induction received at the start of term.

---

# Part 1: Air Pollution Monitoring System

Air pollution is a leading environmental cause of premature mortality, responsible for around **7 million deaths per year** worldwide (WHO). Accurate measurement is essential, but high-quality sensors are expensive, while low-cost sensors produce noisier signals.

You are working with a start-up designing a compact, low-cost air-quality monitor using a subset of electrochemical and meteorological sensors to estimate **true CO concentration**.

Your goals:

1. Build and compare regression models predicting CO concentration.
2. Quantify the trade-off between model accuracy and hardware cost.
3. Recommend the most cost-effective sensor configuration.

Dataset: from **De Vito et al. (2008)** — urban pollution monitoring (Italy, 2004–2005).
Target: **CO(GT)** in mg/m³.
Data: **q1.csv**

## Variable Summary

| Variable      | Description                     | Range    |
| ------------- | ------------------------------- | -------- |
| PT08.S1(CO)   | Tin oxide sensor response (ppm) | 300–1800 |
| PT08.S2(NMHC) | Non-methane hydrocarbon sensor  | 400–2000 |
| PT08.S3(NOx)  | Nitrogen oxide sensor           | 400–2000 |
| PT08.S4(NO2)  | Nitrogen dioxide sensor         | 400–2000 |
| PT08.S5(O3)   | Ozone sensor                    | 400–2000 |
| T             | Temperature (°C)                | –10–40   |
| RH            | Relative humidity (%)           | 0–100    |
| AH            | Absolute humidity               | 0–50     |
| NOx(GT)       | Reference NOx concentration     | 0–600    |
| NO2(GT)       | Reference NO2 concentration     | 0–250    |
| C6H6(GT)      | Benzene conc. (µg/m³)           | 0–15     |
| CO(GT)        | **Target variable**             | 0–10     |

---

## A. Missing Data

After importing the dataset, report percent missing per feature.
Choose and implement a justified imputation strategy (mean/median, KNN, model-based, etc.)

---

## B. Correlation Structure

Using a correlation plot or pair plot, explore distributions and relationships.
Comment on strong correlations and multicollinearity.

---

## C. OLS Model

Split: **80% train / 20% test**.
Fit OLS with all features.

Report:

* Model summary
* Significant variables
* Interpretation
* Why some coefficients are extremely large or have huge p-values (hint: multicollinearity).

Use `statsmodels` for richer diagnostics.

---

## D. Ridge & Lasso Regression

With cross-validated regularisation:

* Plot coefficient paths vs λ (log-scale)
* Use **scaled features**
* Discuss regularisation’s effect on multicollinearity & interpretability

---

## E. Kernel Ridge Regression (Polynomial Degree 2)

Does a quadratic kernel improve performance? Explain.

---

## F. Kernel Ridge Regression (RBF)

Tune λ and kernel width (grid search with CV).
Reduce grid density if computation is too slow.

---

## G. Compare All Models

Models:

* OLS
* Ridge
* Lasso
* Kernel Ridge (poly degree 2)
* Kernel Ridge (RBF)

Report **6 metrics**:

* **In-sample:** R², MAE, RMSE
* **Out-of-sample:** R², MAE, RMSE

Compare performance and interpretability trade-offs.

---

## H. Sensor Cost Constraint

| Feature       | Cost (£) |
| ------------- | -------- |
| PT08.S1(CO)   | 1000     |
| PT08.S2(NMHC) | 800      |
| PT08.S3(NOx)  | 700      |
| PT08.S4(NO2)  | 700      |
| PT08.S5(O3)   | 900      |
| T             | 150      |
| RH            | 150      |
| AH            | 200      |
| NOx(GT)       | 1800     |
| NO2(GT)       | 1800     |
| C6H6(GT)      | 2500     |

Total cost for the monitor must not exceed **£4000**.

Using the **lasso path**, construct models with increasing total sensor cost. For each:

* Selected features
* Total cost
* Cross-validated RMSE

Plot **Cost vs RMSE** to get the cost–accuracy frontier.

Identify:

* A low-cost monitor (≤ £2500)
* A high-performance monitor (≤ £4000)

Evaluate their test-set RMSEs.

---

## I. Optimal Alert Threshold

Legal CO threshold: **5 mg/m³**

Costs:

* False positive: **£2,000**
* False negative: **£10,000**

Deterministic rule: issue alert if prediction **ŷ > t**.

For a range of thresholds *t*, compute **expected cost** via CV or validation.
Choose the threshold minimising expected cost.

---

## J. Compare Designs

Compare expected costs for:

* Low-cost monitor
* High-performance monitor

Discuss whether added sensors are justified by cost reduction.

---

# Part 2: Detecting Cyber Attacks from Network Flows

NCSC must detect malicious traffic in national infrastructure. False negatives are extremely costly; false positives waste analyst time.

Dataset: **q2.csv**, containing benign traffic and several DDoS attack types.
Each record contains flow-level features and a binary target (BENIGN vs MALICIOUS).

Reference: *Heard et al. (2018), Data Science for Cyber-Security.*

---

## Variables

| Name                        | Type                   |
| --------------------------- | ---------------------- |
| Flow Duration               | Continuous             |
| Total Fwd Packets           | Continuous             |
| Total Backward Packets      | Continuous             |
| Total length of Fwd packets | Continuous             |
| Total length of Bwd packets | Continuous             |
| Flow IAT Mean               | Continuous             |
| Flow IAT Std                | Continuous             |
| Packet Length Mean          | Continuous             |
| Packet Length Std           | Continuous             |
| Average Packet Size         | Continuous             |
| Active Mean                 | Continuous             |
| Idle Mean                   | Continuous             |
| Protocol                    | Categorical (6, 17, 0) |
| Target                      | 1/0                    |

**Hint:** One-hot encode `Protocol`.

Split: **80/20**, stratified.
Tune hyperparameters with K-fold CV.

---

## A. Exploratory Analysis (10 pts)

* Class balance
* Plot distributions of **Flow Duration** and **Total Fwd Packets**
* Report 2 insights on class separation

---

## B. Baseline Logistic Regression (10 pts)

* Fit logistic regression
* Confusion matrix
* Accuracy, ROC curve, ROC-AUC
* Interpret results

---

## C. Penalised Logistic Regression (L1) (15 pts)

* Fit L1-regularised logistic regression
* Report accuracy, ROC, ROC-AUC
* Perform **20 bootstrap resamples**
* For each resample, record which features are selected
* Plot proportion selected
* Identify top-5 influential features (largest |coeff|) and interpret

---

## D. Linear SVM (15 pts)

Fit:

* Hard margin
* Soft margin (tune C ∈ {0.1, 1, 10, 100})

For each:

* Accuracy
* ROC-AUC
* Number of support vectors
* 2D decision boundary using any two continuous features
* Explain why hard-margin SVM is fragile here

---

## E. Kernel SVM (RBF) (20 pts)

Tune:

* C ∈ {0.1, 1, 10, 100}
* γ ∈ {0.001, 0.1, 1, 10}

Compare test performance to logistic regression.

---

## F. Cost-Sensitive Evaluation (15 pts)

False negative cost: **£1,000,000**
False positive cost: **£1,000**

For each model:

1. Compute expected cost using the default decision threshold.
2. Adjust threshold to minimise cost.
3. Recompute expected cost.
4. Plot:

   * ROC
   * Cost curves

Expected cost formula:

```
Cost(t) = C_FN × (1 − TPR(t)) × N_pos  +  C_FP × FPR(t) × N_neg
```

Choose *t* minimising the cost.

---

## G. Reflection (Policy Briefing) (15 pts)

Max **500 words** covering:

1. Comparison of Logistic, Penalised Logistic, Linear SVM, Kernel SVM

   * Interpretability
   * Accuracy
   * Stability under resampling
   * Robustness to changing costs
2. Deployment strategy balancing explainability and performance
3. Risks: shift in attack types; model vulnerability; monitoring and adaptation strategies
