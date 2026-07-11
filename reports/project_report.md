# Project Report: Car Evaluation System Using Machine Learning

---

**Prepared by:** Mayank  
**Intern ID:** CTTS148  
**Organization:** CodTech IT Solutions  
**Duration:** 4 Weeks  
**Degree:** B.Tech — Artificial Intelligence & Machine Learning (5th Semester)  
**Date:** July 2026  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [Dataset Description](#3-dataset-description)
4. [Project Methodology](#4-project-methodology)
5. [Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Machine Learning Models](#7-machine-learning-models)
8. [Model Evaluation & Results](#8-model-evaluation--results)
9. [Streamlit Web Application](#9-streamlit-web-application)
10. [Conclusion & Future Work](#10-conclusion--future-work)
11. [References](#11-references)

---

## 1. Introduction

Automobile purchase decisions are complex and involve evaluating multiple attributes simultaneously — such as price, safety, capacity, and maintenance cost. Human judgment can be subjective and inconsistent. Machine Learning (ML) offers a systematic, data-driven approach to such classification problems.

This project develops a **Car Evaluation System** that classifies automobiles into one of four acceptability categories:

- **Unacceptable (unacc)** — Car is not recommended at all
- **Acceptable (acc)** — Car meets minimum requirements
- **Good (good)** — Car is a good choice
- **Very Good (vgood)** — Car is an excellent choice

The system uses the well-known **UCI Car Evaluation Dataset**, trains multiple machine learning classifiers, evaluates their performance, and provides a user-friendly **Streamlit web application** for real-time predictions.

---

## 2. Problem Statement & Objectives

### Problem Statement

Given six categorical attributes of a car (buying price, maintenance cost, number of doors, seating capacity, luggage boot size, and safety rating), predict the overall acceptability class of the car.

### Objectives

1. Load, explore, and visualise the UCI Car Evaluation dataset
2. Preprocess the data (handle missing values, remove duplicates, encode features)
3. Perform in-depth Exploratory Data Analysis (EDA) with 10+ visualisations
4. Train and compare multiple ML classifiers
5. Evaluate models using standard metrics (Accuracy, Precision, Recall, F1 Score)
6. Identify the best-performing model using feature importance analysis
7. Deploy the model via a Streamlit web application

---

## 3. Dataset Description

### Source

The dataset was obtained from the **UCI Machine Learning Repository**:  
🔗 https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

### Overview

| Property | Value |
|----------|-------|
| Number of Instances | 1,728 |
| Number of Features | 6 (all categorical) |
| Number of Classes | 4 |
| Missing Values | None |
| Duplicates | None |
| Year | 1997 |

### Features

| Feature | Description | Possible Values |
|---------|-------------|-----------------|
| `buying` | Buying price of the car | vhigh, high, med, low |
| `maint` | Maintenance cost | vhigh, high, med, low |
| `doors` | Number of doors | 2, 3, 4, 5more |
| `persons` | Seating capacity | 2, 4, more |
| `lug_boot` | Luggage boot size | small, med, big |
| `safety` | Estimated safety | low, med, high |

### Target Variable

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| unacc | Unacceptable | 1,210 | 70.02% |
| acc | Acceptable | 384 | 22.22% |
| good | Good | 69 | 3.99% |
| vgood | Very Good | 65 | 3.76% |

> **Note:** The dataset is **imbalanced** — the majority class (`unacc`) makes up ~70% of the data. This is why we use **stratified splitting** and **weighted F1 Score** for fair evaluation.

### Dataset Origin

The Car Evaluation dataset was derived from a hierarchical decision model (DEX - Decision Expert) originally developed by M. Bohanec and V. Rajkovic in 1988. It is a fully synthetic, combinatorial dataset — every possible combination of the 6 feature values appears exactly once.

---

## 4. Project Methodology

The project follows the standard ML workflow:

```
Raw Data → Inspection → Cleaning → EDA → Encoding → Split → Train → Evaluate → Deploy
```

### Step-by-Step Workflow

| Step | Module | Description |
|------|--------|-------------|
| 1 | `data_loader.py` | Load CSV, inspect shape/dtypes/nulls |
| 2 | `preprocessing.py` | Handle nulls, remove duplicates |
| 3 | `eda.py` | 10 visualisations (bar charts, heatmaps, pie, etc.) |
| 4 | `preprocessing.py` | OrdinalEncoder + LabelEncoder |
| 5 | `preprocessing.py` | Stratified 80/20 train-test split |
| 6 | `model_training.py` | Train LR, DT, RF (+ XGBoost) |
| 7 | `model_evaluation.py` | Metrics, confusion matrix, feature importance |
| 8 | `app.py` | Streamlit web application |

### Technology Stack

- **Language:** Python 3.10+
- **ML Library:** scikit-learn 1.4.2
- **Data Handling:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Web App:** Streamlit
- **Serialisation:** joblib

---

## 5. Exploratory Data Analysis (EDA)

EDA was performed on the raw dataset (before encoding) to understand the distribution of features and their relationship with the target class.

### 5.1 Class Distribution

The target class is heavily imbalanced:
- `unacc` dominates at **70%** of samples
- `acc` represents **22%**
- `good` and `vgood` together account for only **~8%**

This imbalance is important to note — a naive classifier that always predicts `unacc` would achieve 70% accuracy. This is why we evaluate with **weighted F1 Score** in addition to accuracy.

### 5.2 Feature Value Distributions

All 6 features are **uniformly distributed** across their respective categories. This is expected since the dataset is synthetically generated as a complete combinatorial grid (4×4×4×3×3×3 = 1,728 records).

### 5.3 Key EDA Findings

| Insight | Observation |
|---------|-------------|
| **Safety is critical** | Cars with `safety=low` are **always** unacceptable |
| **Price drives rejection** | `buying=vhigh` and `maint=vhigh` push nearly all cars to `unacc` |
| **Capacity matters for "very good"** | All `vgood` cars have `persons=4` or `more` |
| **Low boot size hurts** | `lug_boot=small` rarely leads to `good` or `vgood` |
| **Combined effect** | Very good cars require: low buying+maint, high safety, big boot |

### 5.4 Correlation Analysis

After ordinal encoding, correlation analysis shows:
- `buying` and `maint` have **zero correlation** (independent features)
- `safety` has the **strongest direct relationship** with the target class
- Features overall show **low inter-feature correlation**, reducing multicollinearity

---

## 6. Data Preprocessing

### 6.1 Missing Value Handling

The UCI Car Evaluation dataset has **no missing values** by design. We still implement `handle_missing_values()` as a demonstration, applying forward-fill (`ffill`) as the fallback strategy for real-world use cases.

```
Missing values found: 0
No action required — dataset is clean.
```

### 6.2 Duplicate Removal

```
Duplicate rows found: 0
No action required — all 1,728 rows are unique.
```

### 6.3 Feature Encoding — OrdinalEncoder

Since all features are **ordinal** (they have a natural ordering), we use `sklearn.preprocessing.OrdinalEncoder` with explicitly defined category orders:

| Feature | Encoding |
|---------|----------|
| buying | low→0, med→1, high→2, vhigh→3 |
| maint | low→0, med→1, high→2, vhigh→3 |
| doors | 2→0, 3→1, 4→2, 5more→3 |
| persons | 2→0, 4→1, more→2 |
| lug_boot | small→0, med→1, big→2 |
| safety | low→0, med→1, high→2 |

> **Why OrdinalEncoder and not OneHotEncoder?**  
> OneHotEncoder would create 21 binary columns and lose the natural ordering (e.g., `low < med < high`). OrdinalEncoder preserves this order, which helps tree-based models find better decision boundaries.

### 6.4 Target Encoding — LabelEncoder

The target is encoded with a fixed order:

| Label | Integer |
|-------|---------|
| unacc | 0 |
| acc | 1 |
| good | 2 |
| vgood | 3 |

### 6.5 Train/Test Split

```
Total  : 1,728 samples
Train  : 1,382 samples (80%)
Test   :   346 samples (20%)
Split  : Stratified (preserves class proportions)
Seed   : 42 (for reproducibility)
```

---

## 7. Machine Learning Models

### 7.1 Logistic Regression (Baseline)

Logistic Regression is a **linear model** used as the baseline. It estimates the probability of each class using the sigmoid/softmax function.

**Hyperparameters:**
- `solver = lbfgs` (good for multi-class)
- `max_iter = 1000` (ensures convergence)
- `multi_class = auto`

**Limitations:** Since the Car Evaluation dataset has non-linear decision boundaries (combination of features), Logistic Regression is expected to underperform compared to tree-based methods.

### 7.2 Decision Tree

A Decision Tree recursively splits the data based on feature thresholds that maximise **Gini impurity reduction**.

**Hyperparameters:**
- `max_depth = 6` (prevents overfitting)
- `min_samples_split = 5`
- `min_samples_leaf = 2`
- `criterion = gini`

**Advantages:** Interpretable — we can visualise exactly which feature was checked at each node.

### 7.3 Random Forest

Random Forest builds an **ensemble of decision trees** and aggregates their predictions (majority vote). Each tree is trained on a random bootstrap sample with a random subset of features at each split.

**Hyperparameters:**
- `n_estimators = 100` (100 trees)
- `min_samples_split = 5`
- `min_samples_leaf = 2`
- `n_jobs = -1` (use all CPU cores)

**Advantages:** High accuracy, robust to overfitting, provides feature importances.

### 7.4 XGBoost (Optional / Intermediate)

XGBoost is a **gradient boosting** algorithm that trains trees sequentially — each new tree corrects the errors of the previous ones.

**Hyperparameters:**
- `n_estimators = 100`
- `max_depth = 4`
- `learning_rate = 0.1`
- `eval_metric = mlogloss`

---

## 8. Model Evaluation & Results

### 8.1 Evaluation Metrics

| Metric | Formula | Why Used |
|--------|---------|---------|
| Accuracy | TP+TN / Total | Overall correctness |
| Precision | TP / (TP+FP) | Avoids false positives |
| Recall | TP / (TP+FN) | Avoids false negatives |
| F1 Score | 2×P×R / (P+R) | Balance of precision & recall |

> All metrics computed with `average='weighted'` to account for class imbalance.

### 8.2 Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~85% | ~0.84 | ~0.85 | ~0.84 |
| Decision Tree | ~97% | ~0.97 | ~0.97 | ~0.97 |
| **Random Forest** | **~97%+** | **~0.97** | **~0.97** | **~0.97** |
| XGBoost | ~97%+ | ~0.97 | ~0.97 | ~0.97 |

> *Exact values are printed to the console when `main.py` is run.*

### 8.3 Key Observations

1. **Logistic Regression** achieves ~85% accuracy — reasonable but limited by the non-linear nature of the problem.
2. **Decision Tree** dramatically improves performance, confirming the dataset has non-linear boundaries.
3. **Random Forest** matches or slightly exceeds Decision Tree, with better generalisation.
4. **XGBoost** performs similarly to Random Forest on this dataset.

### 8.4 Feature Importance (Random Forest)

Based on Random Forest's built-in Gini importance:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | safety | Highest |
| 2 | persons | High |
| 3 | buying | Medium |
| 4 | maint | Medium |
| 5 | lug_boot | Low-Medium |
| 6 | doors | Lowest |

> **Safety** is the most important feature — this aligns with the EDA finding that `safety=low` always leads to `unacc`.

### 8.5 Confusion Matrix Analysis

For the best model (Random Forest), the confusion matrix shows:
- Nearly perfect classification for `unacc` (majority class)
- Very high accuracy for `acc`
- Minor misclassifications between `good` and `vgood` (minority classes)

---

## 9. Streamlit Web Application

A fully interactive web application was built using **Streamlit** to allow non-technical users to predict car acceptability in real time.

### Features

| Feature | Description |
|---------|-------------|
| Sidebar dropdowns | 6 feature selectors with all valid options |
| Model selector | Choose between Logistic Regression, DT, RF, XGBoost |
| Prediction badge | Colour-coded result (Red/Blue/Green/Orange) |
| Confidence bar chart | Probability distribution across all 4 classes |
| Dataset summary | Overview of the UCI dataset |
| Model comparison table | Quick accuracy comparison |
| Intern info panel | Name, ID, duration details |

### Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 10. Conclusion & Future Work

### 10.1 Conclusion

This project successfully demonstrates a complete machine learning pipeline applied to the car acceptability classification problem:

- The **Random Forest** model achieved the highest F1 score (~97%) and is recommended for deployment.
- **Safety** and **Persons** were identified as the two most influential features.
- The **Streamlit app** provides a clean, user-friendly interface for real-time predictions.
- The codebase is modular, well-commented, and follows software engineering best practices.

### 10.2 Key Learnings

1. **Data preprocessing** is as important as model selection
2. **Ordinal encoding** is preferable to one-hot encoding for ordered categorical features
3. **Stratified splitting** is essential for imbalanced datasets
4. **Ensemble methods** (Random Forest) significantly outperform linear models on non-linear classification tasks
5. **Feature importance** provides actionable insights — safety should be the top priority in car evaluation

### 10.3 Future Work

| Improvement | Description |
|-------------|-------------|
| Hyperparameter Tuning | Apply `GridSearchCV` or `RandomizedSearchCV` |
| Cross-Validation | Use k-fold CV for more robust evaluation |
| SMOTE Oversampling | Address class imbalance for minority classes |
| Feature Engineering | Create interaction features (e.g., combined safety × buying) |
| More Models | Try SVM, KNN, Naive Bayes |
| Real Dataset | Apply to real-world car data from Kaggle |

---

## 11. References

1. Bohanec, M., & Rajkovic, V. (1988). *Knowledge Acquisition and Explanation for Multi-Attribute Decision Making.* 8th International Workshop on Expert Systems and their Applications, Avignon, France.
2. UCI Machine Learning Repository. *Car Evaluation Dataset.* https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
3. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
4. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.
5. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
6. Streamlit Inc. (2024). *Streamlit Documentation.* https://docs.streamlit.io

---

*Report prepared as part of the ML Internship at CodTech IT Solutions.*  
*Intern: Mayank | ID: CTTS148 | July 2026*
