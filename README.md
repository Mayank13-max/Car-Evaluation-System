# Car Evaluation System 🚗

> **CodTech IT Solutions — ML Internship Project**  
> **Intern:** Mayank | **ID:** CTTS148 | **Duration:** 4 Weeks  
> **Degree:** B.Tech AIML (5th Semester)

---

## 📌 Project Overview

This project builds a complete **Machine Learning pipeline** to classify cars into four acceptability categories:

| Class | Meaning |
|-------|---------|
| `unacc` | Unacceptable |
| `acc` | Acceptable |
| `good` | Good |
| `vgood` | Very Good |

It uses the **UCI Car Evaluation Dataset** (1,728 instances, 6 categorical features) and compares three ML classifiers: Logistic Regression, Decision Tree, and Random Forest (+ optional XGBoost).

---

## 📁 Folder Structure

```
car-evaluation-system/
│
├── data/
│   └── car.csv                    # Raw dataset (UCI Car Evaluation)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Helper functions (save/load, plot style)
│   ├── data_loader.py             # Load & inspect dataset
│   ├── preprocessing.py           # Clean, encode, split data
│   ├── eda.py                     # 10 exploratory visualisations
│   ├── model_training.py          # Train all ML models
│   └── model_evaluation.py        # Evaluate, compare, feature importance
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl                # (if XGBoost installed)
│   ├── ordinal_encoder.pkl
│   └── label_encoder.pkl
│
├── visualizations/
│   └── *.png                      # 10 saved charts
│
├── notebooks/
│   └── car_evaluation.ipynb       # Full Jupyter Notebook
│
├── reports/
│   └── project_report.md          # 8–10 page project report
│
├── app.py                         # Streamlit web application
├── main.py                        # Full ML pipeline entry point
├── requirements.txt
└── README.md
```

---

## 🗂 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
- **Instances:** 1,728
- **Features:** 6 (all categorical/ordinal)
- **Target:** `class` — 4 categories
- **Missing Values:** None

| Feature | Values |
|---------|--------|
| buying | vhigh, high, med, low |
| maint | vhigh, high, med, low |
| doors | 2, 3, 4, 5more |
| persons | 2, 4, more |
| lug_boot | small, med, big |
| safety | low, med, high |

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project

```bash
cd "d:\My projects\car evaluation system"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Step 1 — Train models & generate visualisations

```bash
python main.py
```

This will:
- Load and inspect the dataset
- Perform EDA (saves 10 plots to `visualizations/`)
- Train 3–4 ML models (saves `.pkl` files to `models/`)
- Print evaluation metrics & confusion matrices

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Step 3 — Open the Jupyter Notebook

```bash
jupyter notebook notebooks/car_evaluation.ipynb
```

---

## 🤖 Models Used

| Model | Type | Notes |
|-------|------|-------|
| **Logistic Regression** | Linear | Baseline model, `max_iter=1000` |
| **Decision Tree** | Tree-based | `max_depth=6`, interpretable |
| **Random Forest** | Ensemble | `n_estimators=100`, best performer |
| **XGBoost** *(optional)* | Gradient Boost | Install: `pip install xgboost` |

---

## 📊 Visualisations (10 total)

| # | Filename | Description |
|---|----------|-------------|
| 1 | `01_class_distribution.png` | Bar chart of target classes |
| 2 | `02_class_distribution_pie.png` | Pie chart of class proportions |
| 3 | `03_feature_distributions.png` | 6-subplot feature value counts |
| 4 | `04_correlation_heatmap.png` | Pearson correlation heatmap |
| 5 | `05_buying_vs_class.png` | Buying price vs class |
| 6 | `06_safety_vs_class.png` | Safety rating vs class |
| 7 | `07_luggage_vs_class.png` | Luggage boot vs class |
| 8 | `08_persons_vs_class.png` | Person capacity vs class (stacked) |
| 9 | `09_maint_vs_class.png` | Maintenance cost vs class |
| 10 | `10_feature_class_heatmap.png` | Mean feature value per class |
| + | `confusion_matrix_*.png` | Per-model confusion matrix |
| + | `feature_importance_*.png` | Feature importance (RF, DT) |
| + | `model_comparison.png` | All metrics compared side-by-side |

---

## 📈 Expected Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | ~85% | ~0.84 |
| Decision Tree | ~97% | ~0.97 |
| **Random Forest** | **~97%+** | **~0.97+** |
| XGBoost | ~97%+ | ~0.97+ |

> ✅ Random Forest is the recommended model for deployment.

---

## 📝 Project Report

See [`reports/project_report.md`](reports/project_report.md) for the full 8–10 page report covering:
- Introduction & Objectives
- Dataset Description
- Methodology
- EDA Findings
- Model Results & Comparison
- Conclusion & Future Work
- References

---

## 🧰 Technology Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scikit-learn | 1.4.2 | ML models & preprocessing |
| matplotlib | 3.8.3 | Plotting |
| seaborn | 0.13.2 | Statistical visualisations |
| streamlit | 1.32.2 | Web application |
| joblib | 1.3.2 | Model serialisation |
| xgboost | 2.0.3 | Optional gradient boosting |

---

## 📌 Key ML Concepts Demonstrated

- ✅ Ordinal Encoding (with explicit category order)
- ✅ Stratified Train/Test Split
- ✅ Multi-class Classification
- ✅ Weighted Precision / Recall / F1
- ✅ Confusion Matrix
- ✅ Feature Importance
- ✅ Model Comparison

---

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/)
- Internship: CodTech IT Solutions
- Guidance: CodTech ML Team

---

*Made with ❤️ by Mayank | CodTech Intern CTTS148*
