# =============================================================================
# model_training.py — Train & Save ML Models
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Purpose : Train Logistic Regression, Decision Tree, Random Forest
#           (and optional XGBoost), save each model to disk.
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.utils import print_section_header, save_model

# Try importing XGBoost; skip if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  [!] XGBoost not installed. Skipping XGBoost model.")


# ── Individual Model Training Functions ───────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    Logistic Regression is a simple linear model used as a baseline.
    We set max_iter=1000 so the solver converges on this dataset.

    Parameters:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        LogisticRegression: Fitted model.
    """
    print("\n  [1] Training Logistic Regression ...")
    model = LogisticRegression(
        max_iter=1000,           # enough iterations to converge
        random_state=42,
        multi_class="auto",      # handles multi-class automatically
        solver="lbfgs"           # good general-purpose solver
    )
    model.fit(X_train, y_train)
    print("      Done!")
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.

    Decision Trees are interpretable models — we can visualise which
    features drive the splits. max_depth=6 prevents overfitting.

    Parameters:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        DecisionTreeClassifier: Fitted model.
    """
    print("\n  [2] Training Decision Tree ...")
    model = DecisionTreeClassifier(
        max_depth=6,             # limit tree depth to avoid overfitting
        min_samples_split=5,     # a node must have ≥5 samples to be split
        min_samples_leaf=2,      # each leaf must have ≥2 samples
        criterion="gini",        # Gini impurity is the standard criterion
        random_state=42
    )
    model.fit(X_train, y_train)
    print("      Done!")
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Random Forest builds many decision trees and aggregates their votes.
    It is generally the best-performing model on tabular datasets.

    Parameters:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        RandomForestClassifier: Fitted model.
    """
    print("\n  [3] Training Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=100,        # 100 trees in the forest
        max_depth=None,          # grow trees until pure leaves
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1                # use all CPU cores for speed
    )
    model.fit(X_train, y_train)
    print("      Done!")
    return model


def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier (optional, intermediate model).

    XGBoost is a gradient-boosted tree method that often achieves
    state-of-the-art accuracy on tabular datasets.

    Parameters:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        XGBClassifier: Fitted model, or None if XGBoost is not installed.
    """
    if not XGBOOST_AVAILABLE:
        return None

    print("\n  [4] Training XGBoost ...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,       # step size for each boosting round
        use_label_encoder=False,
        eval_metric="mlogloss",  # multi-class log-loss
        random_state=42,
        verbosity=0              # suppress verbose output
    )
    model.fit(X_train, y_train)
    print("      Done!")
    return model


# ── Full Training Pipeline ────────────────────────────────────────────────────

def train_all_models(X_train, y_train) -> dict:
    """
    Train all models and return them in a dictionary.
    Also saves each trained model to the 'models/' directory.

    Parameters:
        X_train : Training feature matrix.
        y_train : Training labels.

    Returns:
        dict: {'Logistic Regression': model, 'Decision Tree': model,
               'Random Forest': model, [optionally 'XGBoost': model]}
    """
    print_section_header("STEP 6 — Model Training")

    models = {}

    # Train each model
    lr  = train_logistic_regression(X_train, y_train)
    dt  = train_decision_tree(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    models["Logistic Regression"] = lr
    models["Decision Tree"]       = dt
    models["Random Forest"]       = rf
    if xgb is not None:
        models["XGBoost"] = xgb

    # Save models to disk
    print("\n  Saving models to disk ...")
    save_model(lr,  "logistic_regression.pkl")
    save_model(dt,  "decision_tree.pkl")
    save_model(rf,  "random_forest.pkl")
    if xgb is not None:
        save_model(xgb, "xgboost.pkl")

    print(f"\n  Total models trained: {len(models)}")
    for name in models:
        print(f"    ✓ {name}")

    return models
