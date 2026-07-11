# =============================================================================
# model_evaluation.py — Evaluate, Compare & Visualise Model Performance
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Purpose : Compute accuracy, precision, recall, F1, confusion matrix,
#           classification report, feature importance plots, and model
#           comparison charts.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.utils import (
    print_section_header,
    save_figure,
    set_plot_style,
    CLASS_DISPLAY_NAMES,
)

# Colour palette used across evaluation plots
PALETTE = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585"]
CLASS_ORDER = ["unacc", "acc", "good", "vgood"]


# ── Single Model Evaluation ───────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str,
                   label_classes=None) -> dict:
    """
    Evaluate a trained model on the test set and print a full report.

    Metrics computed:
        - Accuracy
        - Precision (weighted)
        - Recall    (weighted)
        - F1 Score  (weighted)
        - Classification Report (per-class breakdown)

    Parameters:
        model       : Trained sklearn / XGBoost model.
        X_test      : Test feature matrix.
        y_test      : True test labels.
        model_name  : Name string for display (e.g., 'Random Forest').
        label_classes: Array of class names in label order (optional).

    Returns:
        dict: {'accuracy', 'precision', 'recall', 'f1'}
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  ── {model_name} ──")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1 Score  : {f1:.4f}")

    # Per-class classification report
    target_names = label_classes if label_classes is not None else CLASS_ORDER
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=target_names,
                                zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── Confusion Matrix Visualisation ───────────────────────────────────────────

def plot_confusion_matrix(model, X_test, y_test, model_name: str,
                          label_classes=None, save: bool = True):
    """
    Plot and optionally save a styled confusion matrix heatmap.

    Parameters:
        model       : Trained model.
        X_test      : Test features.
        y_test      : True labels.
        model_name  : Model name (used in plot title & filename).
        label_classes: Class name list.
        save        : Whether to save the figure.
    """
    set_plot_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    labels = label_classes if label_classes is not None else CLASS_ORDER

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save:
        fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        save_figure(fig, fname)

    plt.show()
    plt.close(fig)


# ── Model Comparison Chart ────────────────────────────────────────────────────

def compare_models(results: dict, save: bool = True):
    """
    Plot a grouped bar chart comparing Accuracy, Precision, Recall, and F1
    across all trained models.

    Parameters:
        results (dict): {model_name: {metric: value, ...}, ...}
        save    (bool): Whether to save the figure.
    """
    set_plot_style()

    metrics     = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics)

    x = np.arange(n_metrics)
    bar_width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = ["#4361EE", "#3A0CA3", "#7209B7", "#F72585", "#4CC9F0"]

    for i, (model_name, scores) in enumerate(results.items()):
        values = [scores[m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width,
                      label=model_name, color=colors[i % len(colors)],
                      edgecolor="white", linewidth=0.8, alpha=0.9)

        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save:
        save_figure(fig, "model_comparison.png")

    plt.show()
    plt.close(fig)


# ── Feature Importance ────────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str,
                            save: bool = True):
    """
    Plot a horizontal bar chart showing feature importances.

    Applicable to: Decision Tree, Random Forest (and XGBoost).
    Not applicable to Logistic Regression (uses different coefficients).

    Parameters:
        model        : Trained tree-based model with feature_importances_ attr.
        feature_names: List of feature column names.
        model_name   : Name used in title & filename.
        save         : Whether to save the figure.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"  [!] {model_name} does not support feature_importances_. Skipping.")
        return

    set_plot_style()
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]   # sort descending
    sorted_features = [feature_names[i] for i in indices]
    sorted_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.85, len(sorted_features)))
    bars = ax.barh(sorted_features[::-1], sorted_values[::-1],
                   color=colors, edgecolor="white")

    # Add value labels
    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Importance Score (Gini)", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name}", fontsize=14,
                 fontweight="bold", pad=15)
    ax.set_xlim(0, max(sorted_values) * 1.15)
    plt.tight_layout()

    if save:
        fname = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        save_figure(fig, fname)

    plt.show()
    plt.close(fig)


# ── Full Evaluation Pipeline ──────────────────────────────────────────────────

def evaluate_all_models(models: dict, X_test, y_test,
                        feature_names: list, label_classes=None) -> dict:
    """
    Run full evaluation for every model: metrics, confusion matrix,
    feature importance (if applicable), and comparison chart.

    Parameters:
        models       : {model_name: model_object}
        X_test       : Test feature matrix.
        y_test       : True test labels.
        feature_names: List of feature column names.
        label_classes: Class name list.

    Returns:
        dict: Results {model_name: {metric: value, ...}}
    """
    print_section_header("STEP 7 — Model Evaluation")

    results = {}

    for name, model in models.items():
        scores = evaluate_model(model, X_test, y_test, name, label_classes)
        results[name] = scores
        plot_confusion_matrix(model, X_test, y_test, name, label_classes)

    # Feature importance for tree-based models
    print_section_header("STEP 8 — Feature Importance")
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, feature_names, name)

    # Comparison chart
    print_section_header("STEP 9 — Model Comparison")
    compare_models(results)

    # Print summary table
    print("\n  ── Summary Table ──")
    header = f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    print(header)
    print("  " + "-" * 64)
    for name, scores in results.items():
        print(f"  {name:<22} {scores['accuracy']:>10.4f} "
              f"{scores['precision']:>10.4f} {scores['recall']:>10.4f} "
              f"{scores['f1']:>10.4f}")

    # Identify best model
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n  🏆 Best Model : {best_name}  (F1 = {results[best_name]['f1']:.4f})")

    return results
