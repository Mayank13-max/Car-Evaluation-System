# =============================================================================
# utils.py — Helper/Utility Functions
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Purpose : Reusable utility functions used across the project
# =============================================================================

import os
import joblib
import matplotlib.pyplot as plt


# ── Directory Helpers ─────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters:
        path (str): Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_model(model, filename: str, models_dir: str = "models") -> str:
    """
    Save a trained sklearn model to disk using joblib.

    Parameters:
        model      : Trained sklearn model object.
        filename   : Name for the saved file (e.g., 'random_forest.pkl').
        models_dir : Directory where the model will be saved.

    Returns:
        str: Full path to the saved model file.
    """
    ensure_dir(models_dir)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"  [✓] Model saved → {filepath}")
    return filepath


def load_model(filename: str, models_dir: str = "models"):
    """
    Load a saved sklearn model from disk.

    Parameters:
        filename   : Name of the saved file (e.g., 'random_forest.pkl').
        models_dir : Directory where the model is stored.

    Returns:
        Loaded model object.
    """
    filepath = os.path.join(models_dir, filename)
    model = joblib.load(filepath)
    print(f"  [✓] Model loaded ← {filepath}")
    return model


# ── Plot Helpers ──────────────────────────────────────────────────────────────

def save_figure(fig, filename: str, viz_dir: str = "visualizations", dpi: int = 150) -> str:
    """
    Save a matplotlib figure to the visualizations directory.

    Parameters:
        fig      : matplotlib Figure object.
        filename : Output filename (e.g., 'class_distribution.png').
        viz_dir  : Target directory.
        dpi      : Resolution (dots per inch).

    Returns:
        str: Full path to the saved image.
    """
    ensure_dir(viz_dir)
    filepath = os.path.join(viz_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    print(f"  [✓] Plot saved  → {filepath}")
    return filepath


def set_plot_style() -> None:
    """
    Set a clean, consistent matplotlib style for all plots in this project.
    Uses seaborn-v0_8-whitegrid (compatible with seaborn ≥ 0.12).
    """
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")   # fallback for older versions

    # Custom font sizes
    plt.rcParams.update({
        "figure.facecolor": "#F8F9FA",
        "axes.facecolor":   "#F8F9FA",
        "axes.titlesize":   14,
        "axes.labelsize":   12,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  10,
        "font.family":      "sans-serif",
    })


# ── Class Label Helpers ───────────────────────────────────────────────────────

# Mapping from encoded integer labels back to human-readable strings
CLASS_LABEL_MAP = {0: "Acceptable", 1: "Good", 2: "Unacceptable", 3: "Very Good"}

# Ordered class names (used in classification reports & confusion matrix)
CLASS_NAMES = ["unacc", "acc", "good", "vgood"]
CLASS_DISPLAY_NAMES = ["Unacceptable", "Acceptable", "Good", "Very Good"]


def print_section_header(title: str) -> None:
    """Print a formatted section header to the console."""
    border = "=" * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}")
