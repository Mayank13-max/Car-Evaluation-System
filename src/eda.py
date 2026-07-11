# =============================================================================
# eda.py — Exploratory Data Analysis & Visualisations
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Purpose : Generate 10 informative plots to understand the dataset before
#           training any models.  All charts are saved to visualizations/.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from src.utils import print_section_header, save_figure, set_plot_style

# ── Colour Palette & Class Order ─────────────────────────────────────────────
CLASS_COLORS  = {"unacc": "#E63946", "acc": "#4361EE", "good": "#2DC653", "vgood": "#F9844A"}
CLASS_ORDER   = ["unacc", "acc", "good", "vgood"]
CLASS_LABELS  = {"unacc": "Unacceptable", "acc": "Acceptable",
                 "good": "Good", "vgood": "Very Good"}
FEATURE_COLS  = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]


# ── 1. Class Distribution Bar Chart ──────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot the distribution of the target 'class' column as a bar chart.
    Shows both count and percentage for each class.
    """
    set_plot_style()
    counts = df["class"].value_counts().reindex(CLASS_ORDER)
    percentages = 100 * counts / counts.sum()
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASS_ORDER, counts.values, color=colors,
                  edgecolor="white", linewidth=0.8, width=0.55)

    for bar, pct, cnt in zip(bars, percentages, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticklabels(["Unacceptable", "Acceptable", "Good", "Very Good"])
    ax.set_xlabel("Car Acceptability Class", fontsize=12)
    ax.set_ylabel("Number of Cars", fontsize=12)
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()

    if save:
        save_figure(fig, "01_class_distribution.png")
    plt.show(); plt.close(fig)


# ── 2. Class Distribution Pie Chart ──────────────────────────────────────────

def plot_class_pie(df: pd.DataFrame, save: bool = True) -> None:
    """
    Pie chart showing the proportion of each class in the dataset.
    """
    set_plot_style()
    counts = df["class"].value_counts().reindex(CLASS_ORDER)
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER]
    labels = [CLASS_LABELS[c] for c in CLASS_ORDER]
    explode = [0.05] * 4

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140, explode=explode,
        textprops={"fontsize": 11}, pctdistance=0.8
    )
    for at in autotexts:
        at.set_fontweight("bold")

    ax.set_title("Class Distribution (Pie Chart)", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()

    if save:
        save_figure(fig, "02_class_distribution_pie.png")
    plt.show(); plt.close(fig)


# ── 3. Feature Value Distributions ───────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, save: bool = True) -> None:
    """
    Subplots showing the value counts for each of the 6 features.
    Helps understand the spread of each feature's categories.
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLS):
        counts = df[col].value_counts().sort_index()
        axes[i].bar(counts.index, counts.values,
                    color="#4361EE", edgecolor="white", linewidth=0.7, alpha=0.85)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Category", fontsize=10)
        axes[i].set_ylabel("Count", fontsize=10)

        for j, (label, val) in enumerate(zip(counts.index, counts.values)):
            axes[i].text(j, val + 5, str(val), ha="center", fontsize=9)

    fig.suptitle("Feature Value Distributions", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        save_figure(fig, "03_feature_distributions.png")
    plt.show(); plt.close(fig)


# ── 4. Correlation Heatmap (Encoded Features) ─────────────────────────────────

def plot_correlation_heatmap(X_encoded: pd.DataFrame, save: bool = True) -> None:
    """
    Heatmap of Pearson correlations among encoded feature columns.
    Helps identify which features move together.
    """
    set_plot_style()
    corr = X_encoded.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # hide upper triangle duplicates
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save:
        save_figure(fig, "04_correlation_heatmap.png")
    plt.show(); plt.close(fig)


# ── 5. Buying Price vs Class ──────────────────────────────────────────────────

def plot_buying_vs_class(df: pd.DataFrame, save: bool = True) -> None:
    """
    Grouped bar chart showing how buying price category relates to car class.
    Key insight: very high buying price → almost always Unacceptable.
    """
    set_plot_style()
    buying_order = ["low", "med", "high", "vhigh"]
    ct = pd.crosstab(df["buying"], df["class"])[CLASS_ORDER]
    ct = ct.reindex(buying_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", ax=ax, color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            edgecolor="white", linewidth=0.7, width=0.72, rot=0)

    ax.set_xlabel("Buying Price", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Buying Price vs Car Class", fontsize=14, fontweight="bold", pad=15)
    ax.legend(title="Class", labels=list(CLASS_LABELS.values()), framealpha=0.9)
    ax.set_xticklabels(["Low", "Medium", "High", "Very High"])
    plt.tight_layout()

    if save:
        save_figure(fig, "05_buying_vs_class.png")
    plt.show(); plt.close(fig)


# ── 6. Safety vs Class ───────────────────────────────────────────────────────

def plot_safety_vs_class(df: pd.DataFrame, save: bool = True) -> None:
    """
    Grouped bar chart showing how safety rating relates to car class.
    Key insight: low safety → almost exclusively Unacceptable.
    """
    set_plot_style()
    safety_order = ["low", "med", "high"]
    ct = pd.crosstab(df["safety"], df["class"])[CLASS_ORDER]
    ct = ct.reindex(safety_order)

    fig, ax = plt.subplots(figsize=(9, 6))
    ct.plot(kind="bar", ax=ax, color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            edgecolor="white", linewidth=0.7, width=0.65, rot=0)

    ax.set_xlabel("Safety Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Safety Rating vs Car Class", fontsize=14, fontweight="bold", pad=15)
    ax.legend(title="Class", labels=list(CLASS_LABELS.values()), framealpha=0.9)
    ax.set_xticklabels(["Low", "Medium", "High"])
    plt.tight_layout()

    if save:
        save_figure(fig, "06_safety_vs_class.png")
    plt.show(); plt.close(fig)


# ── 7. Luggage Boot vs Class ─────────────────────────────────────────────────

def plot_luggage_vs_class(df: pd.DataFrame, save: bool = True) -> None:
    """
    Grouped bar chart for luggage boot size vs car class.
    """
    set_plot_style()
    lug_order = ["small", "med", "big"]
    ct = pd.crosstab(df["lug_boot"], df["class"])[CLASS_ORDER]
    ct = ct.reindex(lug_order)

    fig, ax = plt.subplots(figsize=(9, 6))
    ct.plot(kind="bar", ax=ax, color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            edgecolor="white", linewidth=0.7, width=0.65, rot=0)

    ax.set_xlabel("Luggage Boot Size", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Luggage Boot Size vs Car Class", fontsize=14, fontweight="bold", pad=15)
    ax.legend(title="Class", labels=list(CLASS_LABELS.values()), framealpha=0.9)
    ax.set_xticklabels(["Small", "Medium", "Big"])
    plt.tight_layout()

    if save:
        save_figure(fig, "07_luggage_vs_class.png")
    plt.show(); plt.close(fig)


# ── 8. Persons (Capacity) vs Class ──────────────────────────────────────────

def plot_persons_vs_class(df: pd.DataFrame, save: bool = True) -> None:
    """
    Stacked percentage bar chart for persons capacity vs car class.
    Shows relative composition of each class per capacity level.
    """
    set_plot_style()
    persons_order = ["2", "4", "more"]
    ct = pd.crosstab(df["persons"], df["class"])[CLASS_ORDER]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.reindex(persons_order)

    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(persons_order))
    for cls in CLASS_ORDER:
        ax.bar(persons_order, ct_pct[cls].values, bottom=bottom,
               label=CLASS_LABELS[cls], color=CLASS_COLORS[cls],
               edgecolor="white", linewidth=0.7)
        bottom += ct_pct[cls].values

    ax.set_xlabel("Person Capacity", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Person Capacity vs Class (Stacked %)", fontsize=14,
                 fontweight="bold", pad=15)
    ax.legend(title="Class", loc="upper right", framealpha=0.9)
    ax.set_xticklabels(["2 Persons", "4 Persons", "More"])
    plt.tight_layout()

    if save:
        save_figure(fig, "08_persons_vs_class.png")
    plt.show(); plt.close(fig)


# ── 9. Maintenance Cost vs Class ─────────────────────────────────────────────

def plot_maint_vs_class(df: pd.DataFrame, save: bool = True) -> None:
    """
    Grouped bar chart for maintenance cost vs car class.
    """
    set_plot_style()
    maint_order = ["low", "med", "high", "vhigh"]
    ct = pd.crosstab(df["maint"], df["class"])[CLASS_ORDER]
    ct = ct.reindex(maint_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(kind="bar", ax=ax, color=[CLASS_COLORS[c] for c in CLASS_ORDER],
            edgecolor="white", linewidth=0.7, width=0.72, rot=0)

    ax.set_xlabel("Maintenance Cost", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Maintenance Cost vs Car Class", fontsize=14, fontweight="bold", pad=15)
    ax.legend(title="Class", labels=list(CLASS_LABELS.values()), framealpha=0.9)
    ax.set_xticklabels(["Low", "Medium", "High", "Very High"])
    plt.tight_layout()

    if save:
        save_figure(fig, "09_maint_vs_class.png")
    plt.show(); plt.close(fig)


# ── 10. Pairwise Feature Heatmap (Encoded) ──────────────────────────────────

def plot_feature_class_heatmap(df: pd.DataFrame,
                                X_encoded: pd.DataFrame,
                                save: bool = True) -> None:
    """
    Heatmap showing the mean encoded value of each feature per class.
    Higher mean → feature skews toward higher ordinal values for that class.
    """
    set_plot_style()
    X_enc = X_encoded.copy()
    X_enc["class"] = df["class"].values

    mean_by_class = X_enc.groupby("class")[list(X_encoded.columns)].mean()
    mean_by_class = mean_by_class.reindex(CLASS_ORDER)
    mean_by_class.index = [CLASS_LABELS[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(mean_by_class, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Mean Encoded Value"})
    ax.set_title("Mean Feature Value per Class (Encoded)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    plt.tight_layout()

    if save:
        save_figure(fig, "10_feature_class_heatmap.png")
    plt.show(); plt.close(fig)


# ── Full EDA Pipeline ─────────────────────────────────────────────────────────

def run_full_eda(df: pd.DataFrame, X_encoded: pd.DataFrame = None) -> None:
    """
    Run all 10 EDA visualisations in sequence.

    Parameters:
        df         : Raw (original) DataFrame with string categories.
        X_encoded  : Encoded feature DataFrame (needed for heatmap plots).
    """
    print_section_header("STEP 3 — Exploratory Data Analysis (10 Visualisations)")

    print("\n  [1/10] Class Distribution Bar Chart ...")
    plot_class_distribution(df)

    print("  [2/10] Class Distribution Pie Chart ...")
    plot_class_pie(df)

    print("  [3/10] Feature Value Distributions ...")
    plot_feature_distributions(df)

    if X_encoded is not None:
        print("  [4/10] Correlation Heatmap ...")
        plot_correlation_heatmap(X_encoded)

    print("  [5/10] Buying Price vs Class ...")
    plot_buying_vs_class(df)

    print("  [6/10] Safety Rating vs Class ...")
    plot_safety_vs_class(df)

    print("  [7/10] Luggage Boot vs Class ...")
    plot_luggage_vs_class(df)

    print("  [8/10] Persons Capacity vs Class ...")
    plot_persons_vs_class(df)

    print("  [9/10] Maintenance Cost vs Class ...")
    plot_maint_vs_class(df)

    if X_encoded is not None:
        print("  [10/10] Feature–Class Heatmap ...")
        plot_feature_class_heatmap(df, X_encoded)

    print("\n  All visualisations saved to visualizations/")
