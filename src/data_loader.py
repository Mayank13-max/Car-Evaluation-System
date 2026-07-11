# =============================================================================
# data_loader.py — Data Loading & Initial Inspection
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Purpose : Load the UCI Car Evaluation dataset and print a summary of its
#           structure, data types, and basic statistics.
# =============================================================================

import pandas as pd
from src.utils import print_section_header


# ── Column Definitions ────────────────────────────────────────────────────────

# Canonical column names matching the UCI Car Evaluation dataset
COLUMN_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Car Evaluation CSV dataset into a pandas DataFrame.

    The dataset has a header row (buying, maint, doors, persons,
    lug_boot, safety, class).

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print_section_header("STEP 1 — Loading Dataset")

    # Read the CSV file; it already contains a header row
    df = pd.read_csv(filepath)

    print(f"  Dataset loaded from : {filepath}")
    print(f"  Shape               : {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a structured inspection report of the dataset including:
      - First 5 rows
      - Data types and non-null counts
      - Missing value counts per column
      - Duplicate row count
      - Unique value counts per feature

    Parameters:
        df (pd.DataFrame): Dataset to inspect.
    """
    print_section_header("STEP 2 — Data Inspection")

    # ── First 5 rows ──────────────────────────────────────────────────────────
    print("\n  ── First 5 Rows ──")
    print(df.head().to_string(index=False))

    # ── Data types ────────────────────────────────────────────────────────────
    print("\n  ── Data Types & Non-Null Counts ──")
    print(df.info())

    # ── Missing values ────────────────────────────────────────────────────────
    print("\n  ── Missing Values Per Column ──")
    missing = df.isnull().sum()
    print(missing.to_string())
    total_missing = missing.sum()
    print(f"\n  Total missing values: {total_missing}")

    # ── Duplicate rows ────────────────────────────────────────────────────────
    duplicates = df.duplicated().sum()
    print(f"\n  ── Duplicate Rows: {duplicates} ──")

    # ── Unique values per feature ─────────────────────────────────────────────
    print("\n  ── Unique Values Per Feature ──")
    for col in df.columns:
        uniq = df[col].unique().tolist()
        print(f"  {col:12s} ({len(uniq)} unique): {uniq}")

    # ── Target class distribution ─────────────────────────────────────────────
    print("\n  ── Target Class Distribution ──")
    class_counts = df["class"].value_counts()
    for cls, cnt in class_counts.items():
        pct = 100 * cnt / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {cls:8s}: {cnt:5d} ({pct:5.1f}%)  {bar}")


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series with the missing value count per column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Column-wise missing value counts.
    """
    return df.isnull().sum()


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Return the number of duplicate rows in the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        int: Number of duplicate rows.
    """
    return df.duplicated().sum()
