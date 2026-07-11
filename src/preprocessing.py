# =============================================================================
# preprocessing.py — Data Cleaning, Encoding & Splitting
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Purpose : Handle missing values, remove duplicates, ordinal-encode all
#           categorical features, and split the data into train/test sets.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import print_section_header, save_model


# ── Ordinal Category Orders ───────────────────────────────────────────────────
# We define the natural ordering for each feature so that OrdinalEncoder
# assigns meaningful integer values (e.g., low=0, med=1, high=2, vhigh=3).

FEATURE_CATEGORIES = {
    "buying":   ["low", "med", "high", "vhigh"],   # low→0, vhigh→3
    "maint":    ["low", "med", "high", "vhigh"],
    "doors":    ["2", "3", "4", "5more"],           # 2→0, 5more→3
    "persons":  ["2", "4", "more"],                 # 2→0, more→2
    "lug_boot": ["small", "med", "big"],            # small→0, big→2
    "safety":   ["low", "med", "high"],             # low→0, high→2
}

# Target class encoding order (unacc < acc < good < vgood)
TARGET_ORDER = ["unacc", "acc", "good", "vgood"]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    The UCI Car Evaluation dataset has NO missing values, but we demonstrate
    the technique by checking and applying forward-fill as a fallback strategy.

    Parameters:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    print_section_header("STEP 3A — Handling Missing Values")

    missing_before = df.isnull().sum().sum()
    print(f"  Missing values before handling : {missing_before}")

    if missing_before > 0:
        # For categorical data, forward-fill is a simple and common approach
        df = df.fillna(method="ffill")
        print("  Applied: forward-fill (ffill) to fill missing values")
    else:
        print("  No missing values found — dataset is clean!")

    missing_after = df.isnull().sum().sum()
    print(f"  Missing values after handling  : {missing_after}")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    print_section_header("STEP 3B — Removing Duplicates")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)

    removed = before - after
    print(f"  Rows before removal : {before}")
    print(f"  Rows after  removal : {after}")
    print(f"  Duplicate rows removed : {removed}")

    return df


def encode_features(df: pd.DataFrame):
    """
    Ordinal-encode all categorical feature columns using sklearn's
    OrdinalEncoder with explicit category orders.

    Also label-encode the target column ('class') using LabelEncoder.

    Why OrdinalEncoder (not OneHotEncoder)?
    ─────────────────────────────────────────
    All 6 features are ORDINAL (e.g., low < med < high < vhigh).
    OrdinalEncoder preserves this order, which helps tree-based models
    make better splits. OneHotEncoder would lose ordinal information.

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame with original string values.

    Returns:
        X_encoded (pd.DataFrame): Feature matrix with integer-encoded columns.
        y_encoded (np.ndarray)  : Encoded target labels (0–3).
        ordinal_enc (OrdinalEncoder): Fitted encoder for features (used in app).
        label_enc   (LabelEncoder)  : Fitted encoder for target (used in app).
    """
    print_section_header("STEP 4 — Feature Encoding")

    # ── Separate features and target ──────────────────────────────────────────
    feature_cols = list(FEATURE_CATEGORIES.keys())
    X = df[feature_cols].copy()
    y = df["class"].copy()

    # ── Encode features (OrdinalEncoder) ──────────────────────────────────────
    # Build the list of category arrays in column order
    categories = [FEATURE_CATEGORIES[col] for col in feature_cols]

    ordinal_enc = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    X_encoded_array = ordinal_enc.fit_transform(X)
    X_encoded = pd.DataFrame(X_encoded_array, columns=feature_cols)

    print("  Feature encoding (OrdinalEncoder):")
    for col, cats in zip(feature_cols, categories):
        print(f"    {col:12s}: {cats}")

    # ── Encode target (LabelEncoder) ──────────────────────────────────────────
    label_enc = LabelEncoder()
    label_enc.classes_ = np.array(TARGET_ORDER)   # fix the order
    y_encoded = label_enc.transform(y)

    print(f"\n  Target encoding (LabelEncoder):")
    for i, cls in enumerate(label_enc.classes_):
        print(f"    {i} → {cls}")

    print(f"\n  X shape: {X_encoded.shape}")
    print(f"  y shape: {y_encoded.shape}")

    return X_encoded, y_encoded, ordinal_enc, label_enc


def split_data(X: pd.DataFrame, y: np.ndarray,
               test_size: float = 0.2, random_state: int = 42):
    """
    Split features and labels into training and test sets.

    Uses stratified split to ensure class proportions are preserved
    in both train and test sets (important for imbalanced datasets).

    Parameters:
        X            : Feature DataFrame (encoded).
        y            : Target array (encoded).
        test_size    : Fraction of data to use for testing (default 20%).
        random_state : Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    print_section_header("STEP 5 — Train / Test Split")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # stratified to preserve class balance
    )

    print(f"  Total samples : {len(X)}")
    print(f"  Train samples : {len(X_train)} ({100*(1-test_size):.0f}%)")
    print(f"  Test  samples : {len(X_test)} ({100*test_size:.0f}%)")
    print(f"  Random state  : {random_state}")
    print(f"  Stratified    : Yes")

    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df: pd.DataFrame):
    """
    Run the full preprocessing pipeline:
      1. Handle missing values
      2. Remove duplicates
      3. Encode features + target
      4. Split into train/test

    Parameters:
        df (pd.DataFrame): Raw loaded dataset.

    Returns:
        X_train, X_test, y_train, y_test,
        ordinal_enc (OrdinalEncoder),
        label_enc   (LabelEncoder)
    """
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    X, y, ordinal_enc, label_enc = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Save encoders for use in the Streamlit app
    save_model(ordinal_enc, "ordinal_encoder.pkl")
    save_model(label_enc,   "label_encoder.pkl")

    return X_train, X_test, y_train, y_test, ordinal_enc, label_enc
