# =============================================================================
# main.py — Full ML Pipeline Entry Point
# Author  : Mayank | Intern ID: CTTS148 | CodTech Internship
# Run     : python main.py
# Purpose : Orchestrates the entire pipeline from data loading to evaluation.
# =============================================================================

import os
import sys

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set matplotlib backend to Agg to prevent plt.show() from blocking
import matplotlib
matplotlib.use('Agg')

from src.data_loader    import load_data, inspect_data
from src.preprocessing  import preprocess_pipeline, encode_features, handle_missing_values, remove_duplicates, FEATURE_CATEGORIES
from src.eda            import run_full_eda
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models
from src.utils          import print_section_header

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "car.csv")


def main():
    """
    Run the full Car Evaluation Machine Learning pipeline.

    Steps:
        1.  Load dataset
        2.  Inspect data (shape, dtypes, missing values, duplicates)
        3.  Handle missing values
        4.  Remove duplicates
        5.  Run Exploratory Data Analysis (10 visualisations)
        6.  Encode features (OrdinalEncoder) & target (LabelEncoder)
        7.  Train/test split (80/20, stratified)
        8.  Train models (Logistic Regression, Decision Tree,
                          Random Forest, XGBoost)
        9.  Evaluate all models
        10. Print summary
    """
    print_section_header("CAR EVALUATION SYSTEM — ML PIPELINE")
    print(f"  Intern  : Mayank")
    print(f"  ID      : CTTS148")
    print(f"  Company : CodTech IT Solutions")

    # ── Step 1: Load Data ─────────────────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Step 2: Inspect ───────────────────────────────────────────────────────
    inspect_data(df)

    # ── Step 3A: Handle Missing Values ────────────────────────────────────────
    df = handle_missing_values(df)

    # ── Step 3B: Remove Duplicates ────────────────────────────────────────────
    df = remove_duplicates(df)

    # ── Step 3C: EDA (before encoding so we use original string values) ───────
    # First do a quick encode just to pass to EDA heatmaps
    from src.preprocessing import encode_features
    X_encoded_for_eda, _, _, _ = encode_features(df.copy())
    run_full_eda(df, X_encoded_for_eda)

    # ── Steps 4–6: Encode + Split (full pipeline) ─────────────────────────────
    X_train, X_test, y_train, y_test, ordinal_enc, label_enc = \
        preprocess_pipeline(df)

    feature_names = list(FEATURE_CATEGORIES.keys())
    class_names   = list(label_enc.classes_)

    # ── Step 7: Train Models ──────────────────────────────────────────────────
    models = train_all_models(X_train, y_train)

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    results = evaluate_all_models(
        models, X_test, y_test,
        feature_names=feature_names,
        label_classes=class_names
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    print_section_header("PIPELINE COMPLETE")
    print("  All models trained and evaluated.")
    print("  Visualisations saved to  : visualizations/")
    print("  Models saved to          : models/")
    print("\n  To launch the Streamlit app, run:")
    print("    streamlit run app.py\n")


if __name__ == "__main__":
    main()
