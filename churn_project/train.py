"""
train.py - Customer Churn Prediction Training Script
=====================================================
Generates synthetic data, trains multiple models, selects the best,
saves the full sklearn Pipeline to models/churn_pipeline.pkl.

Usage:
    python train.py
"""

import logging
import os
import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_generator import generate_telecom_churn_data, get_feature_columns
from utils.model_utils import train_and_compare, get_feature_importance, evaluate_model
from utils.preprocessing import clean_dataframe


# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

PIPELINE_PATH = MODELS_DIR / "churn_pipeline.pkl"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
SAMPLE_CSV_PATH = DATA_DIR / "sample_template.csv"
TRAIN_DATA_PATH = DATA_DIR / "churn_train.csv"
LOG_PATH = PROJECT_ROOT / "training.log"


def setup_logging():
    """
    Windows-safe logging setup.
    - Console: plain ASCII only (no special chars that break cp1252)
    - File: UTF-8 encoded so the log file is readable
    """
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logging.getLogger(__name__)


def main():
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("Customer Churn Prediction - Training Pipeline")
    logger.info("=" * 60)

    # 1. Generate Dataset
    logger.info("Generating synthetic telecom churn dataset (5,000 rows)...")
    df = generate_telecom_churn_data(n_samples=5_000, random_state=42)
    df.to_csv(TRAIN_DATA_PATH, index=False)
    logger.info("Dataset saved to: %s", TRAIN_DATA_PATH)
    logger.info("Shape: %s  |  Churn rate: %.1f%%", df.shape, (df["Churn"] == "Yes").mean() * 100)

    # 2. Save Sample CSV Template
    sample = df.drop(columns=["Churn"]).head(10)
    sample.to_csv(SAMPLE_CSV_PATH, index=False)
    logger.info("Sample CSV template saved to: %s", SAMPLE_CSV_PATH)

    # 3. Clean and Prepare Features
    df = clean_dataframe(df)
    numerical_cols, categorical_cols = get_feature_columns()

    X = df[numerical_cols + categorical_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # 4. Train and Compare Models
    logger.info("Training candidate models (this may take a minute)...")
    best_pipeline, best_model_name, all_results = train_and_compare(
        X_train, X_test, y_train, y_test, numerical_cols, categorical_cols
    )

    # 5. Print Comparison Table
    logger.info("")
    logger.info("Model Comparison:")
    metrics_order = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    header = "{:<25}".format("Model") + "".join("{:>12}".format(m) for m in metrics_order)
    logger.info(header)
    logger.info("-" * len(header))
    for model_name, scores in all_results.items():
        marker = " << BEST" if model_name == best_model_name else ""
        row = "{:<25}".format(model_name) + "".join(
            "{:>12.4f}".format(scores.get(m, 0)) for m in metrics_order
        )
        logger.info(row + marker)

    logger.info("")
    logger.info("Best model selected: %s", best_model_name)

    # 6. Save Full Pipeline
    payload = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "all_results": all_results,
    }
    with open(PIPELINE_PATH, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Pipeline saved to: %s", PIPELINE_PATH)

    # 7. Save Metrics JSON
    metrics_data = {
        "best_model": best_model_name,
        "all_results": all_results,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    logger.info("Metrics saved to: %s", METRICS_PATH)

    # 8. Feature Importance
    feature_names, importances = get_feature_importance(
        best_pipeline, numerical_cols, categorical_cols
    )
    if len(feature_names):
        top_idx = np.argsort(importances)[::-1][:10]
        logger.info("")
        logger.info("Top-10 Feature Importances:")
        for i in top_idx:
            logger.info("  %-40s %.4f", feature_names[i], importances[i])

    logger.info("")
    logger.info("Training complete! Run:  streamlit run app/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
