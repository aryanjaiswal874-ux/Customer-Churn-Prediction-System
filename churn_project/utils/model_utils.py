"""
Model Training & Evaluation Utilities
Builds sklearn pipelines, trains models, compares metrics, selects best model.
"""

import logging
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

logger = logging.getLogger(__name__)


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - StandardScales numerical features
    - OneHotEncodes categorical features (drop='first' to avoid multicollinearity)
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",  # gracefully handles unseen categories at inference
        drop="first",             # removes one dummy to avoid dummy trap
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, model: Any) -> Pipeline:
    """Attach preprocessor and model into a single Pipeline."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def get_candidate_models() -> Dict[str, Any]:
    """Return dict of model name → estimator instance."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handles class imbalance
            random_state=42,
            C=0.5,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=42,
        ),
    }


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a fitted pipeline on test data.
    Returns dict with Accuracy, Precision, Recall, F1 Score, ROC-AUC.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Yes", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="Yes", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, pos_label="Yes", zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score((y_test == "Yes").astype(int), y_prob)

    logger.info(f"Metrics: {metrics}")
    return metrics


def train_and_compare(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numerical_cols: list,
    categorical_cols: list,
) -> Tuple[Pipeline, str, Dict[str, Dict[str, float]]]:
    """
    Train all candidate models, evaluate on test set, select best by F1 Score.

    Returns:
        best_pipeline: Fitted Pipeline of the best model
        best_model_name: Name of best model
        all_results: {model_name: {metric: value}}
    """
    candidates = get_candidate_models()
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    all_results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        logger.info(f"Training {name}...")
        pipeline = build_pipeline(preprocessor, model)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        all_results[name] = metrics
        fitted_pipelines[name] = pipeline
        logger.info(f"{name}: {metrics}")

    # Select best model by F1 Score (best for imbalanced churn problem)
    best_model_name = max(all_results, key=lambda k: all_results[k]["F1 Score"])
    best_pipeline = fitted_pipelines[best_model_name]

    logger.info(f"\nBest model: {best_model_name} (F1={all_results[best_model_name]['F1 Score']:.4f})")
    return best_pipeline, best_model_name, all_results


def get_feature_importance(
    pipeline: Pipeline, numerical_cols: list, categorical_cols: list
) -> Tuple[list, np.ndarray]:
    """
    Extract feature names and importances from a fitted pipeline.
    Works for RandomForest and GradientBoosting (tree-based models).
    For LogisticRegression, uses |coef_| as importance proxy.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Get feature names after OHE
    cat_feature_names = (
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(categorical_cols)
        .tolist()
    )
    all_feature_names = numerical_cols + cat_feature_names

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return [], np.array([])

    return all_feature_names, importances
