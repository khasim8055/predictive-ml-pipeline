"""
Model training, comparison, and evaluation module.
Trains multiple classifiers, compares performance, and selects the best model.
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
import logging
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_models() -> dict:
    """Return a dictionary of models to compare."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
    }


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute evaluation metrics for a trained model.
    
    Args:
        model: Trained sklearn-compatible classifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def train_and_compare(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
) -> tuple:
    """
    Train all models, compare metrics, and select the best performer.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Label vectors
        feature_names: List of feature column names
        
    Returns:
        Tuple of (results_dict, best_model_name, best_model_object)
    """
    models = get_models()
    results = {}

    logger.info(f"Training and evaluating {len(models)} models...")
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("=" * 70)

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = {"metrics": metrics, "model": model}

        auc_str = f"{metrics.get('roc_auc', 'N/A'):>10}" if isinstance(metrics.get("roc_auc"), float) else f"{'N/A':>10}"
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {auc_str}")

    print("=" * 70)

    # Select best model by F1 score (balanced metric for imbalanced data)
    best_name = max(results, key=lambda k: results[k]["metrics"]["f1_score"])
    best_model = results[best_name]["model"]
    best_f1 = results[best_name]["metrics"]["f1_score"]

    logger.info(f"Best model: {best_name} (F1={best_f1:.4f})")

    # Feature importance (if available)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_importance = sorted(
            zip(feature_names, importances), key=lambda x: x[1], reverse=True
        )
        logger.info("Top 10 features:")
        for feat, imp in feature_importance[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
        results["feature_importance"] = feature_importance

    return results, best_name, best_model


def save_model(model, scaler, feature_names: list, model_name: str, metrics: dict, output_dir: str = "models"):
    """
    Save the trained model, scaler, and metadata.
    
    Args:
        model: Trained model object
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        model_name: Name of the model
        metrics: Evaluation metrics dictionary
        output_dir: Directory to save artifacts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(model, model_path)

    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "feature_names": feature_names,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "scaler_path": scaler_path,
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")


def generate_report(results: dict, output_dir: str = "reports") -> str:
    """
    Generate a markdown evaluation report.
    
    Args:
        results: Results dictionary from train_and_compare
        output_dir: Directory for the report
        
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)

    lines = [
        "# Model Evaluation Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Model Comparison\n",
        f"| {'Model':<25} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10} | {'ROC AUC':>10} |",
        f"|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|",
    ]

    for name, data in results.items():
        if name == "feature_importance":
            continue
        m = data["metrics"]
        auc = f"{m.get('roc_auc', 'N/A'):>10}" if isinstance(m.get("roc_auc"), float) else f"{'N/A':>10}"
        lines.append(
            f"| {name:<25} | {m['accuracy']:>10.4f} | {m['precision']:>10.4f} | {m['recall']:>10.4f} | {m['f1_score']:>10.4f} | {auc} |"
        )

    # Best model
    best_name = max(
        (k for k in results if k != "feature_importance"),
        key=lambda k: results[k]["metrics"]["f1_score"],
    )
    lines.append(f"\n**Best Model:** {best_name} (selected by F1 Score)\n")

    # Feature importance
    if "feature_importance" in results:
        lines.append("## Top 10 Feature Importances\n")
        lines.append(f"| {'Rank':>4} | {'Feature':<30} | {'Importance':>12} |")
        lines.append(f"|{'-'*6}|{'-'*32}|{'-'*14}|")
        for i, (feat, imp) in enumerate(results["feature_importance"][:10], 1):
            lines.append(f"| {i:>4} | {feat:<30} | {imp:>12.4f} |")

    # Confusion matrix for best model
    best_cm = results[best_name]["metrics"]["confusion_matrix"]
    lines.append(f"\n## Confusion Matrix ({best_name})\n")
    lines.append("```")
    lines.append(f"              Predicted 0    Predicted 1")
    lines.append(f"Actual 0      {best_cm[0][0]:>10}    {best_cm[0][1]:>10}")
    lines.append(f"Actual 1      {best_cm[1][0]:>10}    {best_cm[1][1]:>10}")
    lines.append("```")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")
    return report_path
