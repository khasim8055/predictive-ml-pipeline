"""
Main pipeline runner for Predictive Maintenance ML Pipeline.
Orchestrates data generation, preprocessing, training, evaluation, and artifact saving.

Usage:
    python run_pipeline.py
    python run_pipeline.py --samples 20000
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline import generate_synthetic_data, clean_data, engineer_features, prepare_splits
from model_training import train_and_compare, save_model, generate_report


def main(n_samples: int = 10000):
    print("=" * 70)
    print("  PREDICTIVE MAINTENANCE ML PIPELINE")
    print("=" * 70)

    # Step 1: Generate data
    print("\n[1/5] Generating synthetic dataset...")
    df = generate_synthetic_data(n_samples=n_samples)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_dataset.csv", index=False)
    print(f"       Saved raw dataset: data/raw_dataset.csv ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Step 2: Clean data
    print("\n[2/5] Cleaning data...")
    df = clean_data(df)

    # Step 3: Feature engineering
    print("\n[3/5] Engineering features...")
    df = engineer_features(df)
    df.to_csv("data/processed_dataset.csv", index=False)
    print(f"       Saved processed dataset: data/processed_dataset.csv ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Step 4: Train and compare models
    print("\n[4/5] Training and comparing models...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_splits(df)
    results, best_name, best_model = train_and_compare(X_train, X_test, y_train, y_test, feature_names)

    # Step 5: Save artifacts
    print("\n[5/5] Saving model artifacts and report...")
    save_model(
        model=best_model,
        scaler=scaler,
        feature_names=feature_names,
        model_name=best_name,
        metrics=results[best_name]["metrics"],
    )
    report_path = generate_report(results)

    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE")
    print(f"  Best model: {best_name} (F1={results[best_name]['metrics']['f1_score']:.4f})")
    print(f"  Artifacts saved to: models/")
    print(f"  Report saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Maintenance ML Pipeline")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples to generate")
    args = parser.parse_args()
    main(n_samples=args.samples)
