"""
Unit tests for the predictive maintenance ML pipeline.
Tests data generation, cleaning, feature engineering, and model training.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import generate_synthetic_data, clean_data, engineer_features, prepare_splits
from model_training import get_models, evaluate_model, train_and_compare


class TestDataPipeline:
    """Tests for data generation and preprocessing."""

    def test_generate_data_shape(self):
        df = generate_synthetic_data(n_samples=500)
        assert df.shape[0] == 500
        assert "failure" in df.columns
        assert "temperature" in df.columns
        assert "machine_id" in df.columns

    def test_generate_data_types(self):
        df = generate_synthetic_data(n_samples=100)
        assert np.issubdtype(df["temperature"].dtype, np.floating)
        assert np.issubdtype(df["failure"].dtype, np.integer)
        assert pd.api.types.is_string_dtype(df["machine_type"])

    def test_failure_rate_reasonable(self):
        df = generate_synthetic_data(n_samples=5000)
        failure_rate = df["failure"].mean()
        assert 0.05 < failure_rate < 0.30, f"Failure rate {failure_rate:.2%} outside expected range"

    def test_missing_values_exist(self):
        df = generate_synthetic_data(n_samples=5000)
        total_missing = df.isnull().sum().sum()
        assert total_missing > 0, "Expected some missing values in synthetic data"

    def test_clean_data_no_nulls(self):
        df = generate_synthetic_data(n_samples=1000)
        df_clean = clean_data(df)
        assert df_clean.isnull().sum().sum() == 0

    def test_clean_data_preserves_rows(self):
        df = generate_synthetic_data(n_samples=1000)
        df_clean = clean_data(df)
        assert len(df_clean) <= len(df)
        assert len(df_clean) > 0

    def test_feature_engineering_adds_columns(self):
        df = generate_synthetic_data(n_samples=500)
        df = clean_data(df)
        original_cols = len(df.columns)
        df = engineer_features(df)
        assert len(df.columns) > original_cols

    def test_feature_engineering_creates_anomaly_flags(self):
        df = generate_synthetic_data(n_samples=500)
        df = clean_data(df)
        df = engineer_features(df)
        assert "total_anomalies" in df.columns
        assert "temperature_anomaly" in df.columns

    def test_prepare_splits_shapes(self):
        df = generate_synthetic_data(n_samples=1000)
        df = clean_data(df)
        df = engineer_features(df)
        X_train, X_test, y_train, y_test, features, scaler = prepare_splits(df)

        assert X_train.shape[0] == 800  # 80% of 1000
        assert X_test.shape[0] == 200   # 20% of 1000
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == 800
        assert len(y_test) == 200

    def test_stratified_split(self):
        df = generate_synthetic_data(n_samples=2000)
        df = clean_data(df)
        df = engineer_features(df)
        _, _, y_train, y_test, _, _ = prepare_splits(df)

        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.02, "Stratification failed"


class TestModelTraining:
    """Tests for model training and evaluation."""

    @pytest.fixture
    def sample_data(self):
        df = generate_synthetic_data(n_samples=1000)
        df = clean_data(df)
        df = engineer_features(df)
        return prepare_splits(df)

    def test_all_models_loadable(self):
        models = get_models()
        assert len(models) == 5
        assert "XGBoost" in models
        assert "RandomForest" in models

    def test_model_training_runs(self, sample_data):
        X_train, X_test, y_train, y_test, features, scaler = sample_data
        results, best_name, best_model = train_and_compare(X_train, X_test, y_train, y_test, features)

        assert len(results) >= 5
        assert best_name in results
        assert best_model is not None

    def test_metrics_in_range(self, sample_data):
        X_train, X_test, y_train, y_test, features, scaler = sample_data
        results, _, _ = train_and_compare(X_train, X_test, y_train, y_test, features)

        for name, data in results.items():
            if name == "feature_importance":
                continue
            m = data["metrics"]
            assert 0 <= m["accuracy"] <= 1
            assert 0 <= m["precision"] <= 1
            assert 0 <= m["recall"] <= 1
            assert 0 <= m["f1_score"] <= 1

    def test_confusion_matrix_shape(self, sample_data):
        X_train, X_test, y_train, y_test, features, scaler = sample_data
        results, _, _ = train_and_compare(X_train, X_test, y_train, y_test, features)

        for name, data in results.items():
            if name == "feature_importance":
                continue
            cm = data["metrics"]["confusion_matrix"]
            assert len(cm) == 2
            assert len(cm[0]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
