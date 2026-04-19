"""
Data loading and preprocessing module for predictive maintenance pipeline.
Handles synthetic dataset generation, cleaning, feature engineering, and train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic predictive maintenance dataset.
    
    Simulates sensor readings from industrial machines with features like
    temperature, vibration, pressure, RPM, and operational hours.
    Machine failure is the binary target variable.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sensor features and failure labels
    """
    np.random.seed(random_state)
    logger.info(f"Generating synthetic dataset with {n_samples} samples...")

    # Machine metadata
    machine_ids = np.random.choice([f"M{i:03d}" for i in range(1, 51)], size=n_samples)
    machine_types = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n_samples, p=[0.4, 0.35, 0.25])
    operational_hours = np.random.exponential(scale=2000, size=n_samples).clip(100, 15000)

    # Sensor readings (normal operating conditions)
    temperature = np.random.normal(loc=70, scale=8, size=n_samples)
    vibration = np.random.normal(loc=40, scale=5, size=n_samples)
    pressure = np.random.normal(loc=100, scale=10, size=n_samples)
    rpm = np.random.normal(loc=1500, scale=200, size=n_samples)
    humidity = np.random.normal(loc=50, scale=10, size=n_samples).clip(10, 95)
    power_consumption = np.random.normal(loc=250, scale=30, size=n_samples)
    noise_level = np.random.normal(loc=75, scale=8, size=n_samples)

    # Inject failure patterns (correlated anomalies)
    failure_probability = np.full(n_samples, 0.05)  # baseline 5% failure rate

    # Strong failure signals (correlated with sensor readings)
    failure_probability += np.where(temperature > 78, 0.25, 0)
    failure_probability += np.where(vibration > 45, 0.25, 0)
    failure_probability += np.where(pressure > 110, 0.15, 0)
    failure_probability += np.where(operational_hours > 6000, 0.20, 0)
    failure_probability += np.where(rpm > 1700, 0.10, 0)
    failure_probability += np.where(power_consumption > 280, 0.12, 0)
    
    # Combined risk: machines with BOTH high temp AND high vibration are very likely to fail
    failure_probability += np.where((temperature > 75) & (vibration > 43), 0.30, 0)

    # Type-specific failure rates
    failure_probability += np.where(np.array(machine_types) == "TypeC", 0.05, 0)

    # Add noise
    failure_probability += np.random.uniform(-0.03, 0.03, size=n_samples)
    failure_probability = failure_probability.clip(0, 1)

    # Generate binary failure labels (~12-15% failure rate)
    failure = (np.random.uniform(size=n_samples) < failure_probability).astype(int)

    # Introduce some missing values (~2% randomly)
    mask_temp = np.random.random(n_samples) < 0.02
    mask_vib = np.random.random(n_samples) < 0.015
    mask_pressure = np.random.random(n_samples) < 0.01

    temperature_with_na = temperature.copy()
    vibration_with_na = vibration.copy()
    pressure_with_na = pressure.copy()

    temperature_with_na[mask_temp] = np.nan
    vibration_with_na[mask_vib] = np.nan
    pressure_with_na[mask_pressure] = np.nan

    df = pd.DataFrame({
        "machine_id": machine_ids,
        "machine_type": machine_types,
        "operational_hours": np.round(operational_hours, 1),
        "temperature": np.round(temperature_with_na, 2),
        "vibration": np.round(vibration_with_na, 2),
        "pressure": np.round(pressure_with_na, 2),
        "rpm": np.round(rpm, 1),
        "humidity": np.round(humidity, 2),
        "power_consumption": np.round(power_consumption, 2),
        "noise_level": np.round(noise_level, 2),
        "failure": failure,
    })

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Failure rate: {failure.mean():.2%}")
    logger.info(f"Missing values: temperature={mask_temp.sum()}, vibration={mask_vib.sum()}, pressure={mask_pressure.sum()}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: handle missing values, remove duplicates, validate ranges.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    initial_shape = df.shape

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Fill missing values with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled {n_missing} missing values in '{col}' with median={median_val:.2f}")

    # Clip physical impossibilities
    df["temperature"] = df["temperature"].clip(0, 200)
    df["vibration"] = df["vibration"].clip(0, 200)
    df["pressure"] = df["pressure"].clip(0, 300)
    df["rpm"] = df["rpm"].clip(0, 5000)
    df["humidity"] = df["humidity"].clip(0, 100)

    logger.info(f"Shape: {initial_shape} -> {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture domain-relevant patterns.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    logger.info("Engineering features...")

    # Interaction features
    df["temp_vibration_ratio"] = df["temperature"] / (df["vibration"] + 1e-6)
    df["power_per_rpm"] = df["power_consumption"] / (df["rpm"] + 1e-6)
    df["temp_pressure_product"] = df["temperature"] * df["pressure"]

    # Binned operational hours (machine age categories)
    df["machine_age_bin"] = pd.cut(
        df["operational_hours"],
        bins=[0, 2000, 5000, 8000, 15000],
        labels=["new", "mid_life", "aging", "old"],
    )

    # Z-score based anomaly flags
    for col in ["temperature", "vibration", "pressure", "rpm"]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[f"{col}_zscore"] = (df[col] - mean_val) / (std_val + 1e-6)
        df[f"{col}_anomaly"] = (df[f"{col}_zscore"].abs() > 2).astype(int)

    # Aggregate anomaly count
    anomaly_cols = [c for c in df.columns if c.endswith("_anomaly")]
    df["total_anomalies"] = df[anomaly_cols].sum(axis=1)

    # Encode categoricals
    le_type = LabelEncoder()
    df["machine_type_encoded"] = le_type.fit_transform(df["machine_type"])

    le_age = LabelEncoder()
    df["machine_age_encoded"] = le_age.fit_transform(df["machine_age_bin"].astype(str))

    logger.info(f"Added {len(df.columns)} total columns after feature engineering")
    return df


def prepare_splits(
    df: pd.DataFrame,
    target: str = "failure",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Prepare train/test splits with feature scaling.
    
    Args:
        df: Feature-engineered DataFrame
        target: Target column name
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    logger.info("Preparing train/test splits...")

    # Select numeric features only (drop IDs, raw categoricals, target)
    drop_cols = ["machine_id", "machine_type", "machine_age_bin", target]
    feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype in [np.float64, np.int64, np.int32]]

    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train failure rate: {y_train.mean():.2%}, Test failure rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, feature_cols, scaler


if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv("data/raw_dataset.csv", index=False)
    
    df = clean_data(df)
    df = engineer_features(df)
    df.to_csv("data/processed_dataset.csv", index=False)
    
    X_train, X_test, y_train, y_test, features, scaler = prepare_splits(df)
    print(f"\nFeatures used ({len(features)}): {features}")
    print(f"Ready for model training.")
