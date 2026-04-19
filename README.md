# Predictive Maintenance ML Pipeline

An end-to-end machine learning pipeline for predicting industrial machine failures using sensor data. Compares 5 classifiers, selects the best performer, and saves production-ready artifacts — all runnable with a single command.

## Overview

Manufacturing downtime from unexpected machine failures costs industries billions annually. This project builds a complete ML pipeline that ingests sensor data (temperature, vibration, pressure, RPM, power consumption), engineers predictive features, trains and compares multiple models, and outputs a deployment-ready classifier with evaluation metrics.

**Key capabilities:**
- Synthetic but realistic sensor data generation with configurable failure patterns
- Automated data cleaning, missing value imputation, and feature engineering
- 5-model comparison: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM
- Evaluation report with metrics table, confusion matrix, and feature importances
- Model serialization with metadata for deployment
- Unit tests (14 tests, pytest)
- Docker support for containerized execution

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Data Generation │────▶│  Cleaning &  │────▶│    Feature        │
│  (Sensor Data)   │     │  Validation  │     │    Engineering    │
└─────────────────┘     └──────────────┘     └───────┬───────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Model Artifacts │◀────│  Evaluation  │◀────│  Model Training   │
│  & Report        │     │  & Selection │     │  (5 Classifiers)  │
└─────────────────┘     └──────────────┘     └───────────────────┘
```

## Quick Start

```bash
# Clone the repo
git clone https://github.com/khasim8055/predictive-ml-pipeline.git
cd predictive-ml-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py

# Run with custom sample size
python run_pipeline.py --samples 20000
```

## Results

Pipeline output from a 10,000-sample run:

| Model              | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.7860   | 0.6613    | 0.1752 | 0.2770   | 0.7117  |
| Random Forest      | 0.7970   | 0.6937    | 0.2372 | 0.3535   | 0.7667  |
| **Gradient Boosting**| **0.7925** | **0.6188** | **0.2949** | **0.3994** | **0.7652** |
| XGBoost            | 0.7845   | 0.5808    | 0.2842 | 0.3816   | 0.7443  |
| SVM (RBF)          | 0.7900   | 0.7308    | 0.1624 | 0.2657   | 0.7099  |

**Best model:** Gradient Boosting (selected by F1 Score — chosen over accuracy because the dataset is imbalanced at ~23% failure rate, making F1 a more meaningful metric than accuracy for this use case)

## Feature Engineering

The pipeline creates 22 features from 8 raw sensor readings:

- **Interaction features:** temperature-vibration ratio, power-per-RPM, temperature-pressure product
- **Z-score anomaly flags:** per-sensor deviation detection (|z| > 2)
- **Aggregate anomaly count:** total number of anomalous sensor readings per observation
- **Machine age binning:** operational hours categorized into lifecycle stages (new → mid-life → aging → old)
- **Encoded categoricals:** machine type and age bin as numeric features

## Project Structure

```
predictive-ml-pipeline/
├── run_pipeline.py              # Main pipeline orchestrator
├── src/
│   ├── data_pipeline.py         # Data generation, cleaning, feature engineering
│   └── model_training.py        # Model training, evaluation, reporting
├── tests/
│   └── test_pipeline.py         # 14 unit tests
├── data/                        # Generated datasets (gitignored)
├── models/                      # Saved model artifacts (gitignored)
│   ├── best_model.joblib        # Serialized best model
│   ├── scaler.joblib            # Fitted StandardScaler
│   └── model_metadata.json      # Feature names, metrics, timestamp
├── reports/
│   └── evaluation_report.md     # Auto-generated evaluation report
├── Dockerfile                   # Containerized execution
├── requirements.txt
└── .gitignore
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Expected output: 14 passed
```

Tests cover data generation, cleaning, feature engineering, train/test splits, model training, metric ranges, and confusion matrix shapes.

## Docker

```bash
# Build
docker build -t predictive-ml-pipeline .

# Run
docker run --rm predictive-ml-pipeline
```

## Tech Stack

- **Language:** Python 3.11+
- **ML:** scikit-learn, XGBoost
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Testing:** pytest
- **Deployment:** Docker, joblib
- **Tracking:** MLflow-compatible artifact structure

## Design Decisions

- **Synthetic data over public datasets:** Allows full control over failure patterns and demonstrates data engineering skills, not just model fitting
- **F1 as selection metric:** With ~23% failure rate, accuracy is misleading (a model predicting all-healthy achieves 77% accuracy). F1 balances precision and recall for the minority class
- **5-model comparison:** Shows systematic evaluation rather than jumping to a single algorithm
- **Feature engineering before scaling:** Domain-specific features (interaction terms, anomaly flags) are created before StandardScaler to preserve interpretability
- **Modular architecture:** `data_pipeline.py` and `model_training.py` are independently testable and reusable

## Future Improvements

- [ ] Add SMOTE/oversampling for better minority class handling
- [ ] Implement hyperparameter tuning with Optuna or GridSearchCV
- [ ] Add MLflow experiment tracking integration
- [ ] Build a simple FastAPI inference endpoint
- [ ] Add time-series features (rolling averages, trend detection)

## Author

**Khasim Bin Saleh**
M.Sc. Computer Science (Big Data & AI) — SRH Berlin
[GitHub](https://github.com/khasim8055) | [LinkedIn](https://www.linkedin.com/in/khasim-bin-saleh)
