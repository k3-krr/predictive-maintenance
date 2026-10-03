# Predictive Maintenance for Campus Equipment

## 📌 Overview

Predictive Maintenance for Campus Equipment is a machine-learning project designed to predict potential equipment failures before they occur.

The project uses historical machine telemetry, maintenance records, failure records, and machine information to identify patterns associated with equipment failures and support proactive maintenance decisions.

## 🎯 Problem Statement

Unexpected equipment failures can interrupt campus operations and increase maintenance costs.

This project aims to analyze historical equipment data and predict whether a machine is likely to experience a failure within a defined future time window.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Machine Learning
- Git & GitHub

## 📂 Dataset

The project uses several datasets containing information about campus equipment:

- `PdM_machines.csv` — Machine information
- `PdM_telemetry.csv` — Sensor and telemetry readings
- `PdM_failures.csv` — Recorded machine failures
- `PdM_errors.csv` — Machine error records
- Maintenance records — Historical maintenance activity

## 🔍 Key Tasks

- Data cleaning and preprocessing
- Handling missing values
- Feature engineering
- Combining telemetry and maintenance information
- Identifying failure-related patterns
- Training machine-learning models
- Evaluating model performance
- Generating predictions for future equipment failures

## 📊 Prediction Objective

The main objective is to provide advance warning of potential equipment failures so that maintenance can be planned proactively instead of reacting after a breakdown occurs.

## 📁 Project Structure

```text
predictive-maintenance/
│
├── PdM_machines.csv
├── PdM_telemetry.csv
├── PdM_failures.csv
├── PdM_errors.csv
├── .gitignore
├── README.md
│
└── predictive_maintenance_day2_artifacts/
    ├── selected_model_24h.joblib
    ├── selected_model_6h.joblib
    ├── selected_models.csv
    ├── selected_predictions_24h.csv
    ├── selected_predictions_6h.csv
    ├── selected_unseen_predictions_24h.csv
    ├── selected_unseen_predictions_6h.csv
    └── validation_model_comparison.csv
