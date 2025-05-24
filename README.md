
# 🫀 Aorta Guard: Cardiovascular Disease Detection Pipeline

This repository contains the complete machine learning pipeline for detecting cardiovascular disease using clinical and lifestyle indicators. Developed as part of the AAI-540 MLOps course, the project includes data ingestion, cleaning, feature engineering, model-ready preprocessing, data splitting, and infrastructure preparation.

---

## 📁 Project Structure

```bash
├── data_assets/
│   ├── cardio_train.csv                  # Raw dataset from Kaggle
│   ├── cardio_cleaned.csv                # Cleaned and interpretable version
│   ├── cardio_final_preprocessed.csv     # Encoded and scaled version for modeling
│   └── cardio_engineered.csv             # Added engineered features from domain knowledge
│
├── data_splits/
│   ├── cardio_train_split40%.csv         # Training set (~40%)
│   ├── cardio_val_split10%.csv           # Validation set (~10%)
│   ├── cardio_test_split10%.csv          # Test set (~10%)
│   └── cardio_prod_split40%.csv          # Production reserve set (~40%)
│
├── notebooks_pipeline/
│   ├── cardio_data_split.ipynb           # Stratified data split logic
│   ├── cardio_eda_and_feature_engineering.ipynb  # EDA and feature engineering
│   └── cardio_preprocessing.ipynb        # Standardization, encoding, export
│
├── cardio_feature_store_setup.ipynb      # Setup for SageMaker Feature Store - original data
├── cardio_new_feature_store_setup.ipynb      # Setup for SageMaker Feature Store - new cleaned features data
├── requirements.txt                      # Required packages for the pipeline
├── README.md                             # Project documentation
```
---
## 📊 Dataset Summary

| File                            | Label                 | Shape        | Description                                     |
| ------------------------------- | --------------------- | ------------ | ----------------------------------------------- |
| `cardio_train.csv`              | Original Dataset      | (70,000, 13) | Raw dataset from Kaggle; requires cleaning      |
| `cardio_cleaned.csv`            | Cleaned Dataset       | (68,385, 12) | Cleaned and filtered for EDA                    |
| `cardio_final_preprocessed.csv` | Preprocessed Dataset  | (69,961, 14) | Encoded and scaled for model input              |
| `cardio_engineered.csv`         | Feature-Augmented Set | (68,385, 22) | Added engineered features from domain knowledge |

---
## 🔀 Data Splits Overview
| File                        | Purpose                    | Percentage |
| --------------------------- | -------------------------- | ---------- |
| `cardio_train_split40%.csv` | Model training set         | 40%        |
| `cardio_val_split10%.csv`   | Validation set             | 10%        |
| `cardio_test_split10%.csv`  | Evaluation/test set        | 10%        |
| `cardio_prod_split40%.csv`  | Production simulation data | 40%        |

---
## 📊 Visual Insights Summary
* BMI Distribution: Higher BMI is associated with greater cardiovascular risk.
* Blood Pressure Categories: Stage 1 and Stage 2 dominate in cardio-positive individuals.
* BMI Category: Obese and overweight individuals are more likely to have cardiovascular disease.
* Age Groups: Elevated risk is seen in individuals aged 50 and older.
* Cholesterol/BMI Ratio: Slightly higher in positive cases; may reflect metabolic imbalance.
* Pulse Pressure: Wider and higher ranges suggest vascular strain in positive cases.
* Pairplot Observations: Unique banding pattern in BMI vs. chol_bmi_ratio reveals inverse relationships and possible decision boundaries between cardio classes.

---
## 🧠 Feature Engineering
This pipeline engineered and transformed features based on the cleaned dataset. We removed outliers and converted variables into more insightful representations.
* Used fields: All clinical, lifestyle, and demographic variables
* Combined or bucketed: bp_category, bmi_category, age_group
* Engineered features: bmi, pulse_pressure, age_years, chol_bmi_ratio, lifestyle_score, age_gluc_interaction
* Final output: cardio_engineered.csv stored in data_assets/

## 🧪 Setup & Dependencies
### Requirements
To install the full environment:
```bash
pip install -r requirements.txt
```
---
## 👥 Team Info
AAI-540 Group 4 – Aorta Guard
* Prema Mallikarjunan
* Outhai Xayavongsa (Team Lead)
