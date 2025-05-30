# 🫀 Aorta Guard: Cardiovascular Disease Detection Pipeline

This repository contains the complete machine learning pipeline for detecting cardiovascular disease using clinical and lifestyle indicators. Developed as part of the AAI-540 MLOps course, the project includes data ingestion, cleaning, feature engineering, model-ready preprocessing, data splitting, and infrastructure preparation.

---
## 🎯 Objective

[Dataset Source: Cardiovascular Disease Dataset on Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)

<I>Cardiovascular disease (CVD) remains the leading cause of death globally, yet many detection methods and clinical responses are still reactive, addressing issues only after symptoms emerge. This project tackles that gap by developing a machine learning model capable of predicting the likelihood of CVD before critical events like heart attacks or strokes occur. By analyzing routine health metrics such as blood pressure, cholesterol, glucose levels, age, and lifestyle habits. We aim to uncover predictive patterns that enable earlier, data-driven interventions.

This is a binary classification problem, where the goal is to predict the presence or absence of cardiovascular disease. Beyond prediction, the model empowers physicians to make preventive decisions, reduce emergency incidents, and improve long-term health outcomes. Our mission is to shift the standard of care from reactive treatment to proactive prevention using existing, accessible clinical data.</i>

---
## 📁 Project Structure

```bash
├── data_assets/
│   ├── cardio_train.csv
│   ├── cardio_cleaned.csv
│   ├── cardio_engineered.csv
│
├── data_splits/
│   ├── cardio_train_split40%_v2.csv
│   ├── cardio_val_split10%_v2.csv
│   ├── cardio_test_split10%_v2.csv
│   └── cardio_prod_split40%_v2.csv
│
├── feature_store/
│   ├── cardio_feature_store_setup.ipynb
│   └── cardio_engineered_feature_store_setup.ipynb
│
├── notebooks_pipeline/
│   ├── Models/
│   │   └── cardio_logistic_baseline.ipynb   # (Only logistic baseline here)
│   ├── cardio_data_split_v3.ipynb
│   ├── cardio_eda_and_feature_engineering.ipynb
│   ├── cardio_logistic_baseline_complete.ipynb
│   ├── cardio_model_evaluation_compare.ipynb
│   ├── cardio_preprocessing.ipynb
│   └── cardio_random_forest_complete.ipynb
│
├── requirements.txt
├── README.md
```
---
## 📊 Dataset Summary

| File Path | Label | Shape | Description |
| - | - | - | - |
| `cardio_train.csv` | Original Dataset | (70,000, 13) | Raw dataset from Kaggle; requires cleaning |
| `cardio_cleaned.csv` | Cleaned Dataset | (68,385, 12) | Cleaned and formatted for EDA and feature engineering |
| `cardio_engineered.csv` | Engineered Dataset | (68,385, 24) | Feature engineered version including BMI, pulse pressure, interaction features |
| `cardio_train_split40%_v2.csv` | Training Split (40%) | (27,355, 24) | Preprocessed dataset for model training |
| `cardio_val_split10%_v2.csv` | Validation Split (10%) | (6,838, 24) | Dataset for model validation and tuning |
| `cardio_test_split10%_v2.csv` | Test Split (10%) | (6,838, 24) | Dataset for final model evaluation |
| `cardio_prod_split40%_v2.csv` | Production Reserve (40%) | (27,354, 24) | Reserved for future inference or deployment |

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
This pipeline engineered and transformed features based on the cleaned dataset. We removed outliers, performed feature interactions, categorical binning, and constructed clinically meaningful variables.
* Input fields: Clinical, lifestyle, demographic variables
* Categorical binning: bp_category, bmi_category, age_group
* Engineered features: bmi, pulse_pressure, age_years, chol_bmi_ratio, lifestyle_score, age_gluc_interaction
* Final feature set: 24 features total
* Final output file: cardio_engineered.csv stored in data_assets/

### 🔬 Baseline Model: Logistic Regression

- Algorithm: Logistic Regression (Scikit-learn)
- Hyperparameters: Default (`max_iter=1000`, `random_state=42`)
- Trained on: 40% Training Set
- Validation AUC: 0.791
- Validation Accuracy: 73%
- Model file saved: `models/cardio_logistic_baseline.ipynb`

### 🔬 Initial Model: Random Forest

- Algorithm: Random Forest Classifier (Scikit-learn)
- Hyperparameters: Default (`n_estimators=100`, `random_state=42`)
- Trained on: 40% Training Set
- Validation AUC: 0.797
- Validation Accuracy: 73%
- Model file saved: `models/cardio_random_forest_complete.ipynb`

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
