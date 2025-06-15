# 🫀 Aorta Guard: Cardiovascular Disease Detection Pipeline

This repository contains the complete machine learning pipeline for detecting cardiovascular disease using clinical and lifestyle indicators. Developed as part of the AAI-540 MLOps course, the project includes data ingestion, cleaning, feature engineering, model training, batch inference, feature store setup, and SageMaker + CloudWatch monitoring.

---

## 🎯 Objective

[Dataset Source: Cardiovascular Disease Dataset on Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data)

*Cardiovascular disease (CVD) remains the leading cause of death globally. Many clinical interventions are reactive rather than preventive. This project uses machine learning to identify individuals at risk based on routine health metrics proactively.*

We aim to shift from reactive care to proactive prevention using accessible, structured clinical data.

---

## 📁 Project Structure

```bash
├── cloudwatch_and_monitoring/
│   ├── cardio_cloudwatch_and_data_reports.ipynb
│   ├── cardio_data_and_infrastructure_monitors.ipynb
│   ├── cardio_data_quality_monitoring_schedule.ipynb
│   ├── cardio_delete_enpoint_and_monitoring_schedule.ipynb
│   └── cardio_model_monitoring.ipynb
│
├── data_assets/
│   ├── cloudwatch_files/
│   │   ├── constraints.json
│   │   └── statistics.json
│   ├── data_splits/                                        # Data Split 40/10/40/10
│   │   ├── cardio_prod_split40.csv
│   │   ├── cardio_test_split10.csv
│   │   ├── cardio_train_split40.csv
│   │   └── cardio_val_split10.csv
│   ├── logistic/
│   │   ├── cardio_prod_no_label.csv
│   │   ├── inference.py
│   │   ├── logistic_model.pkl
│   │   └── logistic_model.tar.gz
│   ├── random_forest/
│   │   ├── cardio_prod_no_label_rf.csv
│   │   ├── final_rf_model.tar.gz
│   │   └── inference_rf.py
│   ├── cardio_cleaned.csv
│   ├── cardio_engineered.csv
│   ├── cardio_engineered_clean.csv
│   └── cardio_train.csv
│
├── feature_store/
│   └── cardio_engineered_feature_store_setup.ipynb
│
├── image_output/                                            # All Images contained in the Notebook
│   ├── Final Model Feature Importance.png
│   ├── confusion_matrix_comparison.png
│   ├── roc_curve_comparison.png
│   ├── precision_recall_curve_comparison.png
│   ├── feature_importance-random_forest.png
│   ├── permutation_feature_importances-lollipop_plot.png
│   ├── class_distribution_of_cardio_outcome.png
│   ├── age_distribution.png
│   ├── bmi_distribution_by_cardio_outcome.png
│   ├── bmi_dist_by_cardio_outcome_real_data.png
│   ├── cholesterol_levels_by_cardio_outcome.png
│   ├── pulse_chol_age_lifestyle_by_cardio_outcome.png
│   ├── cloudwatch_results_3hr_060925.png
│   └── cardio endpoint dashboard.png
│
├── notebooks_pipeline/
│   ├── Models/
│       ├── cardio_logistic_baseline_v2.ipynb
│       └── cardio_random_forest.ipynb
│   ├── cardio_final_model.ipynb                             # Completed Final Notebook
│   └── cardio_inference_transform_job_v2.ipynb
│
├── requirements.txt                                         pip install -r requirements.txt
├── MIT License
└── README.md
```

---

## 🧪 Setup & Dependencies
### Requirements
To install the full environment:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Summary

| File Path                         | Label                   | Shape        | Description                                                                 |
|----------------------------------|--------------------------|--------------|-----------------------------------------------------------------------------|
| `cardio_train.csv`               | Original Dataset         | (70,000, 13) | Raw dataset from Kaggle; includes label and all original features          |
| `cardio_cleaned.csv`             | Cleaned Dataset          | (68,385, 12) | Cleaned version with outliers removed and improved formatting              |
| `cardio_engineered.csv`          | Engineered Dataset       | (68,385, 24) | Includes clinical + engineered features (bmi, pulse pressure, etc.)        |
| `cardio_engineered_clean.csv`    | Final Engineered Clean   | (68,385, 24) | Fully cleaned and engineered dataset for modeling                          |
| `cardio_train_split40.csv`       | Training Split (40%)     | (27,355, 24) | Used to train both baseline and optimized models                           |
| `cardio_val_split10.csv`         | Validation Split (10%)   | (6,838, 24)  | Used to validate and tune hyperparameters                                  |
| `cardio_test_split10.csv`        | Test Split (10%)         | (6,838, 24)  | Used to evaluate final model performance                                   |
| `cardio_prod_split40.csv`        | Production Reserve (40%) | (27,354, 24) | Held-out dataset for production inference and monitoring                   |
| `cardio_prod_no_label.csv`       | Prod No-Label (Logistic) | (27,354, 23) | Inference-ready dataset for logistic regression (labels removed)          |
| `cardio_prod_no_label_rf.csv`    | Prod No-Label (RF)       | (27,354, 23) | Inference-ready dataset for random forest model (labels removed)          |

## 📊 Visual Insights Summary

- **Top Feature Importances**: The Random Forest model ranked `systolic_bp`, `chol_bmi_ratio`, and `bmi` as the most important predictors of cardiovascular disease. This confirms the critical role of circulatory and metabolic health markers in early detection.
![Final Model Feature Importance](https://github.com/user-attachments/assets/e53a8a6d-d068-4235-beae-246a055980aa)
- **Cardio Outcome by Age Group**: Risk increases significantly in individuals in their 50s and 60s. The highest number of positive cases is concentrated in the 50s group, emphasizing the need for proactive screening in middle age.
![cardio_outcome_by_age_group](https://github.com/user-attachments/assets/2ad999a9-4241-4d5d-bb97-b837eb6cff32)
- **BMI Distribution**: Higher BMI is associated with greater cardiovascular risk, especially among those classified as overweight or obese.
- **Blood Pressure Categories**: Stage 1 and Stage 2 hypertension are more common in cardio-positive individuals, linking elevated blood pressure to disease risk.
- **BMI Category Trends**: Obese and overweight individuals showed greater prevalence of disease, highlighting BMI's diagnostic relevance.
- **Cholesterol/BMI Ratio**: This ratio is slightly elevated in cardio-positive cases, suggesting metabolic imbalance or lipid-related risk.
- **Pulse Pressure Observations**: Higher pulse pressure ranges were observed in the cardio-positive group, suggesting greater vascular strain.
- **Pairplot Observations**: In BMI vs. chol_bmi_ratio, a clear inverse relationship and cluster separation between classes appear, hinting at decision boundaries the model may exploit.

---

## 🧠 Feature Engineering

This pipeline engineered and transformed features from the cleaned cardiovascular dataset to enhance model accuracy and interpretability. The process included outlier removal, derived metrics, and binning based on clinical relevance.

- **Input Sources**: Clinical, lifestyle, and demographic indicators.
- **Engineered Features**:
  - `bmi`: Body Mass Index (from height and weight)
  - `pulse_pressure`: Difference between systolic and diastolic BP
  - `age_years`: Converted from days to years
  - `chol_bmi_ratio`: Cholesterol level divided by BMI
  - `age_gluc_interaction`: Interaction between age and glucose level
  - `lifestyle_score`: Composite score from smoking, alcohol, and physical activity
- **Categorical Binning**:
  - `bp_category`: Hypertension stage (normal, stage1, stage2)
  - `bmi_category`: Weight classification (underweight, normal, overweight, obese)
  - `age_group`: Age buckets (30s, 40s, 50s, 60s)
- **Final Feature Count**: 24 input features
- **Final Output File**: `cardio_engineered.csv` stored in `data_assets/`

These engineered features enabled both linear (Logistic Regression) and non-linear (Random Forest) models to capture medical patterns that would be missed with raw data alone.

---

## 🔬 Model Overview

We developed and evaluated three versions of our cardiovascular disease prediction models:

### ⚙️ Baseline Model: Logistic Regression

- **Algorithm**: Logistic Regression (Scikit-learn)
- **Hyperparameters**: `max_iter=1000`, `random_state=42`
- **Training Set**: 40% of the cleaned and engineered dataset
- **Validation Accuracy**: 73%
- **Validation AUC**: 0.791
- **Inference Files**: 
  - `logistic_model.pkl`
  - `inference.py`
  - `logistic_model.tar.gz`
- **Batch Input**: `cardio_prod_no_label.csv`  
- **Stored In**: `data_assets/logistic/`

---

### 🌲 Initial Model: Random Forest

- **Algorithm**: Random Forest Classifier (Scikit-learn)
- **Hyperparameters**: `n_estimators=100`, `random_state=42`
- **Training Set**: 40% of the engineered dataset
- **Validation Accuracy**: 73%
- **Validation AUC**: 0.797
- **Inference Files**: 
  - `final_rf_model.joblib`
  - `inference_rf.py`
  - `final_rf_model.tar.gz`
- **Batch Input**: `cardio_prod_no_label_rf.csv`
- **Stored In**: `data_assets/random_forest/`

---

### 🏁 Final Model: Random Forest (Tuned)

- **Improved Hyperparameters**: Tuned with `RandomizedSearchCV`
- **Performance**: Slight AUC improvement over baseline
- **Reason for Selection**: Better feature importance explanations and flexible deployment
- **Deployment**: Used in real-time monitoring endpoint (`cardio-logistic-monitor-endpoint`)
- **Monitoring**: Integrated with SageMaker Model Monitor and CloudWatch Dashboards

---

## 🔍 Model Evaluation

The precision-recall curve illustrates the trade-off between sensitivity (recall) and the precision of our classifier at various thresholds. This is especially helpful in imbalanced medical datasets like cardiovascular prediction, where false positives and false negatives carry significant clinical weight.

![precision_recall_curve_comparison](https://github.com/user-attachments/assets/2aee0fd6-3738-46bb-bb22-a2e036b29f8c)

## 📡 Monitoring & CloudWatch Insights

Our SageMaker deployment includes real-time monitoring using Amazon CloudWatch. The dashboard tracks CPU and memory utilization, disk activity, and invocation error rates. Below is a 3-hour snapshot of model monitoring activity.
![cloudwatch_results_3hr_060925](https://github.com/user-attachments/assets/3897609f-b5b8-45ee-92ca-43ea5ff67095)

---

## 🧪 How to Test the Final Model

```python
import boto3

# Initialize runtime client (change as needed)
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Replace with your deployed endpoint name
endpoint_name = "cardio-logistic-monitor-endpoint"

# Example payload: a comma-separated string of 23 feature values
payload = "50,2,5.51,136.69,110,80,1,1,0,0,1,21.98,50s,Normal,30,4.55,66.12,50,0,stage1,normal,50,-1"

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=payload
)

prediction = response["Body"].read().decode("utf-8")
print("Prediction:", prediction.strip())

---

## 👥 Team Info

**AAI-540 Group 4 – Aorta Guard**

- Prema Mallikarjunan  
- Outhai Xayavongsa *(Team Lead)*
