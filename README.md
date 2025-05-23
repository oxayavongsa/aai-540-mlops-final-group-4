
# 🫀 Aorta Guard: Cardiovascular Disease Detection Pipeline

This repository contains the full end-to-end machine learning pipeline for predicting cardiovascular disease. The project is built for the AAI-540 MLOps course and demonstrates data engineering, feature engineering, model training, evaluation, and deployment readiness.

---

## 📂 Data Assets

| **File Name**                    | **Label**              | **Shape**     | **Main Use Case**           | **Human-readable?** | **Notes**                                               |
|----------------------------------|-------------------------|---------------|-----------------------------|----------------------|----------------------------------------------------------|
| `cardio_train.csv`              | Original Dataset        | (70,000, 13)  | Starting point for pipeline | ✅ Yes               | Raw dataset from Kaggle; requires cleaning and parsing   |
| `cardio_cleaned.csv`            | Cleaned Dataset         | (68,385, 12)  | EDA + Feature Engineering   | ✅ Yes               | Cleaned and transformed; interpretable values            |
| `cardio_final_preprocessed.csv` | Preprocessed Dataset    | (69,961, 14)  | Final modeling input        | ❌ No                | Encoded, scaled, ready for ML model training             |

---

## 📂 Data Splits

These files are stratified subsets derived from the preprocessed dataset to simulate real-world deployment stages.

| **File Name** | **Purpose**                         | **Split %** |
|---------------|-------------------------------------|-------------|
| `train.csv`   | Training the model                  | ~40%        |
| `val.csv`     | Model tuning/validation             | ~10%        |
| `test.csv`    | Final evaluation before deployment  | ~10%        |
| `prod.csv`    | Reserved for production simulation  | ~40%        |

All splits ensure balanced representation of the `cardio` target using stratified sampling.

---

## 📊 Visual Insights Summary

- **BMI Distribution**: Higher BMI is associated with a greater risk of cardiovascular disease.
- **Blood Pressure Categories**: Cardio-positive patients tend to fall in Stage 1 and Stage 2 categories.
- **BMI Category**: Obesity and overweight statuses are more common in the cardio-positive class.
- **Age Groups**: Risk increases notably in patients aged 50 and older.
- **Cholesterol/BMI Ratio**: Slightly elevated in cardio-positive cases, suggesting metabolic concerns.
- **Pulse Pressure**: Higher and more varied in patients with cardiovascular disease.

---

## 🧠 Authors & Team

**Group 4: AAI-540 ML Design Project**  
- Prema Mallikarjunan  
- Outhai Xayavongsa (Team Lead)

---
