# Aorta Guard 🫀

##📊 Project Overview

Aorta Guard is a machine learning system developed to predict the risk of cardiovascular disease (CVD) using anonymized clinical and lifestyle data. The project focuses on early detection and proactive intervention to prevent heart-related incidents before symptoms occur. By leveraging data from routine checkups such as blood pressure, cholesterol, glucose, and lifestyle factors, we aim to shift healthcare from reactive care to preventive strategy.

##🔖 Repository Structure

.
├── data_assets/                 # Raw and transformed datasets
│   ├── cardio_train.csv
│   ├── cardio_cleaned.csv
│   └── cardio_final_preprocessed.csv
│
├── data_splits/                # Stratified data subsets for model training and simulation
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── prod.csv
│
├── notebooks/                  # Jupyter notebooks (EDA, preprocessing, modeling)
│   ├── cardio_eda_after_preprocessing.ipynb
│   ├── cardio_preprocessing.ipynb
│   └── cardio_data_split.ipynb
│
├── models/                     # Model artifacts and saved weights
├── outputs/                    # Visualizations, SHAP plots, reports
└── README.md                   # Project documentation

##📈 Visual Insights

BMI Distribution: Higher BMI values are associated with increased cardio risk.

Blood Pressure Categories: Most individuals with CVD fall into stage 1 or stage 2 hypertension.

BMI Categories: Overweight and obese individuals show higher CVD incidence.

Age Groups: Cardio risk rises significantly in age groups 50s, 60s, and 70+.

Cholesterol/BMI Ratio: Higher ratios correspond to elevated cardio risk.

Pulse Pressure: Higher variability and median values among cardio-positive cases.

##⚙️ Tech Stack

Python 3.12, pandas, scikit-learn, seaborn, matplotlib, boto3, AWS S3

Environment: JupyterLab + SageMaker Studio

##💡 Usage Notes

Always begin with cardio_cleaned.csv for feature engineering.

Use cardio_final_preprocessed.csv and corresponding data_splits/ files for model training.

Avoid using cardio_train.csv directly for modeling.

##🧱 Authors

Group 4: Aorta Guard

Outhai Xayavongsa (Team Lead)

Prema Mallikarjunan

##📅 Project Deadline

June 23, 2025

🔗 Key Links

GitHub: Group 4 Repository

Asana Board: Task Tracker

Google Docs: Team Tracker
