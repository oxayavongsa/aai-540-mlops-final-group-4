
import pandas as pd
from io import StringIO
import joblib
import os

# Define feature columns matching your final dataset (23 columns)
FEATURE_COLUMNS = [
    'age', 'height_ft', 'weight_lbs', 'systolic_bp', 'diastolic_bp',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi',
    'pulse_pressure', 'chol_bmi_ratio', 'height_in', 'age_years',
    'is_hypertensive', 'age_gluc_interaction', 'lifestyle_score', 'gender',
    'bp_category', 'bmi_category', 'age_group', 'cholesterol_label'
]

def model_fn(model_dir):
    import joblib
    import os
    return joblib.load(os.path.join(model_dir, "logistic_model.pkl"))

def input_fn(input_data, content_type):
    df = pd.read_csv(StringIO(input_data), header=None)
    df.columns = FEATURE_COLUMNS
    return df

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return '\n'.join(str(x) for x in prediction)
