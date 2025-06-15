
import pandas as pd
import joblib
import os
from io import StringIO

FEATURE_COLUMNS = [
    'age', 'height_ft', 'weight_lbs', 'systolic_bp', 'diastolic_bp',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi',
    'pulse_pressure', 'chol_bmi_ratio', 'height_in', 'age_years',
    'is_hypertensive', 'age_gluc_interaction', 'lifestyle_score',
    'gender', 'bp_category', 'bmi_category', 'age_group', 'cholesterol_label'
]

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "final_rf_model.joblib")
    return joblib.load(model_path)

def input_fn(input_data, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None)
        df.columns = FEATURE_COLUMNS
        return df
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return '
'.join(map(str, prediction))
