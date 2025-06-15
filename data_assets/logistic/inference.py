
import pandas as pd
from io import StringIO
import joblib
import os

FEATURE_COLUMNS = ['age', 'gender', 'height_ft', 'weight_lbs', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'age_group', 'cholesterol_label', 'pulse_pressure', 'chol_bmi_ratio', 'height_in', 'age_years', 'is_hypertensive', 'bp_category', 'bmi_category', 'age_gluc_interaction', 'lifestyle_score']

def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "logistic_model.pkl"))

def input_fn(input_data, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None)
        if df.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {df.shape[1]}")
        df.columns = FEATURE_COLUMNS
        return df
    else:
        raise ValueError("Unsupported content type: " + content_type)

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return '\n'.join(str(x) for x in prediction)
