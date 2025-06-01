import pickle
import pandas as pd
from io import StringIO

# Define feature columns (excluding cholesterol_label)
FEATURE_COLUMNS = [
    'bmi', 'pulse_pressure', 'chol_bmi_ratio', 'age_gluc_interaction', 'age_years',
    'bp_category', 'bmi_category', 'age_group', 'age', 'gender',
    'systolic_bp', 'diastolic_bp', 'cholesterol', 'gluc', 'smoke',
    'alco', 'active', 'is_hypertensive', 'lifestyle_score'
]

def model_fn(model_dir):
    with open(f"{model_dir}/logistic_model.pkl", "rb") as f:
        return pickle.load(f)

def input_fn(input_data, content_type):
    df = pd.read_csv(StringIO(input_data), header=None)
    df.columns = FEATURE_COLUMNS
    return df

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return '\n'.join(str(x) for x in prediction)
