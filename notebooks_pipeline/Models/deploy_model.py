import boto3
import time

timestamp = int(time.time())

MODEL_NAME = f"logistic-scriptmode-{timestamp}"
TRANSFORM_JOB_NAME = f"logistic-transform-{timestamp}"
ROLE_ARN = "arn:aws:iam::993768311527:role/LabRole"
REGION = "us-east-1"

S3_BUCKET = "sagemaker-us-east-1-993768311527"
MODEL_ARTIFACT = f"s3://{S3_BUCKET}/logistic/logistic_model.tar.gz"
INPUT_DATA = f"s3://{S3_BUCKET}/cardio_data/cardio_prod_no_label.csv"
OUTPUT_PATH = f"s3://{S3_BUCKET}/logistic/output/"

# Script Mode Python container (generic Python environment)
PYTHON_IMAGE = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"

sagemaker = boto3.client("sagemaker")

# Create model with Script Mode environment
sagemaker.create_model(
    ModelName=MODEL_NAME,
    ExecutionRoleArn=ROLE_ARN,
    PrimaryContainer={
        "Image": PYTHON_IMAGE,
        "ModelDataUrl": MODEL_ARTIFACT,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": MODEL_ARTIFACT,
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": REGION
        }
    }
)

# Launch batch transform job
sagemaker.create_transform_job(
    TransformJobName=TRANSFORM_JOB_NAME,
    ModelName=MODEL_NAME,
    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": INPUT_DATA
            }
        },
        "ContentType": "text/csv",
        "SplitType": "Line"
    },
    TransformOutput={
        "S3OutputPath": OUTPUT_PATH
    },
    TransformResources={
        "InstanceType": "ml.m5.large",
        "InstanceCount": 1
    }
)

print(f"Batch transform started: {TRANSFORM_JOB_NAME}")
print(f"Model: {MODEL_NAME}")