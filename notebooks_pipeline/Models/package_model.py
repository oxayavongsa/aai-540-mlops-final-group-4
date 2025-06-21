import tarfile
import boto3
import os

MODEL_DIR = "model"
TAR_FILE = "logistic_model.tar.gz"
S3_BUCKET = "sagemaker-us-east-1-993768311527"
S3_KEY = f"logistic/{TAR_FILE}"

# Package model and inference script for Script Mode
with tarfile.open(TAR_FILE, "w:gz") as tar:
    tar.add(f"{MODEL_DIR}/logistic_model.pkl", arcname="logistic_model.pkl")
    tar.add(f"{MODEL_DIR}/inference.py", arcname="inference.py")

# Upload to S3
s3 = boto3.client("s3")
s3.upload_file(TAR_FILE, S3_BUCKET, S3_KEY)
print(f"Uploaded: s3://{S3_BUCKET}/{S3_KEY}")