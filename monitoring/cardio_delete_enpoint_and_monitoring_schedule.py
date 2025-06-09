
import boto3
from botocore.exceptions import ClientError

# Initialize SageMaker client
region = 'us-east-1'
sm_client = boto3.client('sagemaker', region_name=region)

# Names
endpoint_name = 'cardio-logistic-monitor-endpoint'
monitor_schedule_name = 'cardio-data-monitor-schedule'
endpoint_config_name = endpoint_name

# Delete Monitoring Schedule first
try:
    sm_client.delete_monitoring_schedule(MonitoringScheduleName=monitor_schedule_name)
    print(f"Deleted monitoring schedule: {monitor_schedule_name}")
except ClientError as e:
    if 'ResourceNotFound' in str(e):
        print(f"Monitoring schedule '{monitor_schedule_name}' not found.")
    else:
        print(f"Error deleting monitoring schedule: {e}")

# Delete Endpoint
try:
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Deleted endpoint: {endpoint_name}")
except ClientError as e:
    if 'Could not find endpoint' in str(e):
        print(f"⚠ Endpoint '{endpoint_name}' not found.")
    else:
        print(f"⚠ Error deleting endpoint: {e}")

# Delete Endpoint Config
try:
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print(f"Deleted endpoint config: {endpoint_config_name}")
except ClientError as e:
    if 'Could not find endpoint configuration' in str(e):
        print(f"Endpoint config '{endpoint_config_name}' not found.")
    else:
        print(f"Error deleting endpoint config: {e}")
