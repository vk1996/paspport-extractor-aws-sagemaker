import boto3
from sagemaker import get_execution_role


version='9'
role = get_execution_role()
region=boto3.session.Session().region_name
account_id = boto3.client('sts').get_caller_identity().get('Account')
endpoint_name = "passport-extraction-sagemaker-deployment-"+version
model_bucket_name="passport-extraction-sagemaker-model-bucket"
model_location=f"s3://sagemaker-{region}-{account_id}/{model_bucket_name}/models.tar.gz"
inference_repository_uri=f"{account_id}.dkr.ecr.{region}.amazonaws.com/{endpoint_name}:test"+version
build_endpoint_docker=True
delete_endpoint_after_prediction=False