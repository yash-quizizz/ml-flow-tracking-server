
# Base image for running Python
FROM public.ecr.aws/docker/library/python:3.11.9-bookworm

# Install MLflow and boto3
RUN pip install --upgrade pip
RUN pip install mlflow boto3
RUN apt update
ENV AWS_DEFAULT_REGION=us-east-1
ENV AWS_ACCESS_KEY_ID=${aws_access_key_id}
ENV AWS_SECRET_ACCESS_KEY=${aws_secret_access_key}
ENV MLFLOW_S3_ENDPOINT_URL=https://s3.amazonaws.com



EXPOSE 8080:80


CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:///app/mlflow/mlflow.db --default-artifact-root s3://quizizz-ml-adhoc/mlflow-artifacts --host 0.0.0.0 --port 8080"]
