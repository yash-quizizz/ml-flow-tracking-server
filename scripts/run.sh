#!/bin/bash

docker run -p 8080:8080 -v $(pwd)/mlflow:/app/mlflow  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    --env AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION --env ENV=prod \
     mlflow-tracking-server
