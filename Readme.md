# MLflow Tracking Server

This repository contains the necessary files to set up an MLflow tracking server using Docker, with artifacts stored in S3 and the database stored locally.

## Getting Started

### Prerequisites

- Docker
- Docker Compose (optional)

### Build the Docker Image

```sh
docker build -t mlflow-tracking-server .
```
Run the MLflow Tracking Server
Using Docker:

```sh
docker run -p 8080:8080 -v $(pwd)/mlflow:/app/mlflow  --env AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
--env AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
--env AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION --env ENV=prod  mlflow-tracking-server
```
Using Docker Compose:

```sh
docker-compose up
```

# Setting up tracking using ML FLow

## Logging with MLflow: PyTorch Example

To log metrics, parameters, and models from a PyTorch training run, you can use the following code snippet:

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Dummy data
X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Simple model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    for epoch in range(10):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        mlflow.log_metric("loss", loss.item(), step=epoch)
    mlflow.pytorch.log_model(model, "model")
```

###Integrating MLflow with Hugging Face



To log metrics, parameters, and models from a Hugging Face training run, you can use the following code snippet:

```python
import mlflow
import mlflow.transformers
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# Load dataset and tokenizer
dataset = load_dataset('glue', 'mrpc')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess the data
def preprocess(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

dataset = dataset.map(preprocess, batched=True)

# Define model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)

# Training
with mlflow.start_run():
    mlflow.log_params(training_args.to_dict())
    trainer.train()
    mlflow.transformers.log_model(transformers_model=model, artifact_path="model", task="text-classification")
```

## Logging with MLflow: XGBoost Example

To log metrics, parameters, and models from an XGBoost training run, you can use the following code snippet:

```python
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# Train model
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 100,
}

with mlflow.start_run():
    mlflow.log_params(params)
    model = xgb.train(params, dtrain, evals=[(dtest, "eval")])
    mlflow.log_metric("rmse", model.eval(dtest, "rmse"))
    mlflow.xgboost.log_model(model, "model")
```



## Registering a Model

Once you've logged a model, you can register it in the MLflow Model Registry. Here's an example:

```python
import mlflow
from mlflow.tracking import MlflowClient

# Assuming you have already logged a model in a run
run_id = "your_run_id"
model_name = "your_model_name"
model_uri = f"runs:/{run_id}/model"

# Register the model
client = MlflowClient()
model_details = client.create_registered_model(model_name)
model_version = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id,
)

# Transition model stage
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
)
```

