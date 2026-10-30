import boto3
from datetime import datetime, timezone

# AWS configuration
REGION = "us-east-1"
PROFILE = "sagemaker-dev"
BUCKET = "sagemaker-us-east-1-228736140710"

# IAM role used by SageMaker.
ROLE_ARN = (
    "arn:aws:iam::228736140710:"
    "role/SageMakerNLPExecutionRole"
)

# Fine-tuned BERT model from our latest training run.
MODEL_S3_URI = (
    f"s3://{BUCKET}/nlp-bert-training/"
    "nlp-bert-training-20261029105825/"
    "output/model.tar.gz"
)

# Use the existing AWS credentials.
session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION
)

sagemaker_client = session.client("sagemaker")
s3_client = session.client("s3")

# PyTorch GPU container used for SageMaker evaluation.
# This is the same image family used during training.
EVALUATION_IMAGE = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "pytorch-training:2.7.1-gpu-py312"
)

# Location of the evaluation code archive in S3.
CODE_S3_URI = (
    f"s3://{BUCKET}/nlp-evaluation/code/evaluation_code.tar.gz"
)

# Directory where SageMaker will place our input archives.
MODEL_INPUT_DIR = "/opt/ml/processing/input/model"
CODE_INPUT_DIR = "/opt/ml/processing/input/code"

# Directory where evaluation results will be saved.
OUTPUT_DIR = "/opt/ml/processing/output"


# 1. Generate a unique name for this evaluation job.
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
job_name = f"nlp-bert-evaluation-{timestamp}"

# 2. Specify where SageMaker should save evaluation results.
results_s3_uri = f"s3://{BUCKET}/nlp-evaluation/results/{job_name}/"

# 3. Define the commands that run inside the GPU container.
#
# SageMaker downloads the two archives into the input folders.
# We extract them into temporary directories and run the
# evaluation script from the extracted project directory.
#
# MODEL_DIR tells evaluate_model.py where to find the trained
# model. SM_OUTPUT_DIR tells it where to write metrics.json.
#
# "set -e" stops execution immediately if any command fails.
evaluation_command = (
    "set -e; "
    "mkdir -p /tmp/c /tmp/m; "
    f"tar -xzf {CODE_INPUT_DIR}/evaluation_code.tar.gz -C /tmp/c; "
    f"tar -xzf {MODEL_INPUT_DIR}/model.tar.gz -C /tmp/m; "
    "cd /tmp/c; "
    "pip install -r requirements.txt; "
    "python -m src.evaluate_model"
)

# 4. Configure the Processing job.
processing_job_config = {
    "ProcessingJobName": job_name,
    "RoleArn": ROLE_ARN,

    # Use a single NVIDIA GPU instance.
    "ProcessingResources": {
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.g5.xlarge",
            "VolumeSizeInGB": 30
        }
    },

    # Limit evaluation to a maximum of 2 hours.
    "StoppingCondition": {
        "MaxRuntimeInSeconds": 7200
    },

    # Override the container's default startup command.
    "AppSpecification": {
        "ImageUri": EVALUATION_IMAGE,
        "ContainerEntrypoint": ["bash", "-lc"],
        "ContainerArguments": [evaluation_command]
    },

    # Pass directory locations to our evaluation script.
    "Environment": {
        "MODEL_DIR": "/tmp/m",
        "SM_OUTPUT_DIR": OUTPUT_DIR,
        "PYTHONUNBUFFERED": "1"
    },

    # Download the fine-tuned model and evaluation code from S3.
    "ProcessingInputs": [
        {
            "InputName": "model",
            "S3Input": {
                "S3Uri": MODEL_S3_URI,
                "LocalPath": MODEL_INPUT_DIR,
                "S3DataType": "S3Prefix",
                "S3InputMode": "File"
            }
        },
        {
            "InputName": "code",
            "S3Input": {
                "S3Uri": CODE_S3_URI,
                "LocalPath": CODE_INPUT_DIR,
                "S3DataType": "S3Prefix",
                "S3InputMode": "File"
            }
        }
    ],

    # Automatically upload metrics.json to S3 after evaluation.
    "ProcessingOutputConfig": {
        "Outputs": [
            {
                "OutputName": "evaluation-results",
                "S3Output": {
                    "S3Uri": results_s3_uri,
                    "LocalPath": OUTPUT_DIR,
                    "S3UploadMode": "EndOfJob"
                }
            }
        ]
    }
}


# 5. Submit the evaluation job.
if __name__ == "__main__":
    print(f"Evaluation job: {job_name}")
    print(f"GPU: ml.g5.xlarge")
    print(f"Model: {MODEL_S3_URI}")
    print(f"Results destination: {results_s3_uri}")

    response = sagemaker_client.create_processing_job(
        **processing_job_config
    )

    print("\nProcessing job submitted successfully!")
    print(f"Job ARN: {response['ProcessingJobArn']}")
    print("Use the AWS Console or CLI to monitor its status.")
