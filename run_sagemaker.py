import boto3
from sagemaker.train import ModelTrainer
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.configs import Compute
from sagemaker.core import image_uris
from sagemaker.train.configs import SourceCode
from sagemaker.train.configs import StoppingCondition

# Authenticate using our AWS CLI profile.
aws_session = boto3.Session(
    profile_name="sagemaker-dev",
    region_name="us-east-1"
)

# Create a SageMaker session using the same AWS credentials.
sagemaker_session = Session(boto_session=aws_session)

# Configure the cloud computing resources for model training.
compute = Compute(
    instance_type="ml.g5.xlarge",
    instance_count=1
)

# Retrieve the AWS-provided PyTorch GPU training image.
training_image_uri = image_uris.retrieve(
    framework="pytorch",
    region="us-east-1",
    version="2.7.1",
    py_version="py312",
    image_scope="training",
    instance_type="ml.g5.xlarge"
)

# Specify the Python code and dependencies SageMaker should use.
source_code = SourceCode(
    source_dir=".",
    command="python -m src.train_HFTrainer",
    requirements="requirements.txt",
    ignore_patterns=[
        ".venv",
        ".git",
        "__pycache__",
        "test_trainer",
        "saved_model",
        ".env"
    ]
)

# Configure the SageMaker training job.
model_trainer = ModelTrainer(
    sagemaker_session=sagemaker_session,
    role="arn:aws:iam::228736140710:role/SageMakerNLPExecutionRole",
    base_job_name="nlp-bert-training",
    training_image=training_image_uri,
    source_code=source_code,
    compute=compute,
    stopping_condition=StoppingCondition(
        max_runtime_in_seconds=7200
    )
)

# if __name__ == "__main__":
#     print("AWS Region:", aws_session.region_name)
#     print("GPU:", compute.instance_type)
#     print("Training image:", training_image_uri)
#     print("Training command:", source_code.command)
#     print("Configuration ready!")

# if __name__ == "__main__":
#     model_trainer.train(dry_run=True)
#     print("SageMaker configuration validated!")



if __name__ == "__main__":
    print("Submitting SageMaker training job...", flush=True)
    model_trainer.train(wait=True, logs=True)
    print("Training job completed.", flush=True)