# NLP Transfer Learning with Hugging Face Transformers

An end-to-end NLP project focused on fine-tuning, evaluating, and deploying a pretrained BERT model for text classification using Hugging Face Transformers, PyTorch, and AWS SageMaker.

## Project Overview

This project explores transfer learning for NLP by fine-tuning a pretrained `bert-base-cased` model on the Yelp Review Full dataset.

The project covers the complete machine learning workflow, from dataset preparation and tokenization to model training, evaluation, and cloud deployment.

## Technologies

- **Python / PyTorch:** Deep learning backend used by Hugging Face Trainer.
- **Hugging Face Transformers:** Pretrained BERT model and Trainer API
- **Hugging Face Datasets:** Dataset loading and preprocessing
- **Evaluate / NumPy:** Model evaluation
- **AWS SageMaker:** Cloud-based model training and planned deployment
- **Amazon S3:** Model artifact storage

## Dataset and Model

**Dataset:** [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full)

A text classification dataset containing customer reviews labeled with ratings from 1 to 5.

**Model:** [bert-base-cased](https://huggingface.co/google-bert/bert-base-cased)

A pretrained BERT transformer fine-tuned for five-class text classification.

## Project Workflow

1. **Data Loading:** Load the Yelp Review Full dataset using Hugging Face Datasets.
2. **Preprocessing:** Tokenize reviews, apply padding and truncation, and prepare training and evaluation subsets.
3. **Model Fine-Tuning:** Fine-tune BERT using the Hugging Face Trainer API.
4. **Evaluation:** Evaluate classification performance using accuracy and evaluation loss.
5. **Cloud Training:** Execute GPU-accelerated training jobs using AWS SageMaker.
6. **Deployment:** Deploy the fine-tuned model for inference and expose predictions through an application interface.

## Project Structure

```text
├── load_dataset.py
├── tokenizer.py
├── train_HFTrainer.py
├── README.md
├── requirements.txt
└── .gitignore
```

Additional evaluation and deployment modules will be added as the project progresses.

## Getting Started

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train_HFTrainer.py
```

A compatible accelerator is recommended for BERT fine-tuning.

## AWS SageMaker Integration

The project will use SageMaker managed training jobs to fine-tune BERT on cloud GPU infrastructure.

The planned workflow includes:

- Configuring the SageMaker execution environment
- Launching remote training jobs
- Saving model artifacts to Amazon S3
- Evaluating the trained model
- Deploying the model for inference

Cloud training and deployment instructions will be added once the implementation is complete.

## Project Status

- [x] Dataset loading
- [x] Tokenization and preprocessing
- [x] Hugging Face Trainer implementation
- [ ] Successful fine-tuning and evaluation
- [ ] AWS SageMaker training integration
- [ ] Independent model evaluation
- [ ] Model deployment and inference interface

## Reference

Based on the [Hugging Face fine-tuning tutorial](https://huggingface.co/docs/transformers/training), with independent implementation and extensions for cloud training and deployment.
