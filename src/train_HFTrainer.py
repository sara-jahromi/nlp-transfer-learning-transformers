
import numpy as np
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import os
import src.load_dataset as load_dataset
from .tokenizer import create_subset, tokenize_dataset

# Load accuracy as the evaluation metric.
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute classification accuracy from model predictions."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_model(train_dataset, eval_dataset):
    
    """Fine-tune BERT and evaluate it after each epoch."""

    # Load pretrained BERT with a five-class classification head.
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=5
    )

    # Configure training and evaluation.
    # Default training arguments for the Trainer class.
    # training_args = TrainingArguments(
    #     output_dir="test_trainer",
    #     eval_strategy="epoch"
    # )


    # Custom training arguments for SageMaker.
    training_args = TrainingArguments(
    output_dir="test_trainer",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    report_to="none"
    )


    # Initialize the trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Start fine-tuning.
    trainer.train()
    # Save the fine-tuned model.
    model_dir = os.environ.get("SM_MODEL_DIR", "./saved_model")
    trainer.save_model(model_dir)

    return trainer



if __name__ == "__main__":
    # Load the original Yelp dataset.
    raw_dataset = load_dataset.load_dataset_from_huggingface()

    # Create smaller subsets before tokenization.
    train_data, eval_data = create_subset(raw_dataset)

    # Tokenize the training and evaluation datasets.
    train_data = tokenize_dataset(train_data)
    eval_data = tokenize_dataset(eval_data)

    # Fine-tune BERT.
    trainer = train_model(train_data, eval_data)