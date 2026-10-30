
import os
import json
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.load_dataset import load_dataset_from_huggingface
from src.tokenizer import tokenize_dataset


def evaluate_model():
    # 1. Select the device.
    # SageMaker's NVIDIA GPU is accessible through CUDA.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Evaluation device: {device}", flush=True)

    # 2. Load the original tokenizer used during fine-tuning.
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # 3. Load our fine-tuned BERT model.
    # MODEL_DIR will point to the model directory inside SageMaker.
    # When running locally, use ./saved_model instead.
    model_dir = os.environ.get("MODEL_DIR", "./saved_model")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )

    # Move the model to the selected device and disable dropout.
    model.to(device)
    model.eval()

    # 4. Load the original Yelp Review Full test split.
    # This is separate from the training and validation data
    # used in our latest training run.
    raw_dataset = load_dataset_from_huggingface()
    test_dataset = raw_dataset["test"]

    # Tokenize using the same settings as during training.
    test_dataset = tokenize_dataset(test_dataset)

    # Return the model inputs and labels as PyTorch tensors.
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # 5. Create batches for evaluation.
    # Batch size 16 is a starting point for our SageMaker GPU.
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )

    # 6. Run inference without computing gradients.
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for batch in test_loader:
            # Move input tensors to the GPU.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Run the fine-tuned BERT model.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Choose the class with the highest score.
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Move predictions back to CPU for metric calculation.
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    # 7. Calculate classification metrics.
    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    # Rows represent true classes; columns represent predictions.
    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(5))
    )

    # 8. Display the evaluation results.
    print("\nBERT Evaluation Results")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:\n", cm)

    # 9. Save results as JSON.
    # The Processing job will configure SM_OUTPUT_DIR
    # and upload the output directory to S3.
    output_dir = os.environ.get(
        "SM_OUTPUT_DIR",
        "./evaluation_results"
    )
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "model_directory": model_dir,
        "num_test_samples": len(all_labels),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }

    output_path = os.path.join(output_dir, "metrics.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    evaluate_model()
