from transformers import AutoTokenizer
import src.load_dataset as load_dataset

# Load the pretrained BERT tokenizer.
# The tokenizer converts text into token IDs that BERT can process.
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_fn(examples):
    # Tokenize a batch of text examples.
    # truncation=True: Truncate sequences longer than BERT's maximum length.
    # padding="max_length": Pad all sequences to the maximum length (512 tokens).
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def tokenize_dataset(raw_dataset):
    # Apply the tokenization function to the entire dataset.
    # batched=True: Process multiple examples at once for efficiency.
    return raw_dataset.map(tokenize_fn, batched=True)

# #optional: you can create a smaller subset of full dataset to
# #  fine-tune the model to reduce the time it takes.
# def create_subset(raw_dataset, num_samples=1000):
#     small_train_dataset = raw_dataset["train"].shuffle(seed=42).select(range(num_samples))
#     small_test_dataset = raw_dataset["test"].shuffle(seed=42).select(range(num_samples))
#     return small_train_dataset, small_test_dataset
def create_subset(raw_dataset, num_train=20000, num_val=2000):
    # Shuffle the original training dataset for reproducibility.
    shuffled_train = raw_dataset["train"].shuffle(seed=42)

    # Select non-overlapping subsets for training and validation.
    train_dataset = shuffled_train.select(range(num_train))
    val_dataset = shuffled_train.select(
        range(num_train, num_train + num_val)
    )

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Load the raw dataset from Hugging Face.
    raw_dataset = load_dataset.load_dataset_from_huggingface()
    
    # Tokenize the dataset.
    tokenized_dataset = tokenize_dataset(raw_dataset)
    
    # Print a sample from the tokenized dataset to verify the output.
    print(tokenized_dataset['train'][100])