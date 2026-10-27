from datasets import load_dataset

def load_dataset_from_huggingface():

    dataset = load_dataset("Yelp/yelp_review_full")
    return dataset
 


if __name__ == "__main__":
    print(load_dataset_from_huggingface()['train'][100])