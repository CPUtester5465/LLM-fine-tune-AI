import os
from datasets import load_from_disk

# Define paths
dataset_path = "./datasets"
train_data_path = os.path.join(dataset_path, 'stackoverflow-kubernetes-questions/train')
test_data_path = os.path.join(dataset_path, 'stackoverflowdupquestions-reranking/test')

# Load datasets from disk
train_dataset = load_from_disk(train_data_path)
test_dataset = load_from_disk(test_data_path)

# Print dataset column names
print("Train dataset columns:", train_dataset.column_names)
print("Test dataset columns:", test_dataset.column_names)