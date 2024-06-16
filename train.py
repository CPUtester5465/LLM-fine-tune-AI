import os
import logging
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
dataset_path = "./datasets"
model_path = "./distilbert-base-uncased"  # Path to the model directory
output_dir = "./output"

# Ensure sentencepiece is installed
try:
    import sentencepiece
except ImportError:
    raise ImportError("Please install sentencepiece with `pip install sentencepiece`.")

# Function to check if the dataset exists and list its contents
def check_and_list_dataset_files(dataset_dir):
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        logging.info(f"Contents of {dataset_dir}:")
        for file_name in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, file_name)
            logging.info(f"  {file_name} - Size: {os.path.getsize(file_path)} bytes")
        return True
    return False

# Function to inspect dataset columns
def inspect_dataset_columns(dataset):
    return dataset.column_names

# Function to download datasets
def download_datasets():
    datasets_info = [
        {"name": "sidddddddddddd/kubernetes-llama3", "path": os.path.join(dataset_path, 'kubernetes-llama3')},
        {"name": "mcipriano/stackoverflow-kubernetes-questions", "path": os.path.join(dataset_path, 'stackoverflow-kubernetes-questions')},
        {"name": "mteb/stackoverflowdupquestions-reranking", "path": os.path.join(dataset_path, 'stackoverflowdupquestions-reranking')},
        {"name": "codeparrot/github-code-clean", "path": os.path.join(dataset_path, 'github-code-clean')},
        {"name": "codeparrot/github-code", "path": os.path.join(dataset_path, 'github-code')},
        {"name": "awettig/Pile-Github-0.5B-8K-opt", "path": os.path.join(dataset_path, 'Pile-Github-0.5B-8K-opt')},
        {"name": "lewtun/github-issues", "path": os.path.join(dataset_path, 'github-issues')},
        {"name": "thomwolf/github-python", "path": os.path.join(dataset_path, 'github-python')},
        {"name": "codeparrot/github-jupyter", "path": os.path.join(dataset_path, 'github-jupyter')},
        {"name": "GShadow/github-shell", "path": os.path.join(dataset_path, 'github-shell')},
        {"name": "codeparrot/apps", "path": os.path.join(dataset_path, 'apps')},
        {"name": "deepmind/code_contests", "path": os.path.join(dataset_path, 'code_contests')},
        {"name": "sauravjoshi23/aws-documentation-chunked", "path": os.path.join(dataset_path, 'aws-documentation-chunked')}
    ]

    for dataset_info in datasets_info:
        if not check_and_list_dataset_files(dataset_info["path"]):
            logging.info(f"Downloading dataset {dataset_info['name']}...")
            try:
                dataset = load_dataset(dataset_info['name'])
                dataset.save_to_disk(dataset_info["path"])
                logging.info(f"Dataset {dataset_info['name']} downloaded and saved to {dataset_info['path']}")
            except Exception as e:
                logging.error(f"Error downloading {dataset_info['name']}: {e}")
        else:
            logging.info(f"Dataset {dataset_info['name']} already exists at {dataset_info['path']}")

download_datasets()

# Load and inspect datasets
def load_and_inspect_datasets():
    datasets = {}
    for dataset_name in os.listdir(dataset_path):
        dataset_dir = os.path.join(dataset_path, dataset_name)
        dataset = load_dataset('arrow', data_files=dataset_dir) if any(f.endswith('.arrow') for f in os.listdir(dataset_dir)) else load_from_disk(dataset_dir)
        datasets[dataset_name] = {
            "data": dataset,
            "columns": inspect_dataset_columns(dataset)
        }
    return datasets

datasets = load_and_inspect_datasets()

# Initialize tokenizer and model
logging.info("Initializing tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # Use the slow tokenizer

# Load model configuration and remove quantization settings
config = AutoConfig.from_pretrained(model_path)
if hasattr(config, 'quantization_config'):
    del config.quantization_config

# Load model without quantization settings
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)

# Tokenize datasets
def tokenize_and_format_datasets(datasets):
    logging.info("Tokenizing and formatting datasets...")
    for dataset_name, dataset_info in datasets.items():
        columns = dataset_info["columns"]
        text_key = None
        label_key = None

        # Determine text and label keys based on column inspection
        if 'Question' in columns:
            text_key = 'Question'
            label_key = 'Answer'
        elif 'query' in columns:
            text_key = 'query'
            label_key = 'positive'
        else:
            # Default to the first text column and label column if not found
            text_key = columns[0]
            label_key = columns[-1]

        if text_key and label_key:
            def tokenize_function(examples):
                tokenized_inputs = tokenizer(examples[text_key], padding="max_length", truncation=True)
                tokenized_inputs["label"] = examples[label_key]
                return tokenized_inputs

            dataset_info["data"] = dataset_info["data"].map(tokenize_function, batched=True)
            dataset_info["data"].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

tokenize_and_format_datasets(datasets)

# Combine datasets into training and evaluation sets
logging.info("Combining datasets into training and evaluation sets...")
train_dataset = torch.utils.data.ConcatDataset([dataset_info["data"]['train'] for dataset_info in datasets.values() if 'train' in dataset_info["data"]])
eval_dataset = torch.utils.data.ConcatDataset([dataset_info["data"]['test'] for dataset_info in datasets.values() if 'test' in dataset_info["data"]])

# Training arguments with reduced batch size and mixed precision
logging.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size
    per_device_eval_batch_size=1,   # Reduced batch size
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision training
)

# Trainer
logging.info("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train and evaluate
logging.info("Starting training...")
trainer.train()
logging.info("Training completed.")

logging.info("Starting evaluation...")
trainer.evaluate()
logging.info("Evaluation completed.")