import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
import torch

# Define paths
dataset_path = "./datasets"
model_path = "./Codestral-22B-v0.1-exl2-6_5"  # Path to the model directory
output_dir = "./output"

# Ensure sentencepiece is installed
try:
    import sentencepiece
except ImportError:
    raise ImportError(
        "Please install sentencepiece with `pip install sentencepiece`.")

# Load datasets
train_dataset = load_dataset('arrow', data_files=os.path.join(
    dataset_path, 'Dahoas_base_code_review/train/*.arrow'))
test_dataset = load_dataset('arrow', data_files=os.path.join(
    dataset_path, 'codeparrot_apps/test/*.arrow'))

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_path, use_fast=False)  # Use the slow tokenizer

# Load model configuration and remove quantization settings
config = AutoConfig.from_pretrained(model_path)
if hasattr(config, 'quantization_config'):
    del config.quantization_config

# Load model without quantization settings
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, config=config, ignore_mismatched_sizes=True)

# Tokenize datasets


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=[
                         'input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=[
                        'input_ids', 'attention_mask', 'label'])

# Training arguments with reduced batch size and mixed precision
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
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train and evaluate
trainer.train()
trainer.evaluate()
