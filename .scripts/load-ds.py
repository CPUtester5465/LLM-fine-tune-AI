import os
import json
from datetime import datetime
from transformers import pipeline
from datasets import load_from_disk, DatasetDict
from io import StringIO

# Load the classification model
model_name = "Cielciel/aift-model-review-multiple-label-classification"
classifier = pipeline("zero-shot-classification", model=model_name)

# Function to read prompt from file


def read_prompt(prompt_path):
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

# Function to read schema from file


def read_schema(schema_path):
    with open(schema_path, "r") as file:
        schema = json.load(file)
    return schema

# Function to classify text


def classify_text(prompt, candidate_labels):
    return classifier(prompt, candidate_labels)

# Function to analyze the dataset


def analyze_dataset(dataset: DatasetDict, log_file):
    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert dataset to DataFrame
    df = pd.DataFrame(dataset['train'])

    # Display basic dataset information
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    with open(log_file, 'a') as f:
        f.write("\nDataset Info:\n")
        f.write(info_str)
        f.write("\n\nDataset Head:\n")
        f.write(df.head().to_string())
        f.write("\n\n")

    # Plot some visualizations
    if 'difficulty' in df.columns:
        plt.figure()
        df['difficulty'].value_counts().plot(kind='bar')
        plt.title('Difficulty Distribution')
        plt.xlabel('Difficulty')
        plt.ylabel('Count')
        plt.savefig(f"{log_file}_difficulty_distribution.png")

    # Additional analysis based on dataset structure
    if 'problem_id' in df.columns:
        with open(log_file, 'a') as f:
            f.write("\nProblem ID Statistics:\n")
            f.write(df['problem_id'].describe().to_string())
            f.write("\n\n")

    if 'question' in df.columns:
        with open(log_file, 'a') as f:
            f.write("\nSample Questions:\n")
            f.write(df['question'].head().to_string())
            f.write("\n\n")

# Main function to process all datasets in the folder


def process_datasets(datasets_path, ds_json_path):
    candidate_labels = ["description", "structure", "contents", "metadata",
                        "instructions", "preprocessing", "normalization", "analysis", "function"]

    # Read existing ds.json file
    if os.path.exists(ds_json_path):
        with open(ds_json_path, 'r') as f:
            ds_status = json.load(f)
    else:
        ds_status = {}

    for dataset_name in os.listdir(datasets_path):
        dataset_path = os.path.join(datasets_path, dataset_name)
        if os.path.isdir(dataset_path):
            log_file = os.path.join(
                dataset_path, f"{dataset_name}_analysis_log.txt")

            with open(log_file, 'a') as f:
                f.write(f"\nProcessing dataset: {dataset_name}\n")

            # Debug: print the dataset path
            print(f"Dataset path: {dataset_path}")

            # Check if the dataset is available on disk
            try:
                dataset = load_from_disk(dataset_path)
            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write(
                        f"Failed to load dataset from {dataset_path}: {e}\n")
                continue

            # Read prompt and schema
            prompt_path = os.path.join(dataset_path, "prompt.txt")
            schema_path = os.path.join(dataset_path, "schema.json")

            # Debug: check if prompt and schema files exist
            if not os.path.exists(prompt_path):
                with open(log_file, 'a') as f:
                    f.write(f"Prompt file not found: {prompt_path}\n")
                continue
            if not os.path.exists(schema_path):
                with open(log_file, 'a') as f:
                    f.write(f"Schema file not found: {schema_path}\n")
                continue

            prompt = read_prompt(prompt_path)
            schema = read_schema(schema_path)

            # Classify the prompt
            classification = classify_text(prompt, candidate_labels)

            # Construct description based on classification
            description = f"Description based on classification: {classification}"

            # Print the generated description
            with open(log_file, 'a') as f:
                f.write("Generated Description and Instructions:\n")
                f.write(description)
                f.write("\n\n")

            # Print the schema for reference
            with open(log_file, 'a') as f:
                f.write("\nSchema:\n")
                f.write(json.dumps(schema, indent=4))
                f.write("\n\n")

            # Analyze the dataset
            analyze_dataset(dataset, log_file)

            # Normalize dataset names for ds.json keys
            normalized_name = dataset_name.replace('_', '/')
            ds_status[normalized_name] = {
                "status": "analyzed",
                "schema": True,
                "prompt": True,
                "date_analyzed": datetime.now().isoformat()
            }

            # Print update confirmation
            print(f"Updated ds.json for dataset: {dataset_name}")

    # Write the updated ds.json file
    with open(ds_json_path, 'w') as f:
        json.dump(ds_status, f, indent=4)

    # Verify the contents of ds.json
    with open(ds_json_path, 'r') as f:
        print("Updated ds.json:")
        print(f.read())


# Path to the datasets folder
datasets_path = "datasets"
# Path to the ds.json file
ds_json_path = "ds.json"

# Process all datasets in the folder
process_datasets(datasets_path, ds_json_path)
