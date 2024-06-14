# LLM Fine-Tune AI

This repository contains scripts and resources for fine-tuning the `Codestral-22B-v0.1` model on custom datasets for multiple-label classification tasks.

## Model

`Codestral-22B-v0.1` is trained on a diverse dataset of 80+ programming languages, including the most popular ones, such as Python, Java, C, C++, JavaScript, and Bash. The model can be queried:
- To answer any questions about a code snippet (write documentation, explain, factorize)
- To generate code following specific instructions

## Datasets

The training and test datasets used in this project are:
- `Dahoas_base_code_review`
- `codeparrot_apps`

## Requirements

To run the scripts in this repository, you need the following libraries:

plaintext
torch>=1.9.0
transformers>=4.10.0
datasets>=1.11.0
sentencepiece
huggingface-cli


```
pip install -r requirements.txt
```

Download Model
```
huggingface-cli snapshot-download Codestral-22B-v0.1-exl2-6_5 -d ./Codestral-22B-v0.1-exl2-6_5
```
Download Datasets

For Dahoas_base_code_review:
```
huggingface-cli dataset download --name Dahoas_base_code_review -d ./datasets/Dahoas_base_code_review
```
For codeparrot_apps:
``
huggingface-cli dataset download --name codeparrot_apps -d ./datasets/codeparrot_apps
``