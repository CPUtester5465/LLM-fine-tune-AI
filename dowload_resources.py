import asyncio
import aiohttp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from huggingface_hub import HfFolder
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def download_model(model_name, model_path, token, session):
    loop = asyncio.get_event_loop()
    try:
        if os.path.exists(model_path):
            logger.info(f"Model {model_name} already exists at {model_path}, skipping download.")
            return
        
        # Download tokenizer
        tokenizer = await loop.run_in_executor(None, partial(AutoTokenizer.from_pretrained, model_name, token=token, trust_remote_code=True))
        tokenizer.save_pretrained(model_path)
        
        # Download model
        model = await loop.run_in_executor(None, partial(AutoModelForSequenceClassification.from_pretrained, model_name, token=token, trust_remote_code=True))
        model.save_pretrained(model_path)
        logger.info(f"Model {model_name} downloaded and saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")

async def download_dataset(dataset_name, dataset_path, token, session, retries=3):
    loop = asyncio.get_event_loop()
    try:
        if os.path.exists(dataset_path):
            logger.info(f"Dataset {dataset_name} already exists at {dataset_path}, skipping download.")
            return
        
        # Download dataset
        dataset = await loop.run_in_executor(None, partial(load_dataset, dataset_name, token=token, trust_remote_code=True))
        await loop.run_in_executor(None, dataset.save_to_disk, dataset_path)
        logger.info(f"Dataset {dataset_name} downloaded and saved to {dataset_path}")
    except Exception as e:
        if retries > 0:
            logger.warning(f"Failed to download dataset {dataset_name}, retrying... ({retries} retries left)")
            await download_dataset(dataset_name, dataset_path, token, session, retries - 1)
        else:
            logger.error(f"Failed to download dataset: {e}")

async def main():
    # Define model and dataset names and paths
    model_name = "hikinegi/Llama-2-7b-chat-hf_kubernetes_tuned"
    model_path = "./Llama-2-7b-chat-hf_kubernetes_tuned"
    
    dataset_names_paths = {
        "sidddddddddddd/kubernetes-llama3": "./datasets/kubernetes-llama3",
        "mcipriano/stackoverflow-kubernetes-questions": "./datasets/stackoverflow-kubernetes-questions",
        "mteb/stackoverflowdupquestions-reranking": "./datasets/stackoverflowdupquestions-reranking",
        "codeparrot/github-code-clean": "./datasets/github-code-clean",
        "codeparrot/github-code": "./datasets/github-code",
        "awettig/Pile-Github-0.5B-8K-opt": "./datasets/Pile-Github-0.5B-8K-opt",
        "lewtun/github-issues": "./datasets/github-issues",
        "thomwolf/github-python": "./datasets/github-python",
        "codeparrot/github-jupyter": "./datasets/github-jupyter",
        "GShadow/github-shell": "./datasets/github-shell",
        "codeparrot/apps": "./datasets/apps",
        "deepmind/code_contests": "./datasets/code_contests",
        "sauravjoshi23/aws-documentation-chunked": "./datasets/aws-documentation-chunked"
    }
    
    # Use the provided token
    token = "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"
    
    async with aiohttp.ClientSession() as session:
        # Download model
        await download_model(model_name, model_path, token, session)
        
        # Download datasets
        download_tasks = [
            download_dataset(dataset_name, dataset_path, token, session)
            for dataset_name, dataset_path in dataset_names_paths.items()
        ]
        await asyncio.gather(*download_tasks)

if __name__ == "__main__":
    asyncio.run(main())