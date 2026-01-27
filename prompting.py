# -*- coding: utf-8 -*-
"""
Zero-Shot Prompting for Dementia Classification

This script performs zero-shot prompting experiments on the dementia classification task
using the "cookie theft" picture description dataset.

Key Features:
- Uses the same data loading and model architecture as train.py
- Directly comparable results with fine-tuned models
- Supports decoder-only models (Mistral, LLaMA, GPT-Neo, Gemma, etc.)
- Supports encoder-decoder models (T5, FLAN-T5, etc.)

Usage (from within the dementia-jonatan directory):
    python prompting.py --model mistralai/Mistral-7B-Instruct-v0.3 --data_type symbols --split test
    python prompting.py --model google/flan-t5-base --data_type temporal --split test
"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import get_split_datasets, analyze_misclassified_samples
from architectures import DecoderOnlyArchitecture, EncoderDecoderArchitecture

load_dotenv()


# Model type lists (same as train.py)
ENC_DEC_MODEL_NAMES = ["google-t5/t5-small", "google-t5/t5-base", "google-t5/t5-large",
                       "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
                       "google/t5gemma-s-s-prefixlm", "google/t5gemma-2b-2b-ul2", "google/t5gemma-b-b-ul2"]

DEC_ONLY_MODEL_NAMES = ["google/gemma-3-270m", "google/gemma-3-4b-it", "google/gemma-2-9b-it",
                        "mistralai/Mistral-7B-Instruct-v0.3",
                        "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
                        "meta-llama/Llama-3.1-8B-Instruct",
                        "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-125m",
                        "openai-community/gpt2", "openai-community/gpt2-medium",
                        "openai-community/gpt2-large", "openai-community/gpt2-xl"]


def get_model_type(model_name):
    """Determine model type based on model name."""
    # Check exact matches first
    if model_name in ENC_DEC_MODEL_NAMES:
        return "encoder-decoder"
    if model_name in DEC_ONLY_MODEL_NAMES:
        return "decoder-only"
    
    # Check partial matches for flexibility
    model_lower = model_name.lower()
    if any(x in model_lower for x in ["t5", "bart", "mbart"]):
        return "encoder-decoder"
    if any(x in model_lower for x in ["gemma", "llama", "mistral", "gpt-neo", "gpt2", "falcon", "phi"]):
        return "decoder-only"
    
    # Default to decoder-only
    print(f"Warning: Unknown model type for {model_name}, defaulting to decoder-only")
    return "decoder-only"


# Default prompt template for decoder-only models (uses instruction format)
DECODER_PROMPT = '''<s>[INST] You are a neurologist. Analyze this "cookie theft" image description transcript and classify the participant as either CONTROL or DEMENTIA.

The scene shows: boy on stool reaching for cookie jar, girl reaching out, mother washing dishes, water overflowing.

Key evaluation criteria:
1. COHERENCE: Does the description mention key elements (boy, girl, mother, water, cookie jar) in logical order?
2. DISFLUENCIES: Count of "uh", "um", vague words like "thing" - high count suggests cognitive impairment
3. REPETITION/OMISSION: Repeated elements without new info, missing key scene elements
4. PAUSES: Long pauses (>1.5s) may indicate word-finding difficulties

Classification guidelines:
- CONTROL: Coherent description, most key elements present, few disfluencies, short pauses
- DEMENTIA: Disorganized, missing key elements, many disfluencies, frequent long pauses, repetitive

Now classify this transcript. Respond with only: control or dementia

Transcript:
{}[/INST]</s>'''

# Prompt template for encoder-decoder models (simpler format, no chat template)
ENCODER_DECODER_PROMPT = '''Classify this transcript as "control" or "dementia".

Context: The speaker is describing a "cookie theft" picture showing a boy on a stool reaching for cookies, a girl reaching out, and a mother washing dishes with water overflowing.

Signs of dementia: disorganized speech, missing key elements, disfluencies (uh, um), repetition, long pauses.
Signs of control: coherent description, mentions key elements, few disfluencies.

Transcript: {}

Classification:'''


def set_seed(s=42):
    """Set random seed for reproducibility - same as train.py"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def apply_prompt_to_dataset(dataset, prompt_template):
    """
    Apply the prompt template to each input in the dataset.
    This wraps the transcript in the prompting format before passing to the model.
    
    :param dataset: HuggingFace Dataset with 'input_text', 'target_text', 'ids' columns
    :param prompt_template: Prompt template with {} placeholder for transcript
    :return: New Dataset with prompts applied to input_text
    """
    prompted_inputs = [prompt_template.format(text) for text in dataset['input_text']]
    return Dataset.from_dict({
        "input_text": prompted_inputs,
        "target_text": dataset['target_text'],
        "ids": dataset['ids']
    })


def clean_predictions(ids, predictions, true):
    """
    Filter out invalid predictions, keeping only 'control' and 'dementia'.
    
    :param ids: List of sample IDs
    :param predictions: List of model predictions
    :param true: List of true labels
    :return: Cleaned ids, predictions, true labels
    """
    valid_labels = {'control', 'dementia'}
    valid_indices = [i for i, p in enumerate(predictions) if p in valid_labels]
    
    predictions_clean = [predictions[i] for i in valid_indices]
    true_clean = [true[i] for i in valid_indices]
    ids_clean = [ids[i] for i in valid_indices]
    
    return ids_clean, predictions_clean, true_clean


def calculate_metrics(true, predictions):
    """
    Calculate classification metrics.
    
    :param true: List of true labels
    :param predictions: List of predicted labels
    :return: accuracy, precision, recall, f1
    """
    accuracy = accuracy_score(true, predictions)
    precision = precision_score(true, predictions, pos_label='dementia')
    recall = recall_score(true, predictions, pos_label='dementia')
    f1 = f1_score(true, predictions, pos_label='dementia')
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = ArgumentParser("Zero-shot prompting hyperparameters")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="symbols", choices=["symbols", "no_symbols", "temporal"],
                        help="Dataset type: symbols, no_symbols, or temporal")
    parser.add_argument("--split", type=str, default="test", choices=["test", "full", "train", "eval"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--write_mode", type=str, default='a', choices=['w', 'a'])
    
    args = parser.parse_args()
    
    seed = args.seed
    batch_size = args.batch_size
    model_name = args.model
    data_type = args.data_type
    split = args.split
    max_length = args.max_length
    write_mode = args.write_mode
    
    set_seed(s=seed)
    
    # Set name suffix based on data type
    if data_type == "symbols":
        name_suffix = "_sy"
    elif data_type == "temporal":
        name_suffix = "_temporal"
    else:
        name_suffix = ""
    
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    
    open(f"./results/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv", write_mode).close()
    open(f"./results/misclassifed_groups/count_misclassified_group{name_suffix}.csv", write_mode).close()
    open(f"./results/metrics{name_suffix}.csv", write_mode).close()
    
    # Use empty instruction since we apply our own prompt template
    instruction = ''
    
    # Load datasets using the SAME function as train.py
    if data_type == "symbols":
        dataset_path = "./dataset/text/complete_dataset_text_level_symbols.csv"
    elif data_type == "temporal":
        dataset_path = "./dataset/text/complete_dataset_temporal.csv"
    else:  # no_symbols
        dataset_path = "./dataset/text/complete_dataset_text_level_no_symbols.csv"
    
    print(f"Loading dataset from: {dataset_path}")
    train_dataset, test_dataset, eval_dataset, group_by_id = get_split_datasets(
        dataset_path, instruction, seed=seed
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}, Eval: {len(eval_dataset)}")
    
    # Handle "full" split by combining all splits
    if split == "full":
        # Load full dataset
        full_data = pd.read_csv(dataset_path)
        full_dataset = Dataset.from_dict({
            "input_text": full_data['transcript'].tolist(),
            "target_text": full_data['label'].tolist(),
            "ids": full_data['id'].tolist()
        })
        # Apply prompt to full dataset (prompt selected based on model type)
        model_type = get_model_type(model_name)
        prompt_template = ENCODER_DECODER_PROMPT if model_type == "encoder-decoder" else DECODER_PROMPT
        full_dataset = apply_prompt_to_dataset(full_dataset, prompt_template)
        train_dataset = full_dataset
        test_dataset = full_dataset
        eval_dataset = full_dataset
        print(f"Using FULL dataset: {len(full_dataset)} samples")
    else:
        # Determine model type and select appropriate prompt
        model_type = get_model_type(model_name)
        prompt_template = ENCODER_DECODER_PROMPT if model_type == "encoder-decoder" else DECODER_PROMPT
        
        # Apply prompt template to datasets
        train_dataset = apply_prompt_to_dataset(train_dataset, prompt_template)
        test_dataset = apply_prompt_to_dataset(test_dataset, prompt_template)
        eval_dataset = apply_prompt_to_dataset(eval_dataset, prompt_template)
    
    print(f"Model type: {model_type}")
    print(f"\nInitializing model: {model_name}")
    
    # Initialize model based on architecture type
    if model_type == "encoder-decoder":
        fp16 = 't5gemma' in model_name.lower()
        model = EncoderDecoderArchitecture(
            model_name=model_name,
            learning_rate=1e-4,  # Not used for prompting, but required by interface
            num_epochs=1,        # Not used for prompting
            batch_size=batch_size,
            device="cuda:0",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            fp16=fp16,
            seed=seed,
        )
    else:  # decoder-only
        model = DecoderOnlyArchitecture(
            model_name=model_name,
            learning_rate=1e-4,  # Not used for prompting, but required by interface
            num_epochs=1,        # Not used for prompting
            batch_size=batch_size,
            device="cuda:0",
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            bf16=True,
            max_length=max_length,
            seed=seed,
        )
    
    # Skip training - we're doing zero-shot prompting
    print(f"Zero-shot prompting mode for {model_name} - skipping training.")
    print("------------------------------------")
    
    # Predict on specified split
    predict_split = "test" if split in ["test", "full"] else split
    ids, predictions, true, _, _, _, _ = model.predict(split=predict_split)
    
    # Get model short name (same as train.py)
    if len(model_name.split("/")) == 2:
        model_short_name = model_name.split("/")[1]
    else:
        model_short_name = model_name
    
    # Clean predictions (filter out invalid responses)
    total_predictions = len(predictions)
    ids, predictions, true = clean_predictions(ids, predictions, true)
    valid_predictions = len(predictions)
    
    print(f"Total predictions: {total_predictions}")
    print(f"Valid predictions: {valid_predictions}")
    print(f"Invalid predictions removed: {total_predictions - valid_predictions}")
    
    # Check if we have any valid predictions
    if valid_predictions == 0:
        print("\nERROR: No valid predictions! The model did not output 'control' or 'dementia'.")
        print("This model may be too small or not instruction-tuned for this task.")
        del model
        torch.cuda.empty_cache()
        exit(1)
    
    # Calculate metrics on cleaned predictions
    accuracy, precision, recall, f1 = calculate_metrics(true, predictions)
    
    # Save predictions (same format as train.py)
    preds_df = pd.DataFrame({'id': ids, 'predictions': predictions, 'true': true})
    preds_df.to_csv(f'./results/predictions/{model_short_name}_prompting_preds{name_suffix}.csv', index=False)
    
    # Save metrics (same format as train.py)
    df_metrics = pd.DataFrame({
        "Model": [f"{model_short_name}_prompting"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1": [f1]
    })
    
    try:
        existing = pd.read_csv(f"./results/metrics{name_suffix}.csv")
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame()
    combined = pd.concat([existing, df_metrics], ignore_index=True)
    combined.to_csv(f"./results/metrics{name_suffix}.csv", index=False)
    
    # Analyze misclassified samples (same as train.py)
    ms_group_percent, ms_samples_per_group, misclassified_ids = analyze_misclassified_samples(
        ids, true, predictions, group_by_id, f"{model_short_name}_prompting"
    )
    
    # Save misclassification group percentages (same as train.py)
    try:
        existing = pd.read_csv(f"./results/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv")
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame()
    combined = pd.concat([existing, ms_group_percent], ignore_index=True)
    combined.to_csv(f"./results/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv", index=False)
    
    # Save misclassification group counts (same as train.py)
    try:
        existing = pd.read_csv(f"./results/misclassifed_groups/count_misclassified_group{name_suffix}.csv")
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame()
    combined = pd.concat([existing, ms_samples_per_group], ignore_index=True)
    combined.to_csv(f"./results/misclassifed_groups/count_misclassified_group{name_suffix}.csv", index=False)
    
    # Save misclassified IDs (same as train.py)
    with open(f"./results/misclassified_ids/{model_short_name}_prompting{name_suffix}.txt", "w") as f:
        f.write(f"Misclassified predictions of {model_short_name}_prompting{name_suffix}\n")
        for id in misclassified_ids:
            f.write(f"{id}\n")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Split: {split}")
    print(f"Samples: {len(ids)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Misclassified: {len(misclassified_ids)}")
    print("=" * 60)
    
    print("------------------------------------")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
