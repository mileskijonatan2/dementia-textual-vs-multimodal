import gc
import os
import re
import json
import torch
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import get_split_datasets, analyze_misclassified_samples
from architectures import DecoderOnlyArchitecture, EncoderDecoderArchitecture

load_dotenv()

ENC_DEC_MODEL_NAMES = [
    "google-t5/t5-small", "google-t5/t5-base", "google-t5/t5-large",
    "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
    "google/t5gemma-s-s-prefixlm", "google/t5gemma-2b-2b-ul2", "google/t5gemma-b-b-ul2",
]

DEC_ONLY_MODEL_NAMES = [
    "google/gemma-3-270m", "google/gemma-3-4b-it", "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-125m",
    "openai-community/gpt2", "openai-community/gpt2-medium",
    "openai-community/gpt2-large", "openai-community/gpt2-xl",
    "EleutherAI/gpt-j-6b", "EleutherAI/gpt-j-6B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
]


def get_model_type(model_name):
    if model_name in ENC_DEC_MODEL_NAMES:
        return "encoder-decoder"
    if model_name in DEC_ONLY_MODEL_NAMES:
        return "decoder-only"

    model_lower = model_name.lower()
    if any(x in model_lower for x in ["t5", "bart", "mbart"]):
        return "encoder-decoder"
    if any(x in model_lower for x in ["gemma", "llama", "mistral", "gpt-neo", "gpt-j", "gpt2", "falcon", "phi", "deepseek"]):
        return "decoder-only"

    print(f"Warning: Unknown model type for {model_name}, defaulting to decoder-only")
    return "decoder-only"

CORE_INSTRUCTION = """You are a neurologist. Analyze this "cookie theft" image description transcript and classify the participant as either CONTROL or DEMENTIA.

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
{}"""

ENC_DEC_ZERO_SHOT_INSTRUCTION = """Given a transcript of someone describing the "cookie theft" picture, classify it as "control" or "dementia".

Signs of dementia: frequent hesitations ("uh", "um"), long pauses, repetitions, missing key details (boy, girl, cookie jar, mother, water overflow), vague words ("thing", "stuff"), incomplete sentences.
Signs of control: coherent narrative, mentions most key elements, few disfluencies, complete sentences.

Transcript: {}

Classification:"""

REASONING_INSTRUCTION = """You are a neurologist. Analyze this "cookie theft" image description transcript and classify the participant as either CONTROL or DEMENTIA.

The scene shows: boy on stool reaching for cookie jar, girl reaching out, mother washing dishes, water overflowing.

Key evaluation criteria:
1. COHERENCE: Does the description mention key elements (boy, girl, mother, water, cookie jar) in logical order?
2. DISFLUENCIES: Count of "uh", "um", vague words like "thing" - high count suggests cognitive impairment
3. REPETITION/OMISSION: Repeated elements without new info, missing key scene elements
4. PAUSES: Long pauses (>1.5s) may indicate word-finding difficulties

Classification guidelines:
- CONTROL: Coherent description, most key elements present, few disfluencies, short pauses
- DEMENTIA: Disorganized, missing key elements, many disfluencies, frequent long pauses, repetitive

Now analyze this transcript step-by-step using EXACTLY this format (keep each line short):

OBSERVATIONS: [2-3 key features you notice in the transcript, one sentence]
DISFLUENCIES: [low / medium / high]
CLASSIFICATION: control or dementia

Transcript:
{}"""


def _is_deepseek_r1_family(model_name: str) -> bool:
    return "deepseek-r1" in model_name.lower() or "deepseek-r1-distill" in model_name.lower()


def _wrap_instruction(model_name: str, instruction: str) -> str:
    m = model_name.lower()

    if "deepseek" in m and "distill" in m and "qwen" in m:
        wrapped = "<|im_start|>user\n" + instruction + "<|im_end|>\n<|im_start|>assistant\n"
        return wrapped + "<think>\n"

    if "deepseek" in m and "distill" in m and "llama" in m:
        wrapped = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   + instruction
                   + "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
        return wrapped + "<think>\n"

    if "deepseek" in m:
        wrapped = "<|begin▁of▁sentence|><|User|>" + instruction + "<|Assistant|>"
        if _is_deepseek_r1_family(model_name):
            wrapped += "<think>\n"
        return wrapped

    if any(x in m for x in ["mistral", "mixtral", "zephyr"]):
        return "<s>[INST] " + instruction + " [/INST]</s>"

    if "llama-3" in m or "meta-llama" in m:
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            + instruction
            + "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

    elif "gemma-2" in m:
        return "<bos><start_of_turn>user\n" + instruction + "\n<end_of_turn>\n<start_of_turn>model\n"

    elif any(x in m for x in ["gpt-neo", "gpt-j", "gpt2"]):
        return instruction + "\n\nAnswer:"

    elif any(x in m for x in ["t5", "flan-t5", "bart", "mbart", "t5gemma"]):
        return "Classify as control or dementia:\n\n" + instruction

    else:
        return "### Instruction ###\n" + instruction + "\n\n### Response ###\n"


def get_prompt_template(model_name: str, method: str = "zero_shot") -> str:
    if method == "reasoning":
        return _wrap_instruction(model_name, REASONING_INSTRUCTION)
    elif method == "enc_dec_zero_shot":
        return ENC_DEC_ZERO_SHOT_INSTRUCTION
    else:
        return _wrap_instruction(model_name, CORE_INSTRUCTION)


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def apply_prompt_to_dataset(dataset, prompt_template):
    prompted_inputs = [prompt_template.format(text) for text in dataset['input_text']]
    return Dataset.from_dict({
        "input_text": prompted_inputs,
        "target_text": dataset['target_text'],
        "ids": dataset['ids']
    })


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if cleaned:
        return cleaned
    if '<think>' in text.lower():
        before = text[:text.lower().index('<think>')].strip()
        return before if before else text
    return text


def extract_label_from_tail(text: str, window_words: int = 6):
    if not text:
        return None
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return None
    tail = " ".join(words[-window_words:])
    m = re.search(r"\b(control|dementia)\b", tail, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def parse_reasoning_prediction(raw_text: str):
    text = strip_think_blocks((raw_text or "")).strip()

    match = re.search(r'CLASSIFICATION\s*[:\-]\s*(control|dementia)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    return extract_label_from_tail(text, window_words=6)


def clean_predictions(ids, predictions, true, method="zero_shot"):
    valid_labels = {'control', 'dementia'}
    valid_indices = []
    parsed_predictions = []
    invalid_examples = []

    for i, (pred, t) in enumerate(zip(predictions, true)):
        text = strip_think_blocks(pred or "")
        label = None

        if method == "reasoning":
            label = parse_reasoning_prediction(text)
        elif method == "enc_dec_zero_shot":
            m = re.search(r'CLASSIFICATION\s*[:\-]\s*(control|dementia)', text, re.IGNORECASE)
            if m:
                label = m.group(1).lower()
            else:
                label = extract_label_from_tail(text, window_words=6)
        else:
            label = extract_label_from_tail(text, window_words=6)

        if label in valid_labels:
            valid_indices.append(i)
            parsed_predictions.append(label)
        else:
            if len(invalid_examples) < 8:
                invalid_examples.append({
                    'idx': i, 'id': ids[i], 'true': t,
                    'raw': (pred or '')[:280] + ('…' if pred and len(pred) > 280 else '')
                })

    valid_ids   = [ids[i] for i in valid_indices]
    valid_preds = parsed_predictions
    valid_true  = [true[i] for i in valid_indices]

    total = len(predictions)
    valid_count = len(valid_preds)
    invalid_count = total - valid_count

    print("\n" + "=" * 80)
    print(f"PREDICTION SUMMARY  (method={method})")
    print(f"Total generations: {total}")
    print(f"Valid ('control' or 'dementia'): {valid_count}  ({valid_count/total:.1%})")
    print(f"Invalid / non-compliant: {invalid_count}  ({invalid_count/total:.1%})")

    if invalid_examples:
        print("\nFirst few non-compliant generations:")
        for ex in invalid_examples:
            print(f"  id = {ex['id']:>8}   true = {ex['true']:8}   ->  {ex['raw']!r}")
        print()

    if method == "reasoning" and valid_count > 0:
        print("Sample successful parses:")
        for j, idx in enumerate(valid_indices[:5]):
            raw_short = predictions[idx].strip()[:180]
            if len(predictions[idx].strip()) > 180:
                raw_short += " ..."
            print(f"  id={ids[idx]:>8}  parsed={valid_preds[j]:9}  raw={raw_short!r}")
        print()

    print("=" * 80 + "\n")
    return valid_ids, valid_preds, valid_true


def calculate_metrics(true, predictions):
    accuracy = accuracy_score(true, predictions)
    precision = precision_score(true, predictions, pos_label='dementia')
    recall = recall_score(true, predictions, pos_label='dementia')
    f1 = f1_score(true, predictions, pos_label='dementia')
    return accuracy, precision, recall, f1


def _save_csv_append(df_new, path):
    try:
        existing = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame()
    pd.concat([existing, df_new], ignore_index=True).to_csv(path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser("Zero-shot prompting hyperparameters")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--method", type=str, default="zero_shot",
                        choices=["zero_shot", "reasoning", "enc_dec_zero_shot"],
                        help="Prompting method (used when --methods is not given)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        choices=["zero_shot", "reasoning", "enc_dec_zero_shot"],
                        help="Run multiple prompting methods in sequence for the same model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="symbols",
                        choices=["symbols", "no_symbols", "temporal"],
                        help="Dataset type: symbols, no_symbols, or temporal")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "full", "train", "eval"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max tokens to generate. Default: auto (5 for zero_shot, 150 for reasoning)")
    parser.add_argument("--write_mode", type=str, default='a', choices=['w', 'a'])

    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch_size
    model_name = args.model
    data_type = args.data_type
    split = args.split
    max_length = args.max_length
    write_mode = args.write_mode

    methods_to_run = args.methods if args.methods else [args.method]

    set_seed(s=seed)

    if data_type == "symbols":
        dataset_path = "./dataset/text/complete_dataset_text_level_symbols.csv"
    elif data_type == "temporal":
        dataset_path = "./dataset/text/complete_dataset_temporal.csv"
    else:
        dataset_path = "./dataset/text/complete_dataset_text_level_no_symbols.csv"

    if data_type == "symbols":
        name_suffix = "_sy"
    elif data_type == "temporal":
        name_suffix = "_temporal"
    else:
        name_suffix = ""

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    os.makedirs("./results/predictions", exist_ok=True)
    os.makedirs("./results/misclassifed_groups", exist_ok=True)
    os.makedirs("./results/misclassified_ids", exist_ok=True)

    open(f"./results/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv", write_mode).close()
    open(f"./results/misclassifed_groups/count_misclassified_group{name_suffix}.csv", write_mode).close()
    open(f"./results/metrics{name_suffix}.csv", write_mode).close()

    instruction = ''
    print(f"Loading dataset from: {dataset_path}")
    raw_train, raw_test, raw_eval, group_by_id = get_split_datasets(
        dataset_path, instruction, seed=seed
    )
    print(f"Dataset sizes - Train: {len(raw_train)}, Test: {len(raw_test)}, Eval: {len(raw_eval)}")

    model_type = get_model_type(model_name)
    model_short_name = model_name.split("/")[-1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected — inference will be very slow and quantization may fail.")

    all_results = []

    print(f"\nModel   : {model_name}  ({model_type})")
    print(f"Methods : {methods_to_run}")
    print(f"Data    : {data_type}  |  Split: {split}")

    for method_idx, method in enumerate(methods_to_run, 1):
        method_tag = {
            "reasoning": "reasoning",
            "enc_dec_zero_shot": "enc_dec_zs",
        }.get(method, "zs")

        print(f"\n{'#' * 80}")
        print(f"#  [{method_idx}/{len(methods_to_run)}]  METHOD: {method}")
        print(f"{'#' * 80}\n")

        set_seed(s=seed)

        is_deepseek = "deepseek" in model_name.lower()
        if args.max_new_tokens is not None:
            max_new_tokens = args.max_new_tokens
        elif method == "reasoning":
            max_new_tokens = 1000 if is_deepseek else 150
        else:
            max_new_tokens = 1000 if is_deepseek else 5

        enc_dec_tokens = max_new_tokens if args.max_new_tokens is not None or method == "reasoning" else 6

        prompt_template = get_prompt_template(model_name, method=method)
        model_label = f"{model_short_name}_prompting_{method_tag}"

        if split == "full":
            full_data = pd.read_csv(dataset_path)
            full_dataset = Dataset.from_dict({
                "input_text": full_data['transcript'].tolist(),
                "target_text": full_data['label'].tolist(),
                "ids": full_data['id'].tolist()
            })
            full_dataset = apply_prompt_to_dataset(full_dataset, prompt_template)
            train_dataset = test_dataset = eval_dataset = full_dataset
            print(f"Using FULL dataset: {len(full_dataset)} samples")
        else:
            train_dataset = apply_prompt_to_dataset(raw_train, prompt_template)
            test_dataset  = apply_prompt_to_dataset(raw_test, prompt_template)
            eval_dataset  = apply_prompt_to_dataset(raw_eval, prompt_template)

        print(f"Initializing {model_name}  (type={model_type}, method={method})")
        if model_type == "encoder-decoder":
            fp16 = 't5gemma' in model_name.lower()
            model = EncoderDecoderArchitecture(
                model_name=model_name,
                learning_rate=1e-4,
                num_epochs=1,
                batch_size=batch_size,
                device=device,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                eval_dataset=eval_dataset,
                fp16=fp16,
                seed=seed,
                max_new_tokens=enc_dec_tokens,
            )
        else:
            extra_kwargs = {}
            if is_deepseek:
                extra_kwargs["temperature"] = 0.6
                extra_kwargs["do_sample"] = True

            model = DecoderOnlyArchitecture(
                model_name=model_name,
                learning_rate=1e-4,
                num_epochs=1,
                batch_size=batch_size,
                device=device,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                eval_dataset=eval_dataset,
                bf16=True,
                max_length=max_length,
                seed=seed,
                max_new_tokens=max_new_tokens,
                padding_side="left",
                **extra_kwargs,
            )

        predict_split = "test" if split in ["test", "full"] else split
        print(f"Zero-shot prompting ({method}) for {model_name} — skipping training.")
        print("------------------------------------")
        ids, predictions, true, _, _, _, _ = model.predict(split=predict_split)

        print(f"\nRAW MODEL GENERATIONS — first {min(10, len(predictions))} samples (full outputs)")
        print("=" * 100)
        for i in range(min(10, len(predictions))):
            full_pred = (predictions[i] or "").strip()
            print(f"#{i+1:2d}   id={ids[i]:>8}   true={true[i]:8}   ->  {full_pred!s}")
            print("----- Full output below -----")
            print(full_pred)
            print("")
        print("=" * 100)

        try:
            raw_df = pd.DataFrame({
                'id': ids,
                'raw_generation': [p if p is not None else '' for p in predictions],
                'true': true,
            })
            raw_csv_path = f"./results/predictions/{model_label}_raw{name_suffix}.csv"
            raw_df.to_csv(raw_csv_path, index=False)
            print(f"Saved raw generations to {raw_csv_path}")
        except Exception as e:
            print(f"Warning: failed to save raw generations CSV: {e}")

        original_total = len(predictions)
        ids, predictions, true = clean_predictions(ids, predictions, true, method=method)
        valid_predictions = len(predictions)
        invalid_rate = (original_total - valid_predictions) / original_total if original_total > 0 else 0.0

        print(f"After cleaning: {valid_predictions} valid, "
              f"{original_total - valid_predictions} removed")

        if valid_predictions > 0:
            accuracy, precision, recall, f1 = calculate_metrics(true, predictions)

            preds_df = pd.DataFrame({'id': ids, 'predictions': predictions, 'true': true})
            preds_df.to_csv(
                f'./results/predictions/{model_label}_preds{name_suffix}.csv', index=False
            )

            df_metrics = pd.DataFrame({
                "Model": [model_label], "Accuracy": [accuracy],
                "Precision": [precision], "Recall": [recall],
                "F1": [f1], "Invalid_Rate": [invalid_rate],
            })
            _save_csv_append(df_metrics, f"./results/metrics{name_suffix}.csv")

            ms_pct, ms_cnt, misclassified_ids = analyze_misclassified_samples(
                ids, true, predictions, group_by_id, model_label
            )
            _save_csv_append(
                ms_pct,
                f"./results/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv",
            )
            _save_csv_append(
                ms_cnt,
                f"./results/misclassifed_groups/count_misclassified_group{name_suffix}.csv",
            )

            with open(f"./results/misclassified_ids/{model_label}{name_suffix}.txt", "w") as f:
                f.write(f"Misclassified predictions of {model_label}{name_suffix}\n")
                for id_ in misclassified_ids:
                    f.write(f"{id_}\n")

            all_results.append({
                "Model": model_short_name, "Method": method_tag,
                "Accuracy": accuracy, "Precision": precision,
                "Recall": recall, "F1": f1,
                "Invalid%": f"{invalid_rate:.1%}",
                "Misclassified": len(misclassified_ids),
            })

            print(f"\n  Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
                  f"Recall={recall:.4f}  F1={f1:.4f}  "
                  f"Invalid={invalid_rate:.1%}")
        else:
            df_metrics = pd.DataFrame({
                "Model": [model_label], "Accuracy": [0.0],
                "Precision": [0.0], "Recall": [0.0],
                "F1": [0.0], "Invalid_Rate": [invalid_rate],
            })
            _save_csv_append(df_metrics, f"./results/metrics{name_suffix}.csv")

            all_results.append({
                "Model": model_short_name, "Method": method_tag,
                "Accuracy": 0.0, "Precision": 0.0,
                "Recall": 0.0, "F1": 0.0,
                "Invalid%": f"{invalid_rate:.1%}",
                "Misclassified": "N/A",
            })
            print(f"\n  WARNING: No valid predictions for {model_name} / {method}")

        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("  GPU memory released.\n")

    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    print("=" * 80)
