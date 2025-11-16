import os
import gdown
import subprocess
import torch
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from architectures import EncoderDecoderArchitecture, DecoderOnlyArchitecture, EncoderOnlyArchitecture, Qwen2AudioModel
from utils import get_split_datasets, analyze_misclassified_samples


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def download_audios(destination="./dataset/audio/dementia_audios.7z"):
    # Audios download
    file_id = "1Mgv9wVvH7Az1imX3T4MYZin2-c841ksS"
    output_name = destination

    if not os.path.exists(output_name):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_name, quiet=False)

        #Unzip files
        subprocess.run(["7z", "x", output_name], check=True)


if __name__ == '__main__':
    parser = ArgumentParser("Training hyperparameters")
    parser.add_argument("--models", type=str, nargs='+', help="'enc-dec' for encoder-decoder, 'enc' for encoder-only, 'dec' for decoder-only, 'qwen' for qwen2-audio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--symbols", type=str, default="y")
    parser.add_argument("--write_mode", type=str, default='w')

    args = parser.parse_args()

    seed = args.seed
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    models_included = args.models
    add_symbols = args.symbols
    write_mode = args.write_mode

    assert add_symbols in ["y", "n"]
    assert write_mode in ['w', 'a']

    set_seed(s=seed)
    add_symbols = True if add_symbols == "y" else False

    open("./results/misclassifed_groups/percentage_misclassified_group.csv", write_mode).close()
    open("./results/misclassifed_groups/count_misclassified_group.csv", write_mode).close()
    open("./results/metrics.csv", write_mode).close()

    # Specify the instruction for fine-tuning
    instruction = 'Classify into either "control" or "dementia" the following text: '
    qwen_task_prompt = 'Classify into either "control" or "dementia" the given audio and text: {}\nAnswer: '

    train_dataset, test_dataset, eval_dataset, group_by_id = None, None, None, None
    if add_symbols:
        train_dataset, test_dataset, eval_dataset, group_by_id = get_split_datasets("./dataset/text/complete_dataset_text_level_symbols.csv", instruction, seed=seed)
        # Dataset for encoder-only models
        train_enc_only, test_enc_only, eval_enc_only, _ = get_split_datasets("./dataset/text/complete_dataset_text_level_symbols.csv", "", seed=seed)
    else:
        train_dataset, test_dataset, eval_dataset, group_by_id = get_split_datasets(
            "./dataset/text/complete_dataset_text_level_no_symbols.csv", instruction, seed=seed)
        train_enc_only, test_enc_only, eval_enc_only, _ = get_split_datasets(
            "./dataset/text/complete_dataset_text_level_no_symbols.csv", "", seed=seed)

    enc_dec_model_names = ["google-t5/t5-small", "google-t5/t5-base", "google-t5/t5-large",
                           "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
                           "google/t5gemma-s-s-prefixlm", "google/t5gemma-2b-2b-ul2", "google/t5gemma-b-b-ul2"]

    dec_only_model_names = ["google/gemma-3-270m", "google/gemma-3-4b-it", "google/gemma-2-9b-it",
                            "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-v0.2",
                            "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
                            "meta-llama/Llama-3.1-8B-Instruct",
                            "EleutherAI/gpt-neo-2.7B"]

    enc_only_model_names = ["distilbert/distilroberta-base", "distilbert/distilbert-base-uncased", "microsoft/deberta-v3-base"]

    qwen_model_names = ["Qwen/Qwen2-Audio-7B-Instruct", "Qwen/Qwen2-Audio-7B"]

    models = dict()

    if 'enc-dec' in models_included:
        models["encoder-decoder"] = enc_dec_model_names

    if 'enc' in models_included:
        models["encoder-only"] = enc_only_model_names

    if 'dec' in models_included:
        models['decoder-only'] = dec_only_model_names

    if 'qwen' in models_included:
        models['qwen'] = qwen_model_names

    for k in models.keys():
        print(f"\n+++++ {k} architectures training starts +++++")
        print("====================================")
        model_names = models[k]

        for model_name in model_names:
            model = None
            if k == "encoder-decoder":
                fp16 = True if 't5gemma' in model_name else False
                model = EncoderDecoderArchitecture(model_name=model_name, learning_rate=lr,
                                                   num_epochs=epochs, batch_size=batch_size, device="cuda:0",
                                                   train_dataset=train_dataset,
                                                   test_dataset=test_dataset,
                                                   eval_dataset=eval_dataset,
                                                   fp16=fp16, seed=seed)
            elif k == "decoder-only":
                model = DecoderOnlyArchitecture(model_name=model_name, learning_rate=lr,
                                                num_epochs=epochs, batch_size=batch_size, device="cuda:0",
                                                train_dataset=train_dataset,
                                                test_dataset=test_dataset,
                                                eval_dataset=eval_dataset,
                                                bf16=True, max_length=1024, seed=seed)
            elif k == "encoder-only":
                model = EncoderOnlyArchitecture(model_name=model_name,
                                                learning_rate=lr, num_epochs=epochs, batch_size=batch_size,
                                                device="cuda:0",
                                                train_dataset=train_enc_only,
                                                test_dataset=test_enc_only,
                                                eval_dataset=eval_enc_only,
                                                fp16=True, seed=seed, weight_decay=0.0005)

            elif k == "qwen":
                model = Qwen2AudioModel(model_name=model_name, learning_rate=lr,
                                        num_epochs=epochs, batch_size=batch_size,
                                        device="cuda:0",
                                        train_dataset=train_dataset,
                                        test_dataset=test_dataset,
                                        eval_dataset=eval_dataset,
                                        audio_path="./dataset/audio/data/DementiaBank/audio/{}/cookie/cleaned/{}.mp3",
                                        task_prompt=qwen_task_prompt,
                                        bf16=True, seed=seed, debug=False)

            model.train()
            ids, predictions, true, accuracy, precision, recall, f1 = model.predict()

            preds_df = pd.DataFrame({'id': ids, 'predictions': predictions, 'true': true})

            if len(model_name.split("/")) == 2:
                model_name = model_name.split("/")[1]

            # Save results
            preds_df.to_csv(f'./results/predictions/{model_name}_preds.csv', index=False)

            df_metrics = pd.DataFrame({"Model": [model_name], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1]})
            ms_group_percent, ms_samples_per_group, misclassified_ids = analyze_misclassified_samples(ids, true, predictions, group_by_id, model_name)

            try:
                existing = pd.read_csv("./results/metrics.csv")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
            combined = pd.concat([existing, df_metrics], ignore_index=True)
            combined.to_csv("./results/metrics.csv", index=False)

            try:
                existing = pd.read_csv("./results/misclassifed_groups/percentage_misclassified_group.csv")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
            combined = pd.concat([existing, ms_group_percent], ignore_index=True)
            combined.to_csv("./results/misclassifed_groups/percentage_misclassified_group.csv", index=False)

            try:
                existing = pd.read_csv("./results/misclassifed_groups/count_misclassified_group.csv")
            except pd.errors.EmptyDataError:
                existing = pd.DataFrame()
            combined = pd.concat([existing, ms_samples_per_group], ignore_index=True)
            combined.to_csv("./results/misclassifed_groups/count_misclassified_group.csv", index=False)

            with open(f"./results/misclassified_ids/{model_name}.txt", "w") as f:
                f.write(f"Misclassified predictions of {model_name}\n")
                for id in misclassified_ids:
                    f.write(f"{id}\n")

            print("------------------------------------")
